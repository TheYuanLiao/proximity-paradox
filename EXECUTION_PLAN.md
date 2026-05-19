# Technical Execution Plan

**Project**: The Proximity Paradox — GPS to Counterfactual Segregation Pipeline  
**Scope**: Full implementation of Steps 0–10 (mobility processing + baseline simulation S1)  
**Reference implementation**: [`geo-social-mixing`](https://github.com/MobiSegInsights/geo-social-mixing) `src/data/`  
**First country**: Sweden (Malmö for validation, then Stockholm + Gothenburg)

---

## Implementation Phases

The pipeline has two parts with a clear dependency boundary:

- **Part A (Steps 0–6)**: Mobility processing — adapts proven code from `geo-social-mixing`
- **Part B (Steps 7–10)**: Counterfactual simulation — new code, the research contribution

Each step reads from the previous step's output directory and writes to its own. All I/O uses Parquet (tabular) or GeoParquet (spatial).

---

## Phase 0: Infrastructure

### 0.1 Container environment

Replicate the `geo-social-mixing` devcontainer with modifications for this project.

**Container stack** (`.devcontainer/`):
- Base: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
- Python 3.11 via Micromamba (`geoenv` environment)
- Java 17 (PySpark) + Java 21 (r5r, set as default)
- R with `r5r`, `data.table`, `sf`
- System libs: GDAL, GEOS, PROJ, spatialindex

**Key Python packages**:
| Package | Purpose |
|---------|---------|
| `pyspark` | Parallelized stop detection (Step 1) |
| `infostop` | GPS clustering algorithm (Step 1) |
| `HoWDe` | Home-work classification (Step 2) |
| `geopandas` + `shapely` | All spatial operations |
| `overturemaps` | Building footprint download (Step 3) |
| `r5py` / `r5r` | Isochrone computation (Step 7) |
| `scipy.spatial.cKDTree` | Fast distance queries (Steps 6, 8) |
| `pydantic` | Config validation |
| `click` | CLI entry points |

**Action items**:
- [x] Create `.devcontainer/Dockerfile`, `environment.yml`, `devcontainer.json`
- [x] Create `requirements.txt` (pip fallback)
- [ ] Vendor `infostop` as git submodule (same as geo-social-mixing)
- [ ] Test container build, verify PySpark + r5r both work
- [ ] Set up data mount paths in `devcontainer.json` for target machine

### 0.2 Config system

**Already created**:
- `config/default.yaml` — all pipeline parameters with defaults
- `config/countries/sweden.yaml` — Sweden-specific: cities, socioeconomic tier, CRS, data paths
- `config/poi_categories.yaml` — 12 unified categories (from geo-social-mixing)
- `config/schema.py` — Pydantic validation models

**To implement**:
- [ ] `load_merged_config()` — merge defaults with country overrides, CLI args override both
- [ ] Environment variable override support (for supercomputer jobs)

### 0.3 Data directory bootstrapping

Each country needs:
```
dbs/{country}/
├── raw_gps/          # Input (read-only)
├── stops/            # Step 1
├── home_work/        # Step 2
├── buildings/        # Step 3 input (cached)
├── home_buildings/   # Step 3
├── zones/            # Step 4a
├── socioeconomic/    # Step 4b
├── home_zones/       # Step 5
├── poi/              # Step 6 input
├── poi_assignment/   # Step 6
├── osm_network/      # Step 7 input
├── isochrones/       # Step 7
└── simulation/       # Steps 8–10
    ├── rewired_trips/
    ├── venue_visitors/
    ├── segregation/
    └── localizability/
```

- [ ] Add directory creation to `pipeline.py` orchestrator
- [ ] Validate input data existence before each step

---

## Phase 1: Steps 0–2 — Raw GPS to Home/Work Locations

### Step 0: Raw GPS to device-grouped Parquet (`src/device_logging.py`)

**Status**: Implemented.

**Problem**: Raw GPS data arrives as daily compressed archives (`.csv.gz` or `.zip` containing CSV files) with ~16 columns including PII (IP, user agent, device model). The downstream stop detection pipeline (Step 1) processes data per device group via PySpark `groupBy("device_aid").applyInPandas()`. Loading the full dataset into memory is infeasible on HPC — devices must be pre-partitioned into persistent groups so Step 1 loads one group at a time.

**Raw input schema** (TSV):
```
timestamp  device_aid  device_aid_type  latitude  longitude  horizontal_accuracy
altitude  altitude_accuracy  location_method  ip  user_agent  OS  OS_version
manufacturer  model  carrier
```

**Output schema** (Parquet):
```
timestamp (int64), device_aid (string), latitude (float64),
longitude (float64), location_method (string), grp (int32)
```

**Output layout**: `{output_dir}/format_parquet/grp_{0..N-1}/{date_label}.parquet`

**Device grouping strategy**:
```
grp = int(md5(device_aid).hexdigest(), 16) % n_groups
```
- **Deterministic**: Same device always maps to the same group across all days and runs.
- **Persistent**: Uses MD5 (not Python's `hash()` which is randomized via `PYTHONHASHSEED`).
- **Uniform**: MD5 distributes evenly across groups; no group gets disproportionately large.
- **Single-pass**: No global device scan needed — group assignment is computed inline per record.

**Scale reference**:
| Country | Duration | Groups | Rationale |
|---------|----------|--------|-----------|
| Sweden | 1 year | 50 | ~365 daily files, moderate device count |
| Germany | >1 year | 300 | Higher device volume, longer observation |

**Processing flow**:
1. Discover all `.csv.gz` / `.zip` files in raw directory (recursive, sorted)
2. For each daily file:
   - Read TSV, keep only pipeline columns (strip PII)
   - Drop records with null device_aid/lat/lon
   - Compute `grp = md5(device_aid) % n_groups`
   - Group by `grp`, write one Parquet file per group per day
3. Write `manifest.json` with processing statistics

**Key features**:
- `--resume`: Skip files whose output parquet already exists (crash recovery)
- `--dry-run`: Scan and report device/record counts without writing
- `manifest.json`: Records total devices, records per group, processing time
- Handles both `.csv.gz` and `.zip` input formats
- No full-dataset shuffle — each daily file is processed independently

**CLI**:
```
python src/device_logging.py --raw-dir /data/raw_gps/SE/2024 \
                             --output-dir /data/dbs/sweden \
                             --n-groups 50

python src/device_logging.py --raw-dir /data/raw_gps/DE \
                             --output-dir /data/dbs/germany \
                             --n-groups 300 --resume
```

---

### Step 1: Stop detection (`src/stop_detection.py`)

**Adapt from**: `geo-social-mixing/src/data/stop_detection.py` (448 lines)

**Architecture** (matches reference):
```python
class StopDetection:
    def __init__(self, spark, config: dict)
    def process_batch(self, batch: int) -> None
    def process_all(self, start: int, end: int) -> None
    def merge_all(self, output_file: str) -> None
```

**Key implementation details from reference**:
- Infostop parameters: `R1=30, R2=30, MIN_STAYING_TIME=15, MAX_TIME_BETWEEN=3`
- PySpark `groupBy("device_aid").applyInPandas(infostop_per_user, schema)`
- UTC offset correction per country (Sweden: +3600s)
- Output columns: `device_aid, stop_id, latitude, longitude, arrival_time, departure_time, duration_min`
- Batch-level parquet output, then merge step

**Portability changes from reference**:
- Replace hardcoded Sweden paths with config-driven paths
- Replace hardcoded UTC offset with `config.utc_offset_seconds`
- Parameterize batch count from `config.raw_gps.n_groups`
- Use `pathlib.Path` for all I/O

**CLI**:
```
python src/stop_detection.py --country sweden --batch 0        # one batch
python src/stop_detection.py --country sweden --batch 0 10     # range
python src/stop_detection.py --country sweden --merge          # merge
python src/stop_detection.py --country sweden --all            # all + merge
```

**Validation**: median ~20–50 stops/device, total stop count consistent with GPS coverage

---

### Step 2: Home/work detection (`src/home_work_detection.py`)

**Adapt from**: `geo-social-mixing/src/data/home_work_detection.py` (627 lines)

**Architecture**:
```python
class HomeWorkDetection:
    def __init__(self, spark, config: dict)
    def process_batch(self, batch: int) -> None
    def process_all(self, start: int, end: int) -> None
    def merge_all(self, output_file: str) -> None
    def extract_home_work_locations(self) -> None
```

**Key implementation details from reference**:
- HoWDe library with country-specific parameters
- Input column renaming: `device_aid→uid, localtime→start_time, l_localtime→end_time`
- Batch processing matches stop detection batches
- Final output: one row per device with primary home/work lat/lon
- Home detection rate ~71%

**Portability changes**:
- HoWDe country parameter from config
- Night hours, window sizes from config
- CRS handling per country

**Output columns**: `device_aid, home_latitude, home_longitude, home_grid_id, work_latitude, work_longitude, n_home_days, n_work_days`

**Validation**: ~71% home detection rate, home locations cluster in residential areas

---

## Phase 2: Steps 3–5 — Spatial Context

### Step 3: Home-building linkage (`src/link_home_buildings.py`)

**Adapt from**: `geo-social-mixing/src/data/link_home_buildings.py` (334 lines)

**Implementation**:
1. Load home locations from Step 2
2. Download building footprints from Overture Maps (cache locally)
   - Split download by geographic chunks if country is large
   - Filter to residential buildings only
3. Buffer buildings by 50m (in projected CRS)
4. Spatial join: homes ∩ buffered buildings
5. Flag `has_building = True/False`

**Key details from reference**:
- Overture Maps download via `overturemaps` Python package
- Sweden split into 3 regional bounding boxes (south/central/north)
- Buffer operation in projected CRS (SWEREF99 TM for Sweden)
- `gpd.sjoin()` with `predicate='intersects'` on buffered geometries
- Devices without building match retain raw home coordinate

**Portability**:
- Country bounding box from config
- Projected CRS from config (`crs_projected`)
- Building source configurable (Overture Maps vs. local file)

**Output**: adds `building_id, building_latitude, building_longitude, has_building` to home records

**Validation**: >90% building match in urban areas

---

### Step 4a: Zone harmonization (`src/prepare_zones.py`)

**Adapt from**: `geo-social-mixing/src/data/harmonize_deso.py` (295 lines)

**Implementation per country**:

| Country | Input | Processing |
|---------|-------|------------|
| Sweden | `DeSO_2025.gpkg` + SCB CSVs | Pivot demographics, join income + car ownership |
| USA | Census shapefiles + ACS | Download via `census` package |
| Germany | Destatis boundaries | Country-specific loader |
| Brazil | IBGE setores | Country-specific loader |

**Interface**:
```python
def prepare_zones(country: str, config: dict) -> gpd.GeoDataFrame:
    """Returns GeoDataFrame with: zone_id, geometry, zone_type, population, ..."""
```

**Sweden-specific (from reference)**:
- Join geometry (`desokod`) + birth background + income + car ownership
- Pivot birth background from long→wide: `birth_sweden, birth_europe, birth_other`
- Income percentiles: Q1–Q4 from distribution data
- Calculate `pct_foreign_born`, `cars_per_capita`

**Output**: GeoParquet — `zone_id, geometry, zone_type, population` + country-specific attributes

---

### Step 4b: Socioeconomic data (`src/prepare_socioeconomic.py`)

**New code** (three-tier strategy):

```python
def prepare_socioeconomic(country: str, config: dict) -> pd.DataFrame:
    tier = config["socioeconomic"]["tier"]
    if tier == 1:
        return _prepare_tier1(country, config)
    elif tier == 2:
        return _prepare_tier2(country, config)
    else:
        return _prepare_tier3(country, config)
```

- **Tier 1**: Load country-specific register/census data at zone or grid level
- **Tier 2**: Country-specific investigation (document in `dbs/{country}/socioeconomic/README.md`)
- **Tier 3**: NASA SEDAC gridded deprivation → binary partition via threshold

**Output**: Parquet — `zone_id` (or `grid_id`), `group_variable`, `tier` metadata

---

### Step 5: Home-zone assignment + IPW (`src/assign_home_zone.py`)

**Adapt from**: `geo-social-mixing/src/data/assign_home_deso_ipw.py` (345 lines)

**Implementation**:
1. Filter to devices with `has_home=True AND has_building=True`
2. Create point geometries, spatial join to zones (`predicate='within'`)
3. Count devices per zone
4. Compute raw IPW: `w_raw = (population_z / total_pop) / (n_devices_z / total_devices)`
5. Apply Van de Kerckhove trimming:
   ```
   CV = std(w_raw) / mean(w_raw)
   w0 = sqrt(CV² + 1) × 3.5 × median(w_raw)
   w_trimmed = min(w_raw, w0)
   ```
6. Assign weights, add back excluded devices with `NaN` weight

**Output**: Parquet — `device_id, zone_id, population_weight` + zone-level socioeconomic attributes

**Validation**: median weight ~1.0, max capped

---

## Phase 3: Step 6 — POI Assignment

### Step 6: Tiered POI assignment (`src/assign_poi.py`)

**Adapt from**: `geo-social-mixing/src/data/assign_poi_tiered.py` (500 lines)

**Tier system (from reference)**:
| Tier | Method | Threshold |
|------|--------|-----------|
| 1 | Inside building footprint (STRtree containment) | 0m |
| 2 | KDTree distance | ≤20m |
| 3 | KDTree distance | ≤50m |
| 4 | KDTree distance | ≤100m |
| Unmatched | No POI within radius | — |

**Implementation**:
1. Load POIs, parse WKT building footprints, build spatial indices:
   - `STRtree` for Tier 1 containment queries
   - `cKDTree` for Tier 2–4 distance queries
2. Load valid devices (from Step 5)
3. Load stops, filter to valid devices, exclude stops within 100m of device home
4. Process in chunks (50K stops/chunk) to control memory:
   - For each stop: try Tier 1, then Tier 2, 3, 4
   - Record `poi_id, distance, confidence_tier`
5. Apply unified category mapping (`category_mapper.py`)
6. Write incrementally to output parquet

**Memory management** (critical for large datasets):
- Chunk processing with `del` + `gc.collect()` between chunks
- Incremental parquet writes (append, don't accumulate)
- TQDM progress tracking

**POI category mapping** (`src/category_mapper.py`):
- Maps SafeGraph SUB_CATEGORY → 12 unified categories
- Maps OSM tags → same 12 categories (for fallback/other countries)
- Exclusion list: K-12, healthcare, transit, real estate, manufacturing

**Output**: Parquet — `device_id, stop_id, poi_id, poi_category, poi_latitude, poi_longitude, match_tier, distance_to_poi`

**Validation**: ~46% POI match rate for non-home stops

---

### Filtering (`src/filtering.py`)

**Applied between Steps 5 and 6, or after Step 6**:

```python
def filter_devices(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Remove devices with:
    - < min_stops_per_device stops
    - < min_days_observed days
    - home > max_home_distance_km from any stop
    """

def filter_stops(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Remove stops shorter than tmin."""

def consolidate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge devices mapped to same building with overlapping temporal patterns."""
```

---

## Phase 4: Steps 7–10 — Counterfactual Simulation (New Code)

### Step 7: Isochrone computation (`src/isochrone.py`)

**New code**, using r5r/r5py for routing.

**Architecture**:
```python
class IsochroneComputer:
    def __init__(self, config: dict)
    def compute_for_city(self, city: str, grid_centroids: gpd.GeoDataFrame) -> gpd.GeoDataFrame
    def _compute_r5r(self, centroids: pd.DataFrame, osm_pbf: Path) -> gpd.GeoDataFrame
    def _compute_r5py(self, centroids: pd.DataFrame, osm_pbf: Path) -> gpd.GeoDataFrame
    def _load_precomputed(self, path: Path) -> gpd.GeoDataFrame
```

**Implementation**:
1. Extract unique home grid cell centroids from Step 5 output
2. Download/prepare OSM pedestrian network (`.pbf` file) for the city
3. Compute 15-minute walking isochrones:
   - **r5r backend**: Call `r_scripts/compute_isochrones.R` via `subprocess`
   - **r5py backend**: Use `r5py.TravelTimeMatrixComputer` directly
   - **Precomputed**: Load from file
4. Cache results — skip recomputation if parameters unchanged

**R script** (`r_scripts/compute_isochrones.R`):
```r
library(r5r)
library(sf)
library(data.table)

options(java.parameters = "-Xmx8G")

# Args: osm_pbf, points_csv, output_gpkg, threshold_min, walk_speed
args <- commandArgs(trailingOnly = TRUE)

r5r_core <- setup_r5(data_path = dirname(args[1]))
centroids <- fread(args[2])
origins <- st_as_sf(centroids, coords = c("lon", "lat"), crs = 4326)

iso <- isochrone(r5r_core,
                 origins = origins,
                 mode = "WALK",
                 cutoffs = as.integer(args[4]),
                 walk_speed = as.numeric(args[5]))

st_write(iso, args[3], delete_layer = TRUE)
stop_r5(r5r_core)
```

**Output**: GeoParquet — `grid_id, geometry` (isochrone polygon)

**Validation**: ~1.0–1.5 km radius equivalent; follows street network, not circular

---

### Step 8: Simultaneous trip rewiring (`src/rewiring.py`)

**New code** — the core simulation.

**Architecture**:
```python
class TripRewirer:
    def __init__(self, config: dict)
    def build_poi_indices(self, pois: gpd.GeoDataFrame) -> dict[str, cKDTree]
    def rewire_city(self, city: str) -> pd.DataFrame
    def _rewire_trip(self, trip, isochrone_geom, poi_index, rng) -> dict
    def _sample_with_decay(self, candidates: pd.DataFrame, home_point, rng) -> int
```

**Implementation**:
1. Load inputs: POI assignment (Step 6), isochrones (Step 7), POI locations
2. Build spatial index per POI category: `{category: cKDTree(poi_coords)}`
3. For each non-home trip:
   a. Look up device's home grid → isochrone polygon
   b. Check if assigned POI is within isochrone → if yes, keep (`within_isochrone=True`)
   c. If no: query same-category POI index for all POIs within isochrone
      - If candidates exist: sample one via distance-decay (`localizable=True, rewired=True`)
      - If no candidates: retain original (`localizable=False, rewired=False`)
4. Write output partitioned by city

**Distance-decay sampling**:
```python
def _sample_with_decay(self, candidates, home_point, rng):
    distances = candidates.distance(home_point)
    if self.decay_function == "power":
        weights = distances ** (-self.decay_beta)
    else:  # exponential
        weights = np.exp(-self.decay_beta * distances)
    weights = weights / weights.sum()
    return rng.choice(candidates.index, p=weights)
```

**Spatial query optimization**:
- For each trip: use `isochrone_polygon.contains(poi_point)` is O(n) per POI
- Better: pre-filter with cKDTree radius query (bounding circle of isochrone), then exact containment check
- Or: build per-category R-tree, query with isochrone bounding box, filter by containment

**Output columns**: `stop_id, device_id, original_poi_id, final_poi_id, category, within_isochrone, rewired, localizable, distance_to_final`

**Memory**: Process by city. Within a city, chunk by device partition if needed.

**Validation**: 30–60% trips rewired; 10–30% non-localizable

---

### Step 9: Venue-visitor matrix + ICE (`src/segregation.py`)

**New code**.

**Architecture**:
```python
class SegregationComputer:
    def __init__(self, config: dict)
    def compute_for_city(self, city: str) -> dict[str, pd.DataFrame]
    def _build_venue_visitor_matrix(self, trips: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame
    def _compute_ice_venue(self, venue_visitors: pd.DataFrame) -> pd.DataFrame
    def _compute_ice_neighborhood(self, venue_ice: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame
```

**Implementation**:
1. Load rewired trips (Step 8) + device demographics with IPW weights (Step 5)
2. Build venue-visitor matrix:
   - Group trips by `final_poi_id`
   - For each venue: weighted count of advantaged vs. disadvantaged visitors
3. Compute per-venue ICE:
   ```
   ICE_L = (n_advantaged_L − n_disadvantaged_L) / n_total_L
   ```
4. Compute per-neighborhood experienced ICE:
   ```
   ICE_e_j = Σ_L (ICE_L × n_j_L) / Σ_L n_j_L
   ```
   where `n_j_L` = weighted visits from neighborhood j to venue L
5. Compute for both **baseline** (original trips) and **S1** (rewired trips)
6. Report `Δ_ICE_j = ICE_e_j_s1 − ICE_e_j_baseline`

**Output**:
- `venue_visitors/` — venue-level composition (baseline + S1)
- `segregation/` — neighborhood-level ICE + delta

**Validation**: ICE_s1 more extreme than ICE_baseline for disadvantaged neighborhoods

---

### Step 10: Localizability index (`src/localizability.py`)

**New code**.

**Architecture**:
```python
class LocalizabilityComputer:
    def __init__(self, config: dict)
    def compute_for_city(self, city: str) -> dict[str, pd.DataFrame]
    def _per_neighborhood(self, rewired_trips: pd.DataFrame) -> pd.DataFrame
    def _per_city(self, neighborhood_metrics: pd.DataFrame) -> pd.DataFrame
```

**Metrics per neighborhood**:
```python
localizability_trip = n_localizable / n_total_trips
localizability_category = n_categories_with_local_poi / n_categories_observed
missing_categories = [cat for cat in observed if cat not in local_available]
```

**Metrics per city**:
- Distribution of `localizability_trip` across neighborhoods
- Category-specific missingness rates
- Correlation: `localizability_trip` × neighborhood socioeconomic composition

**Output**: Parquet with neighborhood-level and city-level localizability metrics

---

## Phase 5: CLI + Orchestration

### Pipeline orchestrator (`src/pipeline.py`)

```python
class Pipeline:
    def __init__(self, country: str, city: str = None)
    def run(self, steps: list[int] = None) -> None
    def run_step(self, step: int) -> None
    def validate_step(self, step: int) -> dict
```

**Step dependency graph**:
```
0 (device_logging) → 1 (stops) → 2 (home_work) → 3 (buildings)
                                                       ↓
4a (zones) ──→ 5 (home_zone + IPW) ←── 3 (buildings)
4b (socioeco) ──↗                         ↓
                                    6 (POI assignment) → 7 (isochrones)
                                                              ↓
                                                         8 (rewiring) → 9 (segregation)
                                                              ↓
                                                         10 (localizability)
```

### CLI entry points

**`scripts/run_country.py`**:
```
python scripts/run_country.py --country sweden --steps 1-6    # Part A only
python scripts/run_country.py --country sweden                 # Full pipeline
python scripts/run_country.py --country sweden --steps 7-10 --city Malmö
```

**`scripts/run_city.py`**:
```
python scripts/run_city.py --country sweden --city Malmö       # One city, full pipeline
python scripts/run_city.py --country sweden --city Malmö --step 8  # One step
```

**`scripts/validate.py`**:
```
python scripts/validate.py --country sweden                    # All steps
python scripts/validate.py --country sweden --step 6           # One step
```

---

## Implementation Order

Priority is to get end-to-end results for **one city (Malmö)** as fast as possible, then generalize.

### Sprint 1: Foundation + Steps 1–3

| Task | Module | Dependency |
|------|--------|------------|
| Config loader + validation | `config/schema.py` | — |
| Container build + test | `.devcontainer/` | — |
| Step 0: Raw → grouped Parquet | `src/device_logging.py` | Config |
| Step 1: Stop detection | `src/stop_detection.py` | Adapt from reference |
| Step 2: Home/work detection | `src/home_work_detection.py` | Adapt from reference |
| Step 3: Building linkage | `src/link_home_buildings.py` | Adapt from reference |

**Exit criterion**: Home locations for Swedish devices, linked to buildings, in `dbs/sweden/home_buildings/`

### Sprint 2: Steps 4–6

| Task | Module | Dependency |
|------|--------|------------|
| Step 4a: DeSO zones | `src/prepare_zones.py` | Adapt from reference |
| Step 4b: Socioeconomic (Sweden) | `src/prepare_socioeconomic.py` | Zone geometry |
| Step 5: Home-zone + IPW | `src/assign_home_zone.py` | Adapt from reference |
| Category mapper | `src/category_mapper.py` | Adapt from reference |
| Step 6: POI assignment | `src/assign_poi.py` | Adapt from reference |
| Filtering | `src/filtering.py` | After Step 6 |

**Exit criterion**: `dbs/sweden/poi_assignment/` with stop-POI pairs, categories, and IPW weights

### Sprint 3: Steps 7–8 (core simulation)

| Task | Module | Dependency |
|------|--------|------------|
| OSM network download | — | City bbox |
| Step 7: R script | `r_scripts/compute_isochrones.R` | r5r setup |
| Step 7: Python wrapper | `src/isochrone.py` | R script |
| Step 8: Rewiring | `src/rewiring.py` | Isochrones + POI |
| Tests: rewiring | `tests/test_rewiring.py` | Rewiring logic |

**Exit criterion**: `dbs/sweden/simulation/rewired_trips/` for Malmö with plausible rewiring rates

### Sprint 4: Steps 9–10 + validation

| Task | Module | Dependency |
|------|--------|------------|
| Step 9: Segregation | `src/segregation.py` | Rewired trips |
| Step 10: Localizability | `src/localizability.py` | Rewired trips |
| Tests: segregation | `tests/test_segregation.py` | ICE logic |
| End-to-end validation (Malmö) | `scripts/validate.py` | All steps |
| Diagnostic notebook | `notebooks/07_simulation_results.ipynb` | Results |

**Exit criterion**: ICE baseline vs. S1 comparison for Malmö; localizability maps; direction check passes

### Sprint 5: Scale + generalize

| Task | Module | Dependency |
|------|--------|------------|
| Run Stockholm + Gothenburg | CLI | Validated pipeline |
| Pipeline orchestrator | `src/pipeline.py` | All steps |
| CLI: run_country, run_city | `scripts/` | Orchestrator |
| Second country config (USA) | `config/countries/usa.yaml` | Tier 1 socioeco |
| Diagnostic notebooks (01–05) | `notebooks/` | Per-step outputs |

---

## Cross-Cutting Concerns

### Memory management

Follow `geo-social-mixing` patterns:
- **Chunk processing**: 50K records per chunk for spatial operations
- **Incremental I/O**: Append to parquet, never accumulate full dataset in memory
- **Explicit cleanup**: `del df; gc.collect()` after processing each chunk
- **PySpark**: For Steps 1–2 on large datasets (millions of devices)

### Reproducibility

- All random operations seeded (`config.rewiring.seed = 42`)
- Parquet metadata: write pipeline version, config hash, timestamp to file metadata
- Pin all dependency versions in `environment.yml`

### Portability

- `pathlib.Path` for all file operations (Windows + POSIX)
- No hardcoded paths — all from config YAML or CLI args
- CRS from country config (`crs_projected`)
- UTC offset from country config

### Monitoring

- TQDM progress bars for batch processing
- Optional Telegram notifications for long runs (via `.env`)
- Per-step summary statistics written to log

---

## Validation Checklist

Run `scripts/validate.py` to check:

| Step | Metric | Expected | Action if fails |
|------|--------|----------|-----------------|
| 1 | Stops per device (median) | 20–50 | Check r1/tmin params |
| 2 | Home detection rate | ~70% | Check night_hours, min_days |
| 3 | Building match rate (urban) | >90% | Check buffer distance |
| 4 | Zone coverage | All pop>0 zones have geometry | Check data completeness |
| 5 | IPW weight median | ~1.0 | Check spatial join CRS |
| 5 | IPW weight max | ≤10 (after cap) | Check trimming |
| 6 | POI match rate | ~40–50% | Check search radius |
| 7 | Isochrone area (urban) | ~3–7 km² | Check walk speed, network |
| 8 | Rewiring rate | 30–60% | Check isochrone coverage |
| 8 | Non-localizable rate | 10–30% | Check category coverage |
| 9 | ICE direction | S1 more extreme for disadvantaged | Check ICE formula |
| 10 | Localizability–composition | Negative correlation | Expected finding |
