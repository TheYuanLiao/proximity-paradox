"""Config validation with pydantic."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator


class DeviceLoggingConfig(BaseModel):
    n_groups: int = 50
    hash_function: str = "md5"


class StopDetectionConfig(BaseModel):
    r1: int = 30
    r2: int = 30
    tmin: int = 15
    max_time_between: int = 3
    engine: str = "infostop"
    use_spark: bool = True


class HomeWorkConfig(BaseModel):
    algorithm: str = "HoWDe"
    min_days_home: int = 3
    night_hours: list[int] = [22, 7]
    range_window_home: int = 14
    range_window_work: int = 28


class BuildingsConfig(BaseModel):
    source: str = "overture_maps"
    max_link_distance: int = 50
    buffer_m: int = 50


class FilteringConfig(BaseModel):
    min_stops_per_device: int = 10
    min_days_observed: int = 7
    max_home_distance_km: int = 200


class POITiersConfig(BaseModel):
    tier_1_m: int = 0
    tier_2_m: int = 20
    tier_3_m: int = 50
    tier_4_m: int = 100


class POIConfig(BaseModel):
    search_radius_m: int = 300
    tiered_fallback: bool = True
    category_schema: str = "config/poi_categories.yaml"
    tiers: POITiersConfig = POITiersConfig()
    chunk_size: int = 50000
    source: Optional[str] = None
    fallback: Optional[str] = None


class IsochroneConfig(BaseModel):
    mode: str = "walking"
    speed_kmh: float = 5.0
    threshold_minutes: int = 15
    backend: str = "r5r"


class RewiringConfig(BaseModel):
    decay_function: str = "power"
    decay_beta: float = 1.5
    seed: int = 42


class SegregationConfig(BaseModel):
    metric: str = "ICE"


class IPWConfig(BaseModel):
    max_weight_cap: float = 10.0
    trimming: str = "van_de_kerckhove"
    trimming_factor: float = 3.5


class SparkConfig(BaseModel):
    driver_memory: str = "56g"
    executor_memory: str = "8g"
    cores: int = 18


class GroupValues(BaseModel):
    advantaged: str
    disadvantaged: str


class SocioeconomicConfig(BaseModel):
    tier: int
    zone_type: str
    grid_resolution: int = 250
    group_variable: str
    group_values: GroupValues
    source: str
    variables: list[str] = []


class CityConfig(BaseModel):
    name: str
    bbox: list[float]
    municipality_codes: list[str] = []

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 values: [west, south, east, north]")
        return v


class RawGPSConfig(BaseModel):
    raw_dir: str = ""
    grouped_dir: str = ""
    n_groups: int = 50
    path_template: str = ""


class DefaultConfig(BaseModel):
    device_logging: DeviceLoggingConfig = DeviceLoggingConfig()
    stop_detection: StopDetectionConfig = StopDetectionConfig()
    home_work: HomeWorkConfig = HomeWorkConfig()
    buildings: BuildingsConfig = BuildingsConfig()
    filtering: FilteringConfig = FilteringConfig()
    poi: POIConfig = POIConfig()
    isochrone: IsochroneConfig = IsochroneConfig()
    rewiring: RewiringConfig = RewiringConfig()
    segregation: SegregationConfig = SegregationConfig()
    ipw: IPWConfig = IPWConfig()
    spark: SparkConfig = SparkConfig()


class CountryConfig(BaseModel):
    country: str
    data_root: str
    crs_projected: str = "EPSG:4326"
    utc_offset_seconds: int = 0
    socioeconomic: SocioeconomicConfig
    poi: POIConfig = POIConfig()
    raw_gps: RawGPSConfig = RawGPSConfig()
    cities: list[CityConfig] = []


def load_default_config(path: Path = None) -> DefaultConfig:
    if path is None:
        path = Path(__file__).parent / "default.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return DefaultConfig(**data)


def load_country_config(country: str, config_dir: Path = None) -> CountryConfig:
    if config_dir is None:
        config_dir = Path(__file__).parent / "countries"
    path = config_dir / f"{country}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return CountryConfig(**data)


def load_merged_config(country: str, config_dir: Path = None) -> dict:
    defaults = load_default_config().model_dump()
    country_cfg = load_country_config(country, config_dir).model_dump()
    merged = {**defaults, **country_cfg}
    for key in defaults:
        if key in country_cfg and isinstance(defaults[key], dict) and isinstance(country_cfg[key], dict):
            merged[key] = {**defaults[key], **country_cfg[key]}
    return merged
