"""Step 0: Raw GPS data — device logging and day-level grouped Parquet.

Two stages (following mobi-seg-net/src/data_etl/1-raw2parquet.py):

    log     — Scan raw files, collect unique device IDs per month,
              compute deterministic group assignments (MD5 hash),
              save device log to {output_dir}/devices/devices_{month}.parquet.

    convert — For each day, load ALL hourly raw files for that day,
              assign device groups, write ONE Parquet per group per day:
              {output_dir}/format_parquet/grp_{N}/{month}_{day}.parquet.

Output layout:
    {output_dir}/
    ├── devices/
    │   └── devices_{month}.parquet        (device_aid, grp)
    └── format_parquet/
        ├── grp_0/
        │   ├── 03_01.parquet
        │   ├── 03_02.parquet
        │   └── ...
        ├── grp_1/
        └── ...

Usage:
    # Both stages (log then convert)
    python src/device_logging.py --raw-dir /data/mobile/se_2024/03 \
                                 --output-dir /data/mobile/se_2024 \
                                 --n-groups 50 --workers 8

    # Log devices only
    python src/device_logging.py --raw-dir /data/mobile/se_2024/03 \
                                 --output-dir /data/mobile/se_2024 \
                                 --n-groups 50 --stage log --workers 8

    # Convert only (groups computed on-the-fly via hash)
    python src/device_logging.py --raw-dir /data/mobile/se_2024/03 \
                                 --output-dir /data/mobile/se_2024 \
                                 --n-groups 50 --stage convert --workers 4

    # Dry run
    python src/device_logging.py --raw-dir /data/mobile/se_2024/03 \
                                 --output-dir /data/mobile/se_2024 \
                                 --n-groups 50 --dry-run

Reference:
    mobi-seg-net uses random group assignment (np.random.randint) requiring
    a two-pass workflow. This version uses MD5 hashing for deterministic
    groups, so the convert stage is self-contained.
"""

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
from tqdm import tqdm

COLUMNS_KEEP = ["timestamp", "device_aid", "latitude", "longitude", "location_method"]

PARQUET_SCHEMA = pa.schema([
    ("timestamp", pa.int64()),
    ("device_aid", pa.string()),
    ("latitude", pa.float64()),
    ("longitude", pa.float64()),
    ("location_method", pa.string()),
    ("grp", pa.int32()),
])

ARROW_PARSE_OPTIONS = pcsv.ParseOptions(delimiter="\t")
ARROW_CONVERT_OPTIONS = pcsv.ConvertOptions(
    include_columns=COLUMNS_KEEP,
    column_types={
        "timestamp": pa.int64(),
        "device_aid": pa.string(),
        "latitude": pa.float64(),
        "longitude": pa.float64(),
        "location_method": pa.string(),
    },
)
ARROW_CONVERT_DEVICE_ONLY = pcsv.ConvertOptions(
    include_columns=["device_aid"],
    column_types={"device_aid": pa.string()},
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def device_to_group(device_aid: str, n_groups: int) -> int:
    """Deterministic device-to-group assignment via MD5 hash.

    MD5 is chosen over Python's built-in hash() because hash() is
    randomized across processes (PYTHONHASHSEED) and would break
    the persistence guarantee across separate runs.
    """
    h = hashlib.md5(device_aid.encode("utf-8")).hexdigest()
    return int(h, 16) % n_groups


def discover_days(raw_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """Group raw GPS files by (month, day).

    Supports two directory layouts:
    - Month-level raw_dir:  {raw_dir}/{day}/*.csv.gz
    - Year-level raw_dir:   {raw_dir}/{month}/{day}/*.csv.gz
    """
    all_files = []
    for ext in ("*.csv.gz", "*.csv.zip", "*.zip"):
        all_files.extend(raw_dir.rglob(ext))
    all_files.sort()

    days = defaultdict(list)
    for f in all_files:
        rel = f.relative_to(raw_dir)
        parts = rel.parts

        if len(parts) >= 3 and parts[-3].isdigit() and len(parts[-3]) == 2 \
                and parts[-2].isdigit() and len(parts[-2]) == 2:
            month, day = parts[-3], parts[-2]
        elif len(parts) >= 2 and parts[-2].isdigit() and len(parts[-2]) == 2:
            day = parts[-2]
            month = raw_dir.name if raw_dir.name.isdigit() and len(raw_dir.name) == 2 else "00"
        else:
            continue

        days[(month, day)].append(f)

    return dict(days)


def read_raw_file(file_path: Path) -> pd.DataFrame:
    """Read a single raw GPS file into a DataFrame.

    Uses PyArrow CSV reader for .gz files (faster than pandas for
    gzip decompression + TSV parsing). Falls back to pandas on
    parse errors or for .zip files.
    """
    if file_path.suffix == ".gz":
        try:
            stream = pa.input_stream(str(file_path), compression="gzip")
            table = pcsv.read_csv(
                stream,
                parse_options=ARROW_PARSE_OPTIONS,
                convert_options=ARROW_CONVERT_OPTIONS,
            )
            df = table.to_pandas()
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowKeyError):
            df = pd.read_csv(
                file_path,
                sep="\t",
                compression="gzip",
                usecols=lambda c: c in COLUMNS_KEEP,
                dtype={
                    "timestamp": np.int64,
                    "device_aid": str,
                    "latitude": np.float64,
                    "longitude": np.float64,
                    "location_method": str,
                },
                on_bad_lines="skip",
            )
    elif file_path.suffix == ".zip":
        dfs = []
        with zipfile.ZipFile(file_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            for csv_name in csv_names:
                with zf.open(csv_name) as f:
                    chunk = pd.read_csv(
                        io.TextIOWrapper(f),
                        sep="\t",
                        usecols=lambda c: c in COLUMNS_KEEP,
                        dtype={
                            "timestamp": np.int64,
                            "device_aid": str,
                            "latitude": np.float64,
                            "longitude": np.float64,
                            "location_method": str,
                        },
                        on_bad_lines="skip",
                    )
                    dfs.append(chunk)
        if not dfs:
            return pd.DataFrame(columns=COLUMNS_KEEP)
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    for col in COLUMNS_KEEP:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}' in {file_path}. "
                f"Found columns: {list(df.columns)}"
            )

    return df[COLUMNS_KEEP].dropna(subset=["device_aid", "latitude", "longitude"])


def read_device_ids(file_path: Path) -> np.ndarray:
    """Read only the device_aid column from a raw GPS file.

    Lightweight path for the device logging stage — skips all columns
    except device_aid, so it uses much less memory and time.
    """
    if file_path.suffix == ".gz":
        try:
            stream = pa.input_stream(str(file_path), compression="gzip")
            table = pcsv.read_csv(
                stream,
                parse_options=ARROW_PARSE_OPTIONS,
                convert_options=ARROW_CONVERT_DEVICE_ONLY,
            )
            col = table.column("device_aid")
            return col.drop_null().unique().to_pylist()
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowKeyError):
            df = pd.read_csv(
                file_path,
                sep="\t",
                compression="gzip",
                usecols=["device_aid"],
                dtype={"device_aid": str},
                on_bad_lines="skip",
            )
            return df["device_aid"].dropna().unique()
    elif file_path.suffix == ".zip":
        dfs = []
        with zipfile.ZipFile(file_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    with zf.open(name) as f:
                        chunk = pd.read_csv(
                            io.TextIOWrapper(f),
                            sep="\t",
                            usecols=["device_aid"],
                            dtype={"device_aid": str},
                            on_bad_lines="skip",
                        )
                        dfs.append(chunk)
        if not dfs:
            return np.array([], dtype=str)
        return pd.concat(dfs)["device_aid"].dropna().unique()
    else:
        raise ValueError(f"Unsupported: {file_path.suffix}")


# ---------------------------------------------------------------------------
# Stage 1: Device logging  (parallelised per day)
# ---------------------------------------------------------------------------

def _scan_day_devices(file_paths_str):
    """Return unique device IDs from one day's files."""
    devices = set()
    for f_str in file_paths_str:
        devices.update(read_device_ids(Path(f_str)))
    return devices


def log_devices(raw_dir, output_dir, n_groups, workers=1):
    """Scan raw files, collect unique devices per month, save with group assignments."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    days = discover_days(raw_dir)
    if not days:
        print(f"No raw files found in {raw_dir}")
        return

    print(f"Logging devices from {len(days)} days...")

    month_devices = defaultdict(set)

    if workers <= 1:
        for (month, day), files in tqdm(sorted(days.items()), desc="Scanning devices"):
            day_devs = _scan_day_devices([str(f) for f in files])
            month_devices[month].update(day_devs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for (month, day), files in days.items():
                future = executor.submit(_scan_day_devices, [str(f) for f in files])
                futures[future] = month
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning devices"):
                month = futures[future]
                month_devices[month].update(future.result())

    devices_dir = output_dir / "devices"
    devices_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for month in sorted(month_devices):
        device_list = sorted(month_devices[month])
        devices_df = pd.DataFrame({"device_aid": device_list})
        group_map = {d: device_to_group(d, n_groups) for d in device_list}
        devices_df["grp"] = devices_df["device_aid"].map(group_map).astype(np.int32)

        out_path = devices_dir / f"devices_{month}.parquet"
        devices_df.to_parquet(out_path, index=False)
        total += len(devices_df)
        print(f"  {out_path.name}: {len(devices_df):,} devices")

    print(f"Total unique devices: {total:,}")
    print(f"Groups: {n_groups}, ~{total // max(n_groups, 1):,} devices/group")


# ---------------------------------------------------------------------------
# Stage 2: Day-level conversion  (parallelised per day)
# ---------------------------------------------------------------------------

def _process_day(file_paths_str, output_dir_str, n_groups, month, day, resume):
    """Load all hourly files for one day, assign groups, write one parquet per group."""
    output_dir = Path(output_dir_str) / "format_parquet"
    out_name = f"{month}_{day}.parquet"

    if resume and (output_dir / "grp_0" / out_name).exists():
        return {"month": month, "day": day, "status": "skipped"}

    try:
        dfs = [read_raw_file(Path(f)) for f in file_paths_str]
        df = pd.concat(dfs, ignore_index=True)
        del dfs
    except Exception as e:
        return {"month": month, "day": day, "status": "error", "error": str(e)}

    if df.empty:
        return {"month": month, "day": day, "status": "empty", "records": 0}

    n_raw = len(df)
    unique_devices = df["device_aid"].unique()
    n_devices = len(unique_devices)

    device_group_map = {d: device_to_group(d, n_groups) for d in unique_devices}
    df["grp"] = df["device_aid"].map(device_group_map).astype(np.int32)

    records_per_group = {}
    for grp_id, grp_df in df.groupby("grp"):
        grp_dir = output_dir / f"grp_{int(grp_id)}"
        grp_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(grp_df, schema=PARQUET_SCHEMA, preserve_index=False)
        pq.write_table(table, grp_dir / out_name, compression="snappy")
        records_per_group[int(grp_id)] = len(grp_df)

    return {
        "month": month,
        "day": day,
        "status": "ok",
        "records": n_raw,
        "devices": n_devices,
        "records_per_group": records_per_group,
    }


def convert_days(raw_dir, output_dir, n_groups, workers=1, resume=False):
    """For each day, load all hourly files and write day-level grouped parquets."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    days = discover_days(raw_dir)
    if not days:
        print(f"No raw files found in {raw_dir}")
        return {}

    # Ensure all group directories exist
    fmt_dir = output_dir / "format_parquet"
    for g in range(n_groups):
        (fmt_dir / f"grp_{g}").mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(days)} days into grouped parquets...")
    print(f"Output: {fmt_dir} ({n_groups} groups)")
    print(f"Workers: {workers}")
    if resume:
        print("Resume mode: skipping already-processed days")
    print()

    start_time = time.time()
    args_list = [
        ([str(f) for f in files], str(output_dir), n_groups, month, day, resume)
        for (month, day), files in sorted(days.items())
    ]

    day_stats = []
    if workers <= 1:
        for args in tqdm(args_list, desc="Converting days"):
            day_stats.append(_process_day(*args))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_day, *args): f"{args[3]}_{args[4]}"
                for args in args_list
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Converting days"
            ):
                try:
                    day_stats.append(future.result())
                except Exception as e:
                    label = futures[future]
                    day_stats.append({"day": label, "status": "error", "error": str(e)})

    # Aggregate and report
    elapsed = time.time() - start_time
    files_ok = sum(1 for r in day_stats if r.get("status") == "ok")
    files_skip = sum(1 for r in day_stats if r.get("status") == "skipped")
    total_records = sum(r.get("records", 0) for r in day_stats)
    records_per_group = defaultdict(int)
    for r in day_stats:
        for grp_id, count in r.get("records_per_group", {}).items():
            records_per_group[grp_id] += count

    group_sizes = list(records_per_group.values()) if records_per_group else [0]
    summary = {
        "raw_dir": str(raw_dir),
        "output_dir": str(fmt_dir),
        "n_groups": n_groups,
        "workers": workers,
        "days_processed": files_ok,
        "days_skipped": files_skip,
        "total_records": total_records,
        "records_per_group_min": min(group_sizes),
        "records_per_group_max": max(group_sizes),
        "records_per_group_mean": float(np.mean(group_sizes)),
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    manifest = {"summary": summary, "days": day_stats}
    manifest_path = fmt_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("Conversion complete")
    print(f"{'=' * 60}")
    print(f"  Days processed:  {files_ok}")
    print(f"  Days skipped:    {files_skip}")
    print(f"  Total records:   {total_records:,}")
    print(f"  Device groups:   {n_groups}")
    print(f"  Workers:         {workers}")
    print(f"  Records/group:   {min(group_sizes):,} – {max(group_sizes):,} "
          f"(mean {np.mean(group_sizes):,.0f})")
    print(f"  Elapsed:         {elapsed / 60:.1f} min")
    print(f"  Manifest:        {manifest_path}")

    return summary


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def scan_only(raw_dir, n_groups):
    """Scan all files and report counts without writing output."""
    raw_dir = Path(raw_dir)
    days = discover_days(raw_dir)
    if not days:
        print("No raw files found.")
        return {}

    print(f"Scanning {raw_dir} (dry run)...")
    all_devices = set()
    total_records = 0

    for (month, day), files in tqdm(sorted(days.items()), desc="Scanning"):
        for file_path in files:
            try:
                df = read_raw_file(file_path)
                total_records += len(df)
                all_devices.update(df["device_aid"].unique())
            except Exception as e:
                print(f"\nError reading {file_path}: {e}")

    n_devices = len(all_devices)
    n_days = len(days)
    print(f"\nScan complete:")
    print(f"  Days:    {n_days}")
    print(f"  Files:   {sum(len(f) for f in days.values())}")
    print(f"  Records: {total_records:,}")
    print(f"  Devices: {n_devices:,}")
    print(f"  Avg records/device: {total_records / max(n_devices, 1):.1f}")
    print(f"  Suggested groups:")
    print(f"    50  → ~{n_devices // 50:,} devices/group")
    print(f"    100 → ~{n_devices // 100:,} devices/group")
    print(f"    300 → ~{n_devices // 300:,} devices/group")

    return {"days": n_days, "records": total_records, "unique_devices": n_devices}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Device logging and day-level grouped Parquet conversion"
    )
    parser.add_argument(
        "--raw-dir", type=str, required=True,
        help="Directory containing raw GPS files ({month}/{day}/ or {day}/ layout)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output root directory (devices/ and format_parquet/ created inside)",
    )
    parser.add_argument(
        "--n-groups", type=int, default=50,
        help="Number of device groups (default: 50)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (default: 1). Each worker loads one full day "
             "into memory (~3-5 GB), so total memory ≈ workers × 5 GB.",
    )
    parser.add_argument(
        "--stage", choices=["log", "convert", "all"], default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan files and report counts without writing output",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip days whose output parquet already exists",
    )

    args = parser.parse_args()

    if not Path(args.raw_dir).exists():
        print(f"Error: raw directory not found: {args.raw_dir}")
        sys.exit(1)

    if args.dry_run:
        scan_only(args.raw_dir, args.n_groups)
        return

    if args.stage in ("log", "all"):
        log_devices(args.raw_dir, args.output_dir, args.n_groups, workers=args.workers)

    if args.stage in ("convert", "all"):
        convert_days(
            args.raw_dir, args.output_dir, args.n_groups,
            workers=args.workers, resume=args.resume,
        )


if __name__ == "__main__":
    main()
