"""Step 0: Raw GPS data to device-grouped Parquet.

Converts daily compressed GPS archives (CSV.gz or ZIP containing CSV)
into Parquet files partitioned by deterministic device group assignments.
This is the prerequisite for all downstream processing — the stop
detection pipeline (Step 1) loads one device group at a time instead
of scanning the full dataset.

Raw data delivery format (from providers like Predicio/Pickwell via S3):
    {raw_dir}/
    ├── {month}/{day}/locations-{HH}-part{NNNN}.csv.gz    (S3 layout)
    ├── locations-{HH}-part{NNNN}.csv.gz                  (flat layout)
    └── *.zip containing CSV files                         (ZIP variant)

    Columns (TSV): timestamp, device_aid, device_aid_type, latitude,
        longitude, horizontal_accuracy, altitude, altitude_accuracy,
        location_method, ip, user_agent, OS, OS_version, manufacturer,
        model, carrier

Output format:
    {output_dir}/format_parquet/
    ├── grp_0/
    │   ├── {date_label}.parquet
    │   └── ...
    ├── grp_1/
    └── ...

    Columns: timestamp, device_aid, latitude, longitude, location_method, grp

Device grouping:
    grp = hash(device_aid) % n_groups
    Deterministic: same device always maps to the same group regardless
    of which daily file it appears in. This is critical because downstream
    stop detection operates device-wise within a single group.

Usage:
    # Process all raw files in a directory
    python src/device_logging.py --raw-dir /data/raw_gps/SE/2024 \
                                 --output-dir /data/dbs/sweden \
                                 --n-groups 50

    # Dry run: scan files and report device counts without writing
    python src/device_logging.py --raw-dir /data/raw_gps/SE/2024 \
                                 --output-dir /data/dbs/sweden \
                                 --n-groups 50 --dry-run

    # Resume after interruption (skip already-processed files)
    python src/device_logging.py --raw-dir /data/raw_gps/SE/2024 \
                                 --output-dir /data/dbs/sweden \
                                 --n-groups 50 --resume

Reference:
    In geo-social-mixing, this step was done ad-hoc before the pipeline.
    Sweden 2024 used 50 groups; Germany (>1 year of data) used 300.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

COLUMNS_KEEP = ["timestamp", "device_aid", "latitude", "longitude", "location_method"]

RAW_COLUMNS = [
    "timestamp", "device_aid", "device_aid_type", "latitude", "longitude",
    "horizontal_accuracy", "altitude", "altitude_accuracy", "location_method",
    "ip", "user_agent", "OS", "OS_version", "manufacturer", "model", "carrier",
]

PARQUET_SCHEMA = pa.schema([
    ("timestamp", pa.int64()),
    ("device_aid", pa.string()),
    ("latitude", pa.float64()),
    ("longitude", pa.float64()),
    ("location_method", pa.string()),
    ("grp", pa.int32()),
])


def device_to_group(device_aid: str, n_groups: int) -> int:
    """Deterministic device-to-group assignment via MD5 hash.

    MD5 is chosen over Python's built-in hash() because hash() is
    randomized across processes (PYTHONHASHSEED) and would break
    the persistence guarantee across separate runs.
    """
    h = hashlib.md5(device_aid.encode("utf-8")).hexdigest()
    return int(h, 16) % n_groups


def discover_raw_files(raw_dir: Path) -> list[Path]:
    """Find all raw GPS files in a directory tree.

    Handles three layouts:
    1. S3-style: {month}/{day}/locations-*.csv.gz
    2. Flat: *.csv.gz in one directory
    3. ZIP archives: *.zip containing CSV files
    """
    files = []
    for ext in ("*.csv.gz", "*.csv.zip", "*.zip"):
        files.extend(raw_dir.rglob(ext))
    files.sort()
    return files


def derive_date_label(file_path: Path) -> str:
    """Extract a date label from the file path for the output parquet name.

    Tries several conventions:
    - S3 layout: .../2024/06/01/locations-00-part0000.csv.gz → 06_01_00
    - Flat with date in name: 2024_06_01.csv.gz → 2024_06_01
    - Falls back to filename stem
    """
    parts = file_path.parts
    name = file_path.stem.replace(".csv", "")

    for i, part in enumerate(parts):
        if part.isdigit() and len(part) == 2 and i + 1 < len(parts):
            month = part
            if parts[i + 1].isdigit() and len(parts[i + 1]) == 2:
                day = parts[i + 1]
                return f"{month}_{day}_{name}"

    return name


def read_raw_file(file_path: Path) -> pd.DataFrame:
    """Read a single raw GPS file (CSV.gz or ZIP) into a DataFrame.

    Strips PII columns, keeping only what the pipeline needs.
    """
    if file_path.suffix == ".gz":
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


class RawToParquet:
    """Converts raw GPS archives to device-grouped Parquet files.

    Parameters
    ----------
    raw_dir : Path
        Root directory containing raw GPS files.
    output_dir : Path
        Root output directory. Parquet files are written to
        {output_dir}/format_parquet/grp_{N}/.
    n_groups : int
        Number of device groups (50 for Sweden, 300 for Germany).
    """

    def __init__(self, raw_dir: Path, output_dir: Path, n_groups: int = 50):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir) / "format_parquet"
        self.n_groups = n_groups
        self.stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "total_records": 0,
            "total_records_written": 0,
            "unique_devices": set(),
            "records_per_group": defaultdict(int),
            "start_time": None,
        }

    def _ensure_group_dirs(self):
        for g in range(self.n_groups):
            (self.output_dir / f"grp_{g}").mkdir(parents=True, exist_ok=True)

    def _output_exists(self, date_label: str) -> bool:
        """Check if any group already has this date_label's parquet."""
        sample_path = self.output_dir / "grp_0" / f"{date_label}.parquet"
        return sample_path.exists()

    def process_file(self, file_path: Path, resume: bool = False) -> dict:
        """Process a single raw file into grouped Parquet outputs.

        Returns per-file statistics dict.
        """
        date_label = derive_date_label(file_path)

        if resume and self._output_exists(date_label):
            return {"file": str(file_path), "status": "skipped", "reason": "exists"}

        df = read_raw_file(file_path)

        if df.empty:
            return {"file": str(file_path), "status": "empty", "records": 0}

        n_raw = len(df)
        devices_in_file = set(df["device_aid"].unique())

        df["grp"] = df["device_aid"].map(
            lambda d: device_to_group(d, self.n_groups)
        ).astype(np.int32)

        for grp_id, grp_df in df.groupby("grp"):
            out_path = self.output_dir / f"grp_{grp_id}" / f"{date_label}.parquet"
            table = pa.Table.from_pandas(grp_df, schema=PARQUET_SCHEMA, preserve_index=False)
            pq.write_table(table, out_path, compression="snappy")
            self.stats["records_per_group"][grp_id] += len(grp_df)

        self.stats["files_processed"] += 1
        self.stats["total_records"] += n_raw
        self.stats["total_records_written"] += n_raw
        self.stats["unique_devices"].update(devices_in_file)

        return {
            "file": str(file_path),
            "status": "ok",
            "records": n_raw,
            "devices": len(devices_in_file),
            "date_label": date_label,
        }

    def process_all(self, resume: bool = False) -> dict:
        """Process all raw files in the raw directory.

        Parameters
        ----------
        resume : bool
            If True, skip files whose output already exists.
        """
        self.stats["start_time"] = time.time()
        self._ensure_group_dirs()

        raw_files = discover_raw_files(self.raw_dir)
        if not raw_files:
            print(f"No raw files found in {self.raw_dir}")
            return self.get_summary()

        print(f"Found {len(raw_files)} raw files in {self.raw_dir}")
        print(f"Output: {self.output_dir} ({self.n_groups} groups)")
        if resume:
            print("Resume mode: skipping already-processed files")
        print()

        file_stats = []
        for file_path in tqdm(raw_files, desc="Processing raw files"):
            try:
                result = self.process_file(file_path, resume=resume)
                file_stats.append(result)
                if result["status"] == "skipped":
                    self.stats["files_skipped"] += 1
            except Exception as e:
                file_stats.append({
                    "file": str(file_path),
                    "status": "error",
                    "error": str(e),
                })
                print(f"\nError processing {file_path}: {e}")

        summary = self.get_summary()
        self._write_manifest(summary, file_stats)
        self._print_summary(summary)

        return summary

    def scan_only(self) -> dict:
        """Dry run: scan all files and report device/record counts
        without writing any output."""
        print(f"Scanning {self.raw_dir} (dry run)...")

        raw_files = discover_raw_files(self.raw_dir)
        if not raw_files:
            print("No raw files found.")
            return {}

        all_devices = set()
        total_records = 0
        file_count = 0

        for file_path in tqdm(raw_files, desc="Scanning"):
            try:
                df = read_raw_file(file_path)
                total_records += len(df)
                all_devices.update(df["device_aid"].unique())
                file_count += 1
            except Exception as e:
                print(f"\nError reading {file_path}: {e}")

        n_devices = len(all_devices)
        print(f"\nScan complete:")
        print(f"  Files:   {file_count}")
        print(f"  Records: {total_records:,}")
        print(f"  Devices: {n_devices:,}")
        print(f"  Avg records/device: {total_records / max(n_devices, 1):.1f}")
        print(f"  Suggested groups:")
        print(f"    50  → ~{n_devices // 50:,} devices/group")
        print(f"    100 → ~{n_devices // 100:,} devices/group")
        print(f"    300 → ~{n_devices // 300:,} devices/group")

        return {
            "files": file_count,
            "records": total_records,
            "unique_devices": n_devices,
        }

    def get_summary(self) -> dict:
        elapsed = time.time() - (self.stats["start_time"] or time.time())
        n_devices = len(self.stats["unique_devices"])
        group_counts = dict(self.stats["records_per_group"])

        group_sizes = list(group_counts.values()) if group_counts else [0]
        return {
            "raw_dir": str(self.raw_dir),
            "output_dir": str(self.output_dir),
            "n_groups": self.n_groups,
            "files_processed": self.stats["files_processed"],
            "files_skipped": self.stats["files_skipped"],
            "total_records": self.stats["total_records"],
            "unique_devices": n_devices,
            "devices_per_group_mean": n_devices / max(self.n_groups, 1),
            "records_per_group_min": min(group_sizes),
            "records_per_group_max": max(group_sizes),
            "records_per_group_mean": np.mean(group_sizes),
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

    def _write_manifest(self, summary: dict, file_stats: list):
        """Write processing manifest to output directory."""
        manifest = {
            "summary": summary,
            "files": file_stats,
        }
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest written to {manifest_path}")

    def _print_summary(self, summary: dict):
        elapsed = summary["elapsed_seconds"]
        print(f"\n{'=' * 60}")
        print("Processing complete")
        print(f"{'=' * 60}")
        print(f"  Files processed: {summary['files_processed']}")
        print(f"  Files skipped:   {summary['files_skipped']}")
        print(f"  Total records:   {summary['total_records']:,}")
        print(f"  Unique devices:  {summary['unique_devices']:,}")
        print(f"  Device groups:   {summary['n_groups']}")
        print(f"  Devices/group:   ~{summary['devices_per_group_mean']:.0f}")
        print(f"  Records/group:   {summary['records_per_group_min']:,} – "
              f"{summary['records_per_group_max']:,} "
              f"(mean {summary['records_per_group_mean']:,.0f})")
        print(f"  Elapsed:         {elapsed / 60:.1f} min")
        print(f"  Output:          {summary['output_dir']}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw GPS archives to device-grouped Parquet files"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Directory containing raw GPS files (csv.gz or zip)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output root directory (format_parquet/ created inside)",
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=50,
        help="Number of device groups (default: 50; use 300 for large datasets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and report counts without writing output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files whose output parquet already exists",
    )

    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not raw_dir.exists():
        print(f"Error: raw directory not found: {raw_dir}")
        sys.exit(1)

    converter = RawToParquet(raw_dir, output_dir, n_groups=args.n_groups)

    if args.dry_run:
        converter.scan_only()
    else:
        converter.process_all(resume=args.resume)


if __name__ == "__main__":
    main()
