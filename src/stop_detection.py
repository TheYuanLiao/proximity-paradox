"""Step 1: GPS pings to stays via Infostop + PySpark.

For each device group (from Step 0), loads all day-level parquet files,
applies Infostop stop detection per device using Spark's applyInPandas,
aggregates continuous stays at the same location, and writes stop records
with local-time columns for downstream HoWDe home/work detection.

Input:
    {input_dir}/grp_{N}/{month}_{day}.parquet
    Columns: timestamp, device_aid, latitude, longitude, location_method, grp

Output:
    {output_dir}/stops_{batch}.parquet
    Columns: device_aid, loc, start, end, latitude, longitude,
             size, batch, localtime, l_localtime, duration_min

Usage:
    # Single batch
    python src/stop_detection.py --country sweden \
        --input-dir /data/mobile/se_2024/format_parquet \
        --output-dir /data/mobile/se_2024/stops \
        --batch 0

    # Batch range (0 through 9)
    python src/stop_detection.py --country sweden \
        --input-dir /data/mobile/se_2024/format_parquet \
        --output-dir /data/mobile/se_2024/stops \
        --batch 0 10

    # All batches
    python src/stop_detection.py --country sweden \
        --input-dir /data/mobile/se_2024/format_parquet \
        --output-dir /data/mobile/se_2024/stops \
        --all

    # Merge all batch outputs
    python src/stop_detection.py --country sweden \
        --output-dir /data/mobile/se_2024/stops \
        --merge

Reference:
    geo-social-mixing/src/data/stop_detection.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from infostop import Infostop
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

import yaml

ROOT_DIR = Path(__file__).parent.parent


def load_merged_config(country):
    """Load default + country YAML configs, merged. No pydantic dependency."""
    config_dir = ROOT_DIR / "config"
    with open(config_dir / "default.yaml") as f:
        defaults = yaml.safe_load(f)
    with open(config_dir / "countries" / f"{country}.yaml") as f:
        country_cfg = yaml.safe_load(f)
    merged = {**defaults, **country_cfg}
    for key in defaults:
        if key in country_cfg and isinstance(defaults[key], dict) and isinstance(country_cfg[key], dict):
            merged[key] = {**defaults[key], **country_cfg[key]}
    return merged


# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------

def init_spark(driver_memory="56g", executor_memory="8g", cores=18):
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    conf = SparkConf().setMaster(f"local[{cores}]").setAppName("StopDetection")
    conf.set("spark.executor.heartbeatInterval", "3600s")
    conf.set("spark.network.timeout", "7200s")
    conf.set("spark.sql.files.ignoreCorruptFiles", "true")
    conf.set("spark.driver.memory", driver_memory)
    conf.set("spark.driver.maxResultSize", "0")
    conf.set("spark.executor.memory", executor_memory)
    conf.set("spark.memory.fraction", "0.6")
    conf.set("spark.sql.session.timeZone", "UTC")

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    print(f"Java: {spark._jvm.System.getProperty('java.version')}")
    print(f"Spark UI: {spark.sparkContext.uiWebUrl}")
    return spark


# ---------------------------------------------------------------------------
# Infostop per-user function
# ---------------------------------------------------------------------------

INFOSTOP_OUTPUT_SCHEMA = StructType([
    StructField("device_aid", StringType()),
    StructField("timestamp", LongType()),
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("loc", IntegerType()),
    StructField("stop_latitude", DoubleType()),
    StructField("stop_longitude", DoubleType()),
    StructField("interval", IntegerType()),
])

INFOSTOP_OUTPUT_COLS = [
    "device_aid", "timestamp", "latitude", "longitude",
    "loc", "stop_latitude", "stop_longitude", "interval",
]


def make_infostop_fn(r1, r2, min_staying_time_min, max_time_between_h):
    """Create a per-user Infostop function with parameters baked into the closure.

    Returns a function compatible with Spark's applyInPandas.
    """
    min_staying_time_s = min_staying_time_min * 60
    max_time_between_s = max_time_between_h * 3600

    def infostop_per_user(key, data):
        empty = pd.DataFrame([], columns=INFOSTOP_OUTPUT_COLS)

        # Filter invalid coordinates
        x = data.loc[
            data["latitude"].between(-80, 84)
            & data["longitude"].between(-180, 180)
        ].copy()

        x = (
            x.sort_values("timestamp")
            .drop_duplicates(subset=["latitude", "longitude", "timestamp"])
            .dropna()
            .reset_index(drop=True)
        )

        if len(x) < 2:
            return empty

        # Interpolate across large time gaps so Infostop handles
        # discontinuous data (identical to geo-social-mixing approach)
        x["t_seg"] = x["timestamp"].shift(-1)
        x.loc[x.index[-1], "t_seg"] = x.loc[x.index[-1], "timestamp"] + 1
        x["n"] = x.apply(
            lambda row: range(
                int(row["timestamp"]),
                min(int(row["t_seg"]), int(row["timestamp"]) + max_time_between_s),
                max_time_between_s - 1,
            ),
            axis=1,
        )
        x = x.explode("n")
        x["timestamp"] = x["n"].astype(float)
        x = x[["latitude", "longitude", "timestamp"]].dropna()

        if len(x) < 2:
            return empty

        model = Infostop(
            r1=r1,
            r2=r2,
            label_singleton=True,
            min_staying_time=min_staying_time_s,
            max_time_between=max_time_between_s,
            min_size=2,
            min_spacial_resolution=0,
            distance_metric="haversine",
            weighted=False,
            weight_exponent=1,
            verbose=False,
        )

        try:
            labels = model.fit_predict(
                x[["latitude", "longitude", "timestamp"]].values
            )
        except Exception:
            return empty

        label_medians = model.compute_label_medians()
        x["loc"] = labels
        x["same_loc"] = x["loc"] == x["loc"].shift()
        x["little_time"] = (
            x["timestamp"] - x["timestamp"].shift()
        ) < max_time_between_s
        x["interval"] = (~(x["same_loc"] & x["little_time"])).cumsum()

        latitudes = {k: v[0] for k, v in label_medians.items()}
        longitudes = {k: v[1] for k, v in label_medians.items()}
        x["stop_latitude"] = x["loc"].map(latitudes)
        x["stop_longitude"] = x["loc"].map(longitudes)
        x["device_aid"] = key[0]

        # loc > 0 are actual stops; -1 is moving/noise
        x = x[x["loc"] > 0].copy()
        return x[INFOSTOP_OUTPUT_COLS]

    return infostop_per_user


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class StopDetection:
    def __init__(self, spark, input_dir, output_dir, config):
        self.spark = spark
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        sd = config["stop_detection"]
        self.r1 = sd["r1"]
        self.r2 = sd["r2"]
        self.tmin = sd["tmin"]
        self.max_time_between = sd["max_time_between"]
        self.utc_offset = config.get("utc_offset_seconds", 0)
        self.n_groups = config["raw_gps"]["n_groups"]

        self.file_paths_dict = {}

    def build_file_list(self):
        for grp in range(self.n_groups):
            grp_path = self.input_dir / f"grp_{grp}"
            if grp_path.exists():
                files = sorted(grp_path.glob("*.parquet"))
                self.file_paths_dict[grp] = [str(f) for f in files]
                print(f"Group {grp}: {len(files)} files")
            else:
                print(f"Warning: {grp_path} not found")

    def process_batch(self, batch, resume=False):
        if batch not in self.file_paths_dict:
            print(f"Error: batch {batch} not in file list. "
                  "Run build_file_list() first.")
            return

        out_path = self.output_dir / f"stops_{batch}.parquet"
        if resume and out_path.exists() and out_path.stat().st_size > 0:
            print(f"Skipping batch {batch} (output exists: {out_path})")
            return

        print(f"\n{'=' * 60}")
        print(f"Processing batch {batch}")
        print(f"{'=' * 60}")
        start_time = time.time()

        file_paths = self.file_paths_dict[batch]
        print(f"Loading {len(file_paths)} files...")

        df = self.spark.read.parquet(*file_paths).select(
            "device_aid", "timestamp", "latitude", "longitude"
        )
        n_records = df.count()
        n_devices = df.select("device_aid").distinct().count()
        print(f"Loaded {n_records:,} records from {n_devices:,} devices")

        infostop_fn = make_infostop_fn(
            self.r1, self.r2, self.tmin, self.max_time_between
        )

        print("Applying Infostop...")
        stops = df.groupby("device_aid").applyInPandas(
            infostop_fn, schema=INFOSTOP_OUTPUT_SCHEMA
        )

        print("Aggregating stops...")
        stop_agg = stops.groupby("device_aid", "interval").agg(
            F.first("loc").alias("loc"),
            F.min("timestamp").alias("start"),
            F.max("timestamp").alias("end"),
            F.first("stop_latitude").alias("latitude"),
            F.first("stop_longitude").alias("longitude"),
            F.count("loc").alias("size"),
        )

        # Local-time columns for HoWDe
        stop_agg = (
            stop_agg
            .withColumn(
                "localtime",
                (F.col("start") + F.lit(self.utc_offset)).cast(LongType()),
            )
            .withColumn(
                "l_localtime",
                (F.col("end") + F.lit(self.utc_offset)).cast(LongType()),
            )
        )

        df_stops = stop_agg.toPandas()
        df_stops["batch"] = batch
        df_stops["duration_min"] = (df_stops["end"] - df_stops["start"]) / 60

        out_path = self.output_dir / f"stops_{batch}.parquet"
        df_stops.to_parquet(out_path, index=False)

        elapsed = (time.time() - start_time) / 60
        n_stops = len(df_stops)
        n_users = df_stops["device_aid"].nunique() if n_stops > 0 else 0
        print(f"\nBatch {batch} complete:")
        print(f"  Stops:   {n_stops:,}")
        print(f"  Devices: {n_users:,}")
        print(f"  Time:    {elapsed:.1f} min")
        print(f"  Output:  {out_path}")

    def process_all(self, start_batch=0, end_batch=None, resume=False):
        if end_batch is None:
            end_batch = self.n_groups
        total_start = time.time()
        for batch in range(start_batch, end_batch):
            self.process_batch(batch, resume=resume)
        total_elapsed = (time.time() - total_start) / 60
        print(f"\n{'=' * 60}")
        print(f"All batches complete in {total_elapsed:.1f} minutes")
        print(f"{'=' * 60}")

    def merge_all_stops(self, output_file="stops_all.parquet"):
        print("Merging stop files...")
        all_files = sorted(self.output_dir.glob("stops_*.parquet"))
        all_files = [f for f in all_files if f.name != output_file]
        if not all_files:
            print("No stop files found.")
            return

        dfs = []
        for f in all_files:
            print(f"  {f.name}")
            dfs.append(pd.read_parquet(f))

        merged = pd.concat(dfs, ignore_index=True)
        out_path = self.output_dir / output_file
        merged.to_parquet(out_path, index=False)
        print(f"Merged {len(all_files)} files → {len(merged):,} stops, "
              f"{merged['device_aid'].nunique():,} devices → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stop detection via Infostop + PySpark"
    )
    parser.add_argument(
        "--country", type=str, default="sweden",
        help="Country config to load (default: sweden)",
    )
    parser.add_argument(
        "--input-dir", type=str,
        help="format_parquet directory (overrides config)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Stops output directory (overrides config)",
    )
    parser.add_argument(
        "--batch", type=int, nargs="+",
        help="Batch number(s): single N, or start end range",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all batches",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all existing batch files into one",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip batches whose output file already exists",
    )
    parser.add_argument(
        "--cores", type=int,
        help="Spark cores (overrides config)",
    )
    parser.add_argument(
        "--memory", type=str,
        help="Spark driver memory, e.g. '56g' (overrides config)",
    )

    args = parser.parse_args()

    config = load_merged_config(args.country)

    input_dir = args.input_dir or config["raw_gps"]["grouped_dir"]
    output_dir = args.output_dir or str(
        Path(config["data_root"]) / "dbs" / config["country"] / "stops"
    )

    spark_cfg = config["spark"]
    cores = args.cores or spark_cfg["cores"]
    memory = args.memory or spark_cfg["driver_memory"]

    spark = init_spark(
        driver_memory=memory,
        executor_memory=spark_cfg["executor_memory"],
        cores=cores,
    )

    sd = StopDetection(spark, input_dir, output_dir, config)
    sd.build_file_list()

    if args.merge:
        sd.merge_all_stops()
    elif args.all:
        sd.process_all(resume=args.resume)
    elif args.batch:
        if len(args.batch) == 1:
            sd.process_batch(args.batch[0], resume=args.resume)
        elif len(args.batch) == 2:
            sd.process_all(
                start_batch=args.batch[0], end_batch=args.batch[1],
                resume=args.resume,
            )
        else:
            for b in args.batch:
                sd.process_batch(b, resume=args.resume)
    else:
        parser.print_help()

    spark.stop()


if __name__ == "__main__":
    main()
