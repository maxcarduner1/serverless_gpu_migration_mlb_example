"""Publish task (serverless CPU): append the latest GPU batch-inference output to Delta.

Reads /Volumes/<catalog>/<schema>/artifacts/inference/<LATEST>/predictions.parquet (written by
the GPU `infer` task) and appends it to <catalog>.<schema>.yolo_predictions.
"""

import argparse
from pathlib import Path

from pyspark.sql import SparkSession


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="cmegdemos_catalog")
    p.add_argument("--schema", default="mlb_air_migration")
    args = p.parse_known_args()[0]

    base = Path(f"/Volumes/{args.catalog}/{args.schema}/artifacts/inference")
    ts = (base / "LATEST").read_text().strip()
    src = base / ts / "predictions.parquet"
    table = f"{args.catalog}.{args.schema}.yolo_predictions"

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet(str(src))
    df.write.mode("append").option("mergeSchema", "true").saveAsTable(table)
    print(f"Appended {df.count()} detections from {src} -> {table}")
    spark.sql(f"""SELECT class_name, count(*) AS detections, round(avg(confidence), 3) AS avg_conf
                  FROM {table} WHERE inference_ts = '{ts}' GROUP BY class_name ORDER BY 1""").show()


if __name__ == "__main__":
    main()
