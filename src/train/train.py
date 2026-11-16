"""GPU training task: YOLOv8 fine-tune on Serverless GPU (AI Runtime, 1x A10).

Runs the same way in both paths:
  * DABs:  ai_runtime_task -> command.sh -> python train.py
  * CLI:   databricks air run -f train.yaml --watch

Steps:
  1. Copy the curated YOLO dataset from the UC Volume to local NVMe.
  2. Fine-tune a small pretrained YOLOv8 model (yolov8s by default).
  3. On rank 0: log params, per-epoch metrics, and artifacts to MLflow.
     Checkpoints are mirrored to the artifacts Volume. The model is
     registered in Unity Catalog with the alias @champion.
"""

import argparse
import json
import os
import re
import shutil
import tarfile
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so yolo_pyfunc imports regardless of cwd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="cmegdemos_catalog")
    p.add_argument("--schema", default="mlb_air_migration")
    p.add_argument("--model_name", default="mlb_yolo_detector")
    p.add_argument("--base_model", default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--experiment", default="/Shared/mlb-air-yolo",
                   help="only used when the launcher did not already set an MLflow run")
    return p.parse_known_args()[0]


def clean(k):
    """'metrics/mAP50(B)' -> 'metrics/mAP50' (MLflow-safe metric names)."""
    return re.sub(r"[^\w\-./ ]", "", k.replace("(B)", ""))


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", os.environ.get("NODE_RANK", "0")))
    vol = Path(f"/Volumes/{args.catalog}/{args.schema}")
    src_data = vol / "raw_images" / "yolo"
    ckpt_dir = vol / "artifacts" / "checkpoints" / time.strftime("%Y%m%d-%H%M%S")
    full_model_name = f"{args.catalog}.{args.schema}.{args.model_name}"

    print("MLflow/launcher env:", sorted(k for k in os.environ if k.startswith(("MLFLOW_", "CODE_SOURCE", "NUM_NODES", "LOCAL_WORLD"))))

    # 1) Volume -> local disk. Copy ONE tar file and extract it locally: per-file reads through
    #    the Volume mount are slow (thousands of small images), a single sequential read is fast.
    t0 = time.time()
    work = Path("/tmp/yolo_work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    tar_src = vol / "raw_images" / "yolo_dataset.tar"
    if tar_src.exists():
        shutil.copyfile(tar_src, work / "yolo_dataset.tar")
        with tarfile.open(work / "yolo_dataset.tar") as tar:
            tar.extractall(work, filter="data")
    else:  # fallback: per-file copy (slow)
        shutil.copytree(src_data, work / "yolo")
    local = work / "yolo"
    print(f"Dataset copied to local disk in {time.time() - t0:.0f}s")
    data_cfg = yaml.safe_load((local / "data.yaml").read_text())
    data_cfg["path"] = str(local)
    (local / "data.yaml").write_text(yaml.safe_dump(data_cfg, sort_keys=False))
    n_train = len(list((local / "train" / "images").iterdir()))
    n_val = len(list((local / "val" / "images").iterdir()))
    print(f"Dataset: {n_train} train / {n_val} val images, classes={data_cfg['names']}")

    from ultralytics import YOLO, settings
    import ultralytics

    settings.update({"mlflow": False})  # we log explicitly below (one run, rank 0 only)

    model = YOLO(args.base_model)

    # Checkpoints -> UC Volume, so a run can be inspected or resumed later
    def mirror_checkpoint(trainer):
        if rank == 0 and (trainer.epoch + 1) % 5 == 0 and Path(trainer.last).exists():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(trainer.last, ckpt_dir / "last.pt")

    def log_epoch(trainer):
        if rank == 0:
            m = {clean(k): float(v) for k, v in trainer.metrics.items()}
            if trainer.tloss is not None:
                losses = trainer.label_loss_items(trainer.tloss, prefix="train")
                m.update({clean(k): float(v) for k, v in losses.items()})
            mlflow.log_metrics(m, step=trainer.epoch)

    model.add_callback("on_model_save", mirror_checkpoint)
    model.add_callback("on_fit_epoch_end", log_epoch)

    if not os.environ.get("MLFLOW_RUN_ID") and not os.environ.get("MLFLOW_EXPERIMENT_ID"):
        mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:  # resumes the launcher's run when MLFLOW_RUN_ID is set
        if rank == 0:
            mlflow.log_params({**vars(args), "n_train": n_train, "n_val": n_val,
                               "ultralytics": ultralytics.__version__})

        t0 = time.time()
        model.train(data=str(local / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, workers=args.workers, device=0, seed=0,
                    project="/tmp/yolo_runs", name="train", exist_ok=True, plots=True)
        train_minutes = (time.time() - t0) / 60

        if rank != 0:
            return

        save_dir = Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"

        # Final validation on the best checkpoint
        best_model = YOLO(str(best))
        metrics = best_model.val(data=str(local / "data.yaml"), split="val", imgsz=args.imgsz, verbose=False)
        final = {
            "val_mAP50": float(metrics.box.map50),
            "val_mAP50_95": float(metrics.box.map),
            "val_precision": float(metrics.box.mp),
            "val_recall": float(metrics.box.mr),
            "train_minutes": train_minutes,
        }
        for i, name in data_cfg["names"].items():
            if i < len(metrics.box.maps):
                final[f"val_mAP50_95_{name}"] = float(metrics.box.maps[i])
        mlflow.log_metrics(final)
        print("Final metrics:", json.dumps(final, indent=2))

        # Artifacts: plots, results.csv, weights
        for f in save_dir.glob("*.png"):
            mlflow.log_artifact(str(f), artifact_path="plots")
        if (save_dir / "results.csv").exists():
            mlflow.log_artifact(str(save_dir / "results.csv"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, ckpt_dir / "best.pt")
        shutil.copyfile(save_dir / "weights" / "last.pt", ckpt_dir / "last.pt")

        # Register in Unity Catalog as a pyfunc (input: image_path -> output: detections JSON)
        from mlflow.models import infer_signature
        from yolo_pyfunc import YoloDetector

        mlflow.set_registry_uri("databricks-uc")
        signature = infer_signature(
            pd.DataFrame({"image_path": ["/Volumes/c/s/v/img.jpg"]}),
            pd.DataFrame({"detections": ['[{"cls": 0, "name": "ball", "conf": 0.9, "xyxy": [0, 0, 1, 1]}]']}),
        )
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=YoloDetector(),
            artifacts={"weights": str(best)},
            code_paths=[str(HERE / "yolo_pyfunc.py")],
            pip_requirements=[f"ultralytics=={ultralytics.__version__}", "pandas"],
            signature=signature,
            registered_model_name=full_model_name,
        )
        version = info.registered_model_version
        mlflow.MlflowClient().set_registered_model_alias(full_model_name, "champion", version)

        summary = {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id,
                   "model": full_model_name, "version": version, "checkpoints": str(ckpt_dir), **final}
        (vol / "artifacts" / "train_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Registered {full_model_name} v{version} (@champion)")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
