import os
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
from ultralytics import settings
import torch
import shutil
from pathlib import Path

torch.serialization.add_safe_globals([__import__('ultralytics').nn.tasks.DetectionModel])
settings.update({"mlflow": False})

def train_yolo():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("yolo-hardware-detection")

    with mlflow.start_run(run_name="yolov8m-v1"):
        mlflow.log_params({
            "model": "yolov8m",
            "epochs": 30,
            "imgsz": 640,
            "batch": 8,
        })

        model = YOLO("yolov8m.pt", task="detect")

        results = model.train(
            data=os.path.abspath("data/yolo/dataset.yaml"),
            epochs=30,
            imgsz=640,
            batch=8,
            device=device,
            project="runs/detect",
            name="yolo-v1",
            exist_ok=True,
            verbose=True
        )

        best_pt = None
        for pt in Path("runs").rglob("best.pt"):
            best_pt = str(pt)
            print(f"Found weights → {best_pt}")
            break

        # Copy to models/ for DVC
        dest = "models/yolo/v1/weights/best.pt"
        os.makedirs("models/yolo/v1/weights", exist_ok=True)

        if best_pt:
            shutil.copy(best_pt, dest)
            mlflow.log_artifact(dest)
            print(f"Weights saved → {dest}")
        else:
            print("best.pt not found anywhere in runs/")

        # Log metrics
        metrics = results.results_dict
        mlflow.log_metrics({
            "mAP50":     metrics.get("metrics/mAP50(B)", 0),
            "mAP50-95":  metrics.get("metrics/mAP50-95(B)", 0),
            "precision": metrics.get("metrics/precision(B)", 0),
            "recall":    metrics.get("metrics/recall(B)", 0)
        })

        print(f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.3f}")

if __name__ == "__main__":
    train_yolo()