from prefect import flow, task
import subprocess
import os
from datetime import datetime

@task
def check_drift_task():
    from mlops.drift_detection import check_drift
    print("Checking for drift...")
    return check_drift()

@task
def download_new_data():
    print("Downloading fresh data...")
    result = subprocess.run(
        ["python", "src/data/download.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    return result.returncode == 0

@task
def augment_data():
    print("Augmenting data...")
    result = subprocess.run(
        ["python", "src/data/augment.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    return result.returncode == 0

@task
def prepare_yolo_data():
    print("Preparing YOLO dataset...")
    result = subprocess.run(
        ["python", "src/data/prepare_yolo.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    return result.returncode == 0

@task
def retrain_model():
    print("Retraining YOLO model...")
    result = subprocess.run(
        ["python", "src/models/train_yolo.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    return result.returncode == 0

@task
def rebuild_index():
    print("Rebuilding FAISS index...")
    result = subprocess.run(
        ["python", "src/models/build_index.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    return result.returncode == 0

@task
def notify(success: bool):
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"\nRetraining Pipeline {status}")
    print(f"Timestamp: {datetime.utcnow()}")

@flow(name="asset-intel-retraining")
def retraining_pipeline(force: bool = False):
    print("="*50)
    print("Asset Intel Retraining Pipeline")
    print("="*50)

    # Check drift first
    drift = check_drift_task()

    if not drift and not force:
        print("No drift detected — skipping retraining")
        return

    print("Starting retraining pipeline...")

    # Run pipeline stages
    data_ok = download_new_data()
    if not data_ok:
        notify(False)
        return

    aug_ok = augment_data()
    prep_ok = prepare_yolo_data()
    train_ok = retrain_model()
    index_ok = rebuild_index()

    success = all([data_ok, aug_ok, prep_ok, train_ok, index_ok])
    notify(success)

if __name__ == "__main__":
    # Run manually to test
    retraining_pipeline(force=True)