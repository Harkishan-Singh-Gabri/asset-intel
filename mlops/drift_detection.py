import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sqlalchemy import create_engine
from evidently import Report
from evidently.presets import DataDriftPreset

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./asset_intel.db")

def load_production_data():
    engine = create_engine(DATABASE_URL)
    try:
        scans = pd.read_sql(
            "SELECT clip_score, confidence, flagged FROM scans ORDER BY scanned_at DESC LIMIT 100",
            engine
        )
        return scans
    except Exception as e:
        print(f"No production data yet: {e}")
        return None

def check_drift():
    print("="*50)
    print("Running Drift Detection...")
    print("="*50)

    prod = load_production_data()

    if prod is None or len(prod) < 10:
        print("Not enough production data yet")
        print(f"Need at least 10 scans, have {len(prod) if prod is not None else 0}")
        return False

    # Check performance drift via clip scores
    avg_clip_score = prod["clip_score"].mean()
    flag_rate = prod["flagged"].mean()

    print(f"Total scans analysed : {len(prod)}")
    print(f"Average CLIP score   : {avg_clip_score:.3f}")
    print(f"Flag rate            : {flag_rate:.1%}")

    # Drift conditions
    drift_detected = False

    if avg_clip_score < 0.65:
        print("DRIFT DETECTED: Low CLIP scores")
        drift_detected = True

    if flag_rate > 0.40:
        print("DRIFT DETECTED: High flag rate")
        drift_detected = True

    if not drift_detected:
        print("No drift detected")

    # Save report
    report_data = {
        "timestamp": str(datetime.utcnow()),
        "avg_clip_score": float(avg_clip_score),
        "flag_rate": float(flag_rate),
        "drift_detected": drift_detected,
        "total_scans": len(prod)
    }

    os.makedirs("mlops/reports", exist_ok=True)
    with open("mlops/reports/drift_report.json", "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"Report saved -> mlops/reports/drift_report.json")
    return drift_detected

if __name__ == "__main__":
    drift = check_drift()
    if drift:
        print("\nRecommendation: Retrain model with new data")