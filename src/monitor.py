# src/monitor.py

import os
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# =========================
# Paths
# =========================
REFERENCE_PATH = "data/train_data.csv"
CURRENT_PATH = "logs/inference_logs.csv"

REPORT_DIR = "reports"
REPORT_PATH = f"{REPORT_DIR}/drift_report.html"
DRIFT_THRESHOLD = 0.3

# =========================
# LOAD DATA
# =========================
def load_data():
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(
            f"❌ Không tìm thấy reference data: {REFERENCE_PATH}"
        )
    
    if not os.path.exists(CURRENT_PATH):
        raise FileNotFoundError(
            f"❌ Không tìm thấy inference logs: {CURRENT_PATH}"
        )
    
    reference_data = pd.read_csv(REFERENCE_PATH)
    current_data = pd.read_csv(CURRENT_PATH)

    return reference_data, current_data

# =========================
# PREPROCESS
# =========================

def preprocess(reference_data, current_data):

    feature_cols = [
        "thu_nhap",
        "so_tien_vay",
        "lich_su_no_xau"
    ]

    reference_data = reference_data[feature_cols]
    current_data = current_data[feature_cols]

    return reference_data, current_data

# =========================
# CREATE REPORT
# =========================

def generate_report(reference_data, current_data):

    report = Report(metrics = [DataDriftPreset()])

    report.run(
        reference_data = reference_data,
        current_data = current_data
    )

    return report

# =========================
# SAVE HTML REPORT
# =========================

def save_report(report):
    os.makedirs(REPORT_DIR, exist_ok= True)

    report.save_html(REPORT_PATH)

    print(f"\n📄 Report saved:")
    print(f"{REPORT_PATH}")

# =========================
# ANALYZE DRIFT
# =========================

def analyze_drift(report):
    result = report.as_dict()

    drift_result = result["metrics"][0]["result"]

    dataset_drift = drift_result["dataset_drift"]

    drift_share = drift_result["share_of_drifted_columns"]

    number_of_drifted_columns = drift_result["number_of_drifted_columns"]

    total_columns = drift_result["number_of_columns"]

    print("\n==============================")
    print("📊 DRIFT ANALYSIS")
    print("==============================")

    print(f"Dataset Drift: {dataset_drift}")
    print(f"Drifted Columns: {number_of_drifted_columns}/{total_columns}")
    print(f"Drift Share: {drift_share:.2%}")

    if drift_share >= DRIFT_THRESHOLD:

        print("\n🚨 ALERT!")
        print(
            f"Drift vượt ngưỡng "
            f"({drift_share:.2%} >= {DRIFT_THRESHOLD:.0%})"
        )

        print("👉 Khuyến nghị:")
        print("- Kiểm tra dữ liệu production")
        print("- Xem xét retrain model")

    else:

        print("\n✅ Model ổn định.")
        print("Drift chưa vượt ngưỡng.")      

# =========================
# MAIN
# =========================

def main():

    print("\n🔍 Starting Monitoring Pipeline...")

    # Load
    reference_data, current_data = load_data()

    print("✅ Data loaded")

    # Preprocess
    reference_data, current_data = preprocess(
        reference_data,
        current_data
    )

    print("✅ Data preprocessed")

    # Generate report
    report = generate_report(
        reference_data,
        current_data
    )

    print("✅ Drift report generated")

    # Save HTML
    save_report(report)

    # Analyze drift
    analyze_drift(report)

    print("\n🎉 Monitoring completed.")


if __name__ == "__main__":
    main()