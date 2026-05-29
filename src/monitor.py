"""
Mô-đun Giám sát & Phát hiện Drift (Monitoring & Drift Detection)
=================================================================
Mục đích: Phát hiện data drift bằng cách so sánh dữ liệu huấn luyện gốc
với dữ liệu logs API (dự đoán thực tế) bằng thư viện Evidently AI.

Luồng hoạt động:
  1. Đọc dữ liệu huấn luyện gốc (reference) từ data/train_data.csv
  2. Đọc dữ liệu logs từ API (current) từ logs/inference_logs.csv
  3. Load mô hình từ models/model.pkl
  4. Tính toán dự đoán bằng mô hình
  5. Chạy Evidently DataDriftPreset & ClassificationPreset
  6. Tạo báo cáo HTML chi tiết
  7. Nếu drift vượt ngưỡng: hỏi người dùng có muốn retrain không
  8. Nếu có: merge dữ liệu, backup mô hình cũ, retrain mô hình mới

Dependencies: pandas, numpy, joblib, evidently, subprocess, webbrowser

Chạy: python src/monitor.py
      (Yêu cầu: data/train_data.csv, logs/inference_logs.csv, models/model.pkl)
"""

import os
import sys
import json
import subprocess
import shutil
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently.core.datasets import BinaryClassification


# ============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train_data.csv"
LOGS_PATH = BASE_DIR / "logs" / "inference_logs.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
MODEL_BACKUP_DIR = BASE_DIR / "models" / "backups"

REPORT_PATH = BASE_DIR / "reports" / "drift_report.html"
BEFORE_RETRAIN_REPORT = BASE_DIR / "reports" / "before_retrain_report.html"
AFTER_RETRAIN_REPORT = BASE_DIR / "reports" / "after_retrain_report.html"

TRAIN_SCRIPT = BASE_DIR / "src" / "train.py"

# ============================================================================
# CẤU HÌNH DỮ LIỆU
# ============================================================================

FEATURE_COLS = [
    "thu_nhap",
    "so_tien_vay",
    "thoi_han_vay",
    "diem_tin_dung",
    "tra_hang_thang"
]
TARGET_COL = "lich_su_no_xau"
PRED_COL = "prediction"

# Ngưỡng phát hiện drift (tỷ lệ cột bị drift)
DRIFT_THRESHOLD = 0.3  # 30%

# Chia cách hiển thị
SEP = "=" * 70


# ============================================================================
# SECTION 1: LẤY & LOAD DỮ LIỆU
# ============================================================================

def load_data(data_path: Path) -> pd.DataFrame:
    """
    Đọc dữ liệu từ file CSV.

    Thông số:
    ----------
    data_path : Path
        Đường dẫn tới file CSV

    Trả về:
    -------
    pd.DataFrame
        Dữ liệu đã đọc

    Ngoại lệ:
    ---------
    FileNotFoundError: Nếu file không tồn tại
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[OK] Đã tải dữ liệu từ: {data_path} ({len(df):,} hàng)")
    return df


def load_model(model_path: Path):
    """
    Đọc mô hình từ file .pkl.

    Thông số:
    ----------
    model_path : Path
        Đường dẫn tới file model.pkl

    Trả về:
    -------
    object
        Mô hình RandomForest

    Ngoại lệ:
    ---------
    FileNotFoundError: Nếu file không tồn tại
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {model_path}")

    model = joblib.load(str(model_path))
    print(f"[OK] Đã tải mô hình từ: {model_path}")
    return model


# ============================================================================
# SECTION 2: XỬ LÝ & CHUẨN HÓA DỮ LIỆU
# ============================================================================

def calculate_monthly_payment(df: pd.DataFrame) -> pd.Series:
    """
    Tính tiền trả hàng tháng cho từng hàng dữ liệu.

    Công thức: P * r * (1+r)^n / ((1+r)^n - 1)

    Thông số:
    ----------
    df : pd.DataFrame
        DataFrame chứa các cột: so_tien_vay, thoi_han_vay, diem_tin_dung

    Trả về:
    -------
    pd.Series
        Cột tiền trả hàng tháng
    """
    so_tien_vay = df["so_tien_vay"]
    thoi_han_vay = df["thoi_han_vay"].replace(0, pd.NA)
    diem_tin_dung = df["diem_tin_dung"]

    # Lãi suất năm
    lai_suat_nam = 0.08 + (1 - (diem_tin_dung / 850)) * 0.15
    lai_suat_thang = lai_suat_nam / 12

    # EMI
    tks = (1 + lai_suat_thang) ** thoi_han_vay
    tra_hang_thang = (so_tien_vay * lai_suat_thang * tks / (tks - 1))

    return tra_hang_thang


def preprocess_for_evidently(data: pd.DataFrame) -> Dataset:
    """
    Chuẩn bị dữ liệu cho Evidently (xóa NA, định nghĩa schema).

    Thông số:
    ----------
    data : pd.DataFrame
        Dữ liệu gốc

    Trả về:
    -------
    Dataset
        Đối tượng Evidently Dataset
    """
    # Giữ lại các cột cần thiết
    required_cols = FEATURE_COLS + [TARGET_COL, PRED_COL]
    data_clean = data[required_cols].dropna().copy()

    # Định nghĩa schema cho Evidently
    data_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target=TARGET_COL,
                prediction_labels=PRED_COL,
            )
        ]
    )

    # Chuyển sang Evidently Dataset
    evid_dataset = Dataset.from_pandas(
        data_clean,
        data_definition=data_definition
    )

    return evid_dataset


# ============================================================================
# SECTION 3: TẠO & LƯU BÁOÁO CÁO
# ============================================================================

def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame
) -> Report:
    """
    Sinh báo cáo Drift so sánh hai tập dữ liệu.

    Thông số:
    ----------
    reference_data : pd.DataFrame
        Dữ liệu gốc (train data)
    current_data : pd.DataFrame
        Dữ liệu hiện tại (logs data)

    Trả về:
    -------
    Report
        Đối tượng Report từ Evidently
    """
    ref_evid = preprocess_for_evidently(reference_data)
    cur_evid = preprocess_for_evidently(current_data)

    # Tạo report với DataDrift + Classification metrics
    report = Report([DataDriftPreset(), ClassificationPreset()])
    result = report.run(reference_data=ref_evid, current_data=cur_evid)

    return result


def save_report(result: Report, save_path: Path) -> None:
    """
    Lưu báo cáo HTML.

    Thông số:
    ----------
    result : Report
        Đối tượng Report
    save_path : Path
        Đường dẫn file HTML output
    """
    os.makedirs(save_path.parent, exist_ok=True)
    result.save_html(str(save_path))
    print(f"[OK] Báo cáo đã được lưu: {save_path}")


# ============================================================================
# SECTION 4: PHÂN TÍCH DRIFT
# ============================================================================

def extract_drift_share(result: Report) -> float:
    """
    Trích xuất tỷ lệ drift từ Report (hỗ trợ nhiều phiên bản Evidently).

    Thông số:
    ----------
    result : Report
        Đối tượng Report từ Evidently

    Trả về:
    -------
    float
        Tỷ lệ cột bị drift (0.0 - 1.0)
    """
    try:
        raw = json.loads(result.json())
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể parse JSON: {e}")
        return 0.0

    metrics = raw.get("metrics", [])

    # Thử nhiều cách trích xuất (tương thích nhiều phiên bản)
    for metric_block in metrics:
        candidates = [
            lambda m: m.get("value", {}).get("share_of_drifted_columns"),
            lambda m: m.get("value", {}).get("share"),
            lambda m: m.get("result", {}).get("share_of_drifted_columns"),
            lambda m: m.get("result", {}).get("share"),
        ]

        for candidate_fn in candidates:
            try:
                val = candidate_fn(metric_block)
                if val is not None and isinstance(val, (int, float)):
                    return float(val)
            except (KeyError, TypeError):
                continue

    # Fallback: đếm số cột bị drift
    for metric_block in metrics:
        try:
            cols = metric_block["value"]["drift_by_columns"]
            if cols:
                drifted_count = sum(
                    1 for v in cols.values() if v.get("drift_detected", False)
                )
                return drifted_count / len(cols)
        except (KeyError, TypeError):
            continue

    return 0.0


def analyze_drift(
    result: Report,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    model,
    drift_threshold: float = DRIFT_THRESHOLD
) -> None:
    """
    Phân tích drift và xác định có cần retrain không.

    Thông số:
    ----------
    result : Report
        Báo cáo drift từ Evidently
    reference_data : pd.DataFrame
        Dữ liệu gốc
    current_data : pd.DataFrame
        Dữ liệu logs
    model : object
        Mô hình RandomForest
    drift_threshold : float
        Ngưỡng cảnh báo (0-1)
    """
    drift_share = extract_drift_share(result)

    print(f"\n{'─' * 70}")
    print("  KẾT QUẢ PHÂN TÍCH DATA DRIFT")
    print(f"{'─' * 70}")
    print(f"  📊 Tỷ lệ cột bị drift: {drift_share:.2%}")
    print(f"  ⚠️  Ngưỡng cảnh báo:   {drift_threshold:.0%}")

    if drift_share >= drift_threshold:
        print(f"  🚨 PHÁT HIỆN DRIFT VƯỢT NGƯỠNG!")
        print(f"{'─' * 70}")

        if ask_user_retrain():
            proactive_retrain(reference_data, current_data, model)
        else:
            print("\n  ℹ️  Người dùng chọn không retrain.")
            print(f"     Hãy theo dõi drift và chạy lại monitor.py khi cần.")

    else:
        print(f"  ✓ Hệ thống ổn định - không cần retrain.")
        print(f"{'─' * 70}\n")


def ask_user_retrain() -> bool:
    """
    Hỏi người dùng có muốn thực hiện retrain không.

    Trả về:
    -------
    bool
        True nếu người dùng đồng ý retrain
    """
    try:
        # Mở báo cáo HTML trong browser
        webbrowser.open(f"file://{REPORT_PATH.absolute()}")
    except Exception:
        pass

    response = input(
        "\n[?] Data Drift vượt ngưỡng! Bạn có muốn RETRAIN mô hình không? (y/n): "
    )
    return response.strip().lower() in ['y', 'yes']


# ============================================================================
# SECTION 5: RETRAIN TỰ ĐỘNG
# ============================================================================

def backup_model() -> None:
    """Backup mô hình cũ trước khi retrain."""
    if not MODEL_PATH.exists():
        return

    os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = MODEL_BACKUP_DIR / f"model_backup_{ts}.pkl"
    shutil.copy2(MODEL_PATH, backup_path)
    print(f"  [OK] Backup mô hình cũ: {backup_path}")


def merge_datasets_and_save(
    reference_data: pd.DataFrame,
    inference_logs: pd.DataFrame
) -> pd.DataFrame:
    """
    Gộp dữ liệu train gốc + logs mới để tạo tập huấn luyện mới.

    Thông số:
    ----------
    reference_data : pd.DataFrame
        Dữ liệu huấn luyện gốc
    inference_logs : pd.DataFrame
        Dữ liệu logs từ API

    Trả về:
    -------
    pd.DataFrame
        Dữ liệu hợp nhất
    """
    print("\n  [1/4] Gộp dữ liệu train + logs mới...")

    required_cols = FEATURE_COLS + [TARGET_COL]

    # Copy để không sửa dữ liệu gốc
    ref_work = reference_data.copy()
    log_work = inference_logs.copy()

    # Tính tra_hang_thang nếu chưa có
    if "tra_hang_thang" not in ref_work.columns:
        ref_work["tra_hang_thang"] = calculate_monthly_payment(ref_work)
    if "tra_hang_thang" not in log_work.columns:
        log_work["tra_hang_thang"] = calculate_monthly_payment(log_work)

    ref_df = ref_work[required_cols].dropna().copy()
    log_df = log_work[required_cols].dropna().copy()

    merged_df = pd.concat([ref_df, log_df], ignore_index=True)

    # Ghi đè file train_data.csv
    os.makedirs(TRAIN_PATH.parent, exist_ok=True)
    merged_df.to_csv(TRAIN_PATH, index=False)

    print(f"  [OK] Merge dữ liệu:")
    print(f"       - Reference: {len(ref_df):,} hàng")
    print(f"       - Logs:      {len(log_df):,} hàng")
    print(f"       - Tổng:      {len(merged_df):,} hàng → Ghi vào {TRAIN_PATH}")

    return merged_df


def run_train_script() -> bool:
    """
    Chạy lại train.py để retrain mô hình với dữ liệu mới.

    Trả về:
    -------
    bool
        True nếu train thành công
    """
    print("\n  [2/4] Retrain mô hình...")

    if not TRAIN_SCRIPT.exists():
        print(f"  [LỖI] Không tìm thấy: {TRAIN_SCRIPT}")
        return False

    try:
        proc = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            capture_output=False,
            text=True,
            timeout=300
        )

        if proc.returncode == 0:
            print("\n  [OK] Retrain thành công!")
            return True
        else:
            print(f"\n  [LỖI] train.py kết thúc với mã lỗi: {proc.returncode}")
            return False

    except Exception as e:
        print(f"\n  [LỖI] Lỗi khi chạy train.py: {e}")
        return False


def proactive_retrain(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    old_model
) -> None:
    """
    Thực hiện toàn bộ quy trình retrain tự động.

    Thông số:
    ----------
    reference_data : pd.DataFrame
        Dữ liệu gốc
    current_data : pd.DataFrame
        Dữ liệu logs
    old_model : object
        Mô hình cũ
    """
    print("\n" + SEP)
    print("  🔄 BẮT ĐẦU QUÁ TRÌNH RETRAIN")
    print(SEP)

    # Bước 1: Merge dữ liệu
    merged_data = merge_datasets_and_save(reference_data, current_data)

    # Bước 2: Backup mô hình cũ
    backup_model()

    # Bước 3: Retrain
    success = run_train_script()
    if not success:
        print(f"\n  [THẤT BẠI] Retrain không thành công.")
        print(f"  [ℹ️]  Mô hình cũ được giữ nguyên (backup: {MODEL_BACKUP_DIR})")
        return

    # Bước 4: Tạo báo cáo sau retrain
    print("\n  [3/4] Tạo báo cáo sau retrain...")
    new_model = load_model(MODEL_PATH)

    reference_data[PRED_COL] = new_model.predict(reference_data[FEATURE_COLS])
    current_data[PRED_COL] = new_model.predict(current_data[FEATURE_COLS])

    new_report = generate_drift_report(reference_data, current_data)
    save_report(new_report, REPORT_PATH)

    print(f"\n  [4/4] Mở báo cáo...")
    try:
        webbrowser.open(f"file://{REPORT_PATH.absolute()}")
    except Exception:
        pass

    print(f"\n{SEP}")
    print("  ✓ RETRAIN HOÀN THÀNH THÀNH CÔNG")
    print(SEP + "\n")


# ============================================================================
# SECTION 6: ENTRY POINT
# ============================================================================

def main() -> None:
    """Hàm chính để khởi chạy quá trình monitoring."""

    print(SEP)
    print("   🔍 PHÁT HIỆN VÀ GIÁM SÁT DATA DRIFT")
    print(SEP)

    try:
        # Đọc dữ liệu
        print("\n[...] Đang tải dữ liệu...")
        reference_data = load_data(TRAIN_PATH)
        current_data = load_data(LOGS_PATH)

        # Load mô hình
        print("[...] Đang tải mô hình...")
        model = load_model(MODEL_PATH)

        # Chuẩn bị dữ liệu: tính tra_hang_thang + prediction
        print("\n[...] Chuẩn bị dữ liệu cho Evidently...")

        if "tra_hang_thang" not in reference_data.columns:
            reference_data["tra_hang_thang"] = calculate_monthly_payment(
                reference_data
            )
        if "tra_hang_thang" not in current_data.columns:
            current_data["tra_hang_thang"] = calculate_monthly_payment(
                current_data
            )

        # Tính dự đoán
        reference_data[PRED_COL] = model.predict(reference_data[FEATURE_COLS])
        if PRED_COL not in current_data.columns:
            current_data[PRED_COL] = model.predict(current_data[FEATURE_COLS])

        # Sinh báo cáo
        print("[...] Sinh báo cáo drift (Evidently)...")
        result = generate_drift_report(reference_data, current_data)
        save_report(result, REPORT_PATH)

        # Phân tích drift
        analyze_drift(result, reference_data, current_data, model)

        print(f"\n{SEP}")
        print("   ✓ HOÀN THÀNH GIÁM SÁT")
        print(SEP + "\n")

    except Exception as e:
        print(f"\n[LỖI] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
