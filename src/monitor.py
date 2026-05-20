import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
import shutil
import webbrowser

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently.core.datasets import BinaryClassification

# =============================================================================
# DUONG DAN
# =============================================================================
BASE_DIR         = Path(__file__).resolve().parent.parent

TRAIN_PATH       = BASE_DIR / "data"    / "train_data.csv"
LOGS_PATH        = BASE_DIR / "logs"    / "inference_logs.csv"

MODEL_PATH       = BASE_DIR / "models"  / "model.pkl"
MODEL_BACKUP_DIR = BASE_DIR / "models"  / "backups"

REPORT_PATH           = BASE_DIR / "reports" / "drift_report.html"
BEFORE_RETRAIN_REPORT = BASE_DIR / "reports" / "before_retrain_report.html"
AFTER_RETRAIN_REPORT  = BASE_DIR / "reports" / "after_retrain_report.html"

TRAIN_SCRIPT          = BASE_DIR / "src"     / "train.py"

FEATURE_COLS = ["thu_nhap", "so_tien_vay", "thoi_han_vay", "diem_tin_dung", "tra_hang_thang"]
TARGET_COL   = "lich_su_no_xau"
PRED_COL     = "prediction"

SEP = "=" * 60

# =============================================================================
# LOAD DU LIEU VA MO HINH
# =============================================================================


def load_data(data_path: Path):
    if not data_path.exists():
        raise FileNotFoundError(f"Khong tim thay data data: {data_path}")
    
    data_data = pd.read_csv(data_path)

    print(f"[OK] Da tai du lieu:\n     - {data_path}")
    return data_data

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay model tai: {model_path}")
    
    model = joblib.load(model_path)
    print(f"[OK] Da tai model tu {model_path}")
    return model

# =============================================================================
# TIEN XU LY
# =============================================================================
def preprocess(data: pd.DataFrame):
    """
    Giu lai cac cot can thiet cho monitoring.
    """    
    required_cols = FEATURE_COLS + [
        TARGET_COL,
        PRED_COL,
    ]

    data = data[required_cols].dropna().copy()
    
    """
    Convert mot bo du lieu sang Evidently Dataset.
    """
    data_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target=TARGET_COL,
                prediction_labels=PRED_COL,
            )
        ]
    )

    data_evidently = Dataset.from_pandas(
        data,
        data_definition=data_definition
    )

    return data_evidently


# =============================================================================
# TAO & LUU REPORT
# Evidently 0.7+: report.run() tra ve result object, goi save_html tren do
# =============================================================================
def generate_comp_report(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    ref_evidently = preprocess(reference_data)
    cur_evidently = preprocess(current_data)

    report = Report([DataDriftPreset(), ClassificationPreset()])
    result = report.run(reference_data=ref_evidently, current_data=cur_evidently)
    return result

def generate_single_report(data):
    report = Report([ClassificationPreset()])
    result = report.run(current_data=data)
    return result

def save_report(result, save_path: Path) -> None:
    os.makedirs(save_path.parent, exist_ok=True)
    result.save_html(str(save_path))
    print(f"[OK] Report da duoc luu tai: {save_path}")


# =============================================================================
# TRICH XUAT DRIFT SHARE (tuong thich nhieu phien ban Evidently)
# =============================================================================
def _count_drift_from_columns(metric_block: dict) -> float:
    for key in ("value", "result"):
        cols = metric_block.get(key, {}).get("drift_by_columns", {})
        if cols:
            drifted = sum(1 for v in cols.values() if v.get("drift_detected", False))
            return drifted / len(cols) if cols else 0.0
    return 0.0

def _extract_drift_share(result) -> float:
    try:
        raw = json.loads(result.json())
    except Exception as e:
        print(f"[WARN] Khong the parse result.json(): {e}")
        return 0.0

    metrics = raw.get("metrics", [])

    candidate_paths = [
        lambda m: m["value"]["share_of_drifted_columns"],
        lambda m: m["value"]["share"],
        lambda m: m["result"]["share_of_drifted_columns"],
        lambda m: m["result"]["share"],
        lambda m: _count_drift_from_columns(m),
    ]

    for metric_block in metrics:
        for path_fn in candidate_paths:
            try:
                val = path_fn(metric_block)
                if val is not None:
                    return float(val)
            except (KeyError, TypeError):
                continue

    print("\n[DEBUG] Khong tim thay drift share. Cau truc metrics[0]:")
    print(json.dumps(metrics[0] if metrics else {}, indent=2, default=str)[:2000])
    return 0.0

# =============================================================================
# MOT SO HAM TRUOC KHI RETRAIN
# =============================================================================

def _backup_model() -> None:
    """Luu ban sao model cu truoc khi ghi de."""
    if not MODEL_PATH.exists():
        return
    os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = MODEL_BACKUP_DIR / f"model_backup_{ts}.pkl"
    shutil.copy2(MODEL_PATH, backup_path)
    print(f"  [OK] Da backup model cu tai: {backup_path}")

def calc_tra_hang_thang(df: pd.DataFrame) -> pd.Series:
    so_tien_vay = df["so_tien_vay"]
    thoi_han_vay = (df["thoi_han_vay"].replace(0, pd.NA))
    diem_tin_dung = df["diem_tin_dung"]

    lai_suat_nam = (0.08 + (1 - (diem_tin_dung / 850)) * 0.15)
    lai_suat_thang = lai_suat_nam / 12

    tks = (1 + lai_suat_thang) ** thoi_han_vay
    tra_hang_thang = (so_tien_vay * lai_suat_thang * tks / (tks - 1))

    return tra_hang_thang

def merge_datasets_and_save(
    reference_data: pd.DataFrame,
    inference_logs: pd.DataFrame,
) -> pd.DataFrame:
    print("\n  [1/4] Dang gop du lieu train + inference logs...")

    required_cols = FEATURE_COLS + [TARGET_COL]

    # Dùng bản copy để KHÔNG sửa DataFrame gốc
    ref_work = reference_data.copy()
    log_work = inference_logs.copy()

    if "tra_hang_thang" not in ref_work.columns:
        ref_work["tra_hang_thang"] = calc_tra_hang_thang(ref_work)
    if "tra_hang_thang" not in log_work.columns:
        log_work["tra_hang_thang"] = calc_tra_hang_thang(log_work)

    ref_df = ref_work[required_cols].dropna().copy()
    log_df = log_work[required_cols].dropna().copy()

    merged_df = pd.concat([ref_df, log_df], ignore_index=True)

    os.makedirs(TRAIN_PATH.parent, exist_ok=True)
    merged_df.to_csv(TRAIN_PATH, index=False)

    print(f"[OK] Merge thanh cong:")
    print(f"     - Reference : {len(ref_df):,} dong")
    print(f"     - Logs      : {len(log_df):,} dong")
    print(f"     - Total     : {len(merged_df):,} dong -> Da ghi de vao {TRAIN_PATH}")

    return merged_df

# =============================================================================
# RETRAIN
# =============================================================================
def _ask_user_retrain() -> bool:
    """Hoi y kien nguoi dung"""
    try:
        webbrowser.open(f"file://{REPORT_PATH.absolute()}")
        respone = input("\n[?] Phát hiện Data Drift vượt ngưỡng! Bạn có muốn thực hiện Retrain model không? (y/n): ")
        return respone.strip().lower() in ['y', 'yes']
    except Exception:
        return False

def _run_train_script() -> bool:
    """
    Chay lai train.py voi retrain dataset.
    """
    print("\n  [2/4] Dang retrain model...")
    if not TRAIN_SCRIPT.exists():
        print(
            f"  [ERROR] Khong tim thay train script: "
            f"{TRAIN_SCRIPT}"
        )
        return False

    env = os.environ.copy()
    env["TRAIN_DATA_PATH"] = str(TRAIN_PATH)

    try:
        proc = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            env=env,
            capture_output=False,
            text=True,
        )
        if proc.returncode != 0:
            print(
                f"\n  [ERROR] train.py ket thuc voi ma loi: "
                f"{proc.returncode}"
            )
            return False
        print("\n  [OK] Retrain thanh cong.")
        return True

    except Exception as e:
        print(
            f"\n  [ERROR] Loi khi chay train.py: {e}"
        )
        return False

#===================================================================

def proactive_retrain(reference_data: pd.DataFrame,
                      current_data: pd.DataFrame,
                      old_model) -> None:

    print(SEP)
    print("  BAT DAU QUA TRINH RETRAIN")
    print(SEP)

    # Bước 1: Merge và ghi đè TRAIN_PATH
    merged_data = merge_datasets_and_save(reference_data, current_data)

    # Bước 2: Backup model cũ
    _backup_model()

    # Bước 3: Train model mới
    success = _run_train_script()
    if not success:
        print(f"\n  [THẤT BẠI] Quá trình chạy train.py gặp lỗi.")
        print(f"  [INFO] Model cũ vẫn được giữ nguyên (xem backup tại {MODEL_BACKUP_DIR})")
        return
    print(f"\n  [OK] train.py chạy thành công. Model mới đã được lưu tại: {MODEL_PATH}")

    # Bước 4: After report — vẫn dùng current_data để so sánh công bằng với before
    new_model = load_model(MODEL_PATH)
    merged_data[PRED_COL] = new_model.predict(merged_data[FEATURE_COLS])

    new_report = generate_comp_report(current_data, merged_data)
    save_report(new_report, REPORT_PATH)
    webbrowser.open(f"file://{REPORT_PATH.absolute()}")

    print(f"\n{SEP}")
    print("  RETRAIN HOAN TAT THANH CONG")
    print(SEP)

# =============================================================================
# PHAN TICH DRIFT — goi retrain neu can
# =============================================================================
def analyze_drift(result, reference_data: pd.DataFrame, current_data: pd.DataFrame, model, drift_threshold: float = 0.3) -> None:
    """
    Nhận tham số trực tiếp từ main để kiểm tra tỷ lệ drift của các feature.
    """
    drift_share = _extract_drift_share(result)

    print(f"\n{'─' * 60}")
    print(f"  KẾT QUẢ PHÂN TÍCH DATA DRIFT")
    print(f"{'─' * 60}")
    print(f"  Tỷ lệ cột bị drift : {drift_share:.2%}")
    print(f"  Ngưỡng cảnh báo    : {drift_threshold:.0%}")

    if drift_share >= drift_threshold:
        if not _ask_user_retrain():
            print("\n  [INFO] Nguoi dung chon khong retrain. Giu nguyen mo hinh hien tai.")
            print(f"  [INFO] Hay theo doi drift va chay lai monitor.py khi can thiet.")
            return
        else:
            proactive_retrain(reference_data, current_data, model)
    else:
        print(f"\n  [OK] Hệ thống ổn định — Không cần thực hiện retrain.")
        print(f"{'─' * 60}")

# =============================================================================
# ENTRY POINT
# =============================================================================
def main() -> None:
    print(SEP)
    print("   BẮT ĐẦU QUA TRÌNH KIỂM TRA DATA DRIFT")
    print(SEP)

    # 1. Đọc dữ liệu train gốc (reference) và log thực tế (current)
    reference_data = load_data(TRAIN_PATH)
    current_data = load_data(LOGS_PATH)

    # 2. Load mô hình phân loại hiện tại đang chạy sản xuất
    model = load_model(MODEL_PATH)

    # 3. CHUẨN HÓA DỮ LIỆU: Đảm bảo có đủ cột tính toán và dự đoán trước khi chạy Evidently
    print("\n[...] Kiem tra va bo sung cac cot tinh toan/du doan con thieu...")
    
    # Tự động tính cột 'tra_hang_thang' nếu file csv chưa có
    if "tra_hang_thang" not in reference_data.columns:
        reference_data["tra_hang_thang"] = calc_tra_hang_thang(reference_data)
    if "tra_hang_thang" not in current_data.columns:
        current_data["tra_hang_thang"] = calc_tra_hang_thang(current_data)

    # Sinh cột 'prediction' dựa trên mô hình hiện tại để làm ClassificationPreset
    reference_data[PRED_COL] = model.predict(reference_data[FEATURE_COLS])
    if PRED_COL not in current_data.columns:
        current_data[PRED_COL] = model.predict(current_data[FEATURE_COLS])

    # 4. Tiền xử lý dữ liệu và mapping cấu trúc cho đối tượng Evidently Dataset
    print("\n[...] Tien xu ly du lieu cho doi tuong Evidently Dataset...")
 

    # 5. Tính toán và kết xuất Data Drift so sánh tổng quan giữa Train và Logs
    result = generate_comp_report(reference_data, current_data)
    save_report(result, REPORT_PATH)

    # 6. Phân tích kết quả drift, nếu vượt ngưỡng sẽ kích hoạt quy trình xem xet retrain
    analyze_drift(result, reference_data, current_data, model=model, drift_threshold=0.3)

    print(f"\n{SEP}")
    print("   HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH KIỂM TRA MÔ HÌNH")
    print(SEP)

if __name__ == "__main__":
    main()