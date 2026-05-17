import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib

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
RETRAIN_PATH     = BASE_DIR / "data"    / "retrain_data.csv"
LOGS_PATH        = BASE_DIR / "logs"    / "inference_logs.csv"
MODEL_PATH       = BASE_DIR / "models"  / "model.pkl"
MODEL_BACKUP_DIR = BASE_DIR / "models"  / "backups"
REPORT_PATH      = BASE_DIR / "reports" / "drift_report.html"
TRAIN_SCRIPT     = BASE_DIR / "src"     / "train.py"

FEATURE_COLS = ["thu_nhap", "so_tien_vay", "thoi_han_vay", "diem_tin_dung", "tra_hang_thang"]
TARGET_COL   = "lich_su_no_xau"
PRED_COL     = "prediction"

SEP = "=" * 60


# =============================================================================
# LOAD MODEL
# =============================================================================
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay model tai: {model_path}")
    model = joblib.load(model_path)
    print(f"[OK] Da tai model tu {model_path}")
    return model


# =============================================================================
# LOAD DU LIEU  (tu sinh log mau neu chua co)
# =============================================================================
def _generate_mock_logs(current_path: Path) -> None:
    print(f"[WARN] Khong tim thay file log. Tu dong tao 1000 dong log mau tai: {current_path}")
    os.makedirs(current_path.parent, exist_ok=True)

    np.random.seed(77)
    n = 1000

    thu_nhap = np.random.lognormal(mean=np.log(32_000_000), sigma=0.5, size=n)
    thu_nhap = np.round(thu_nhap.clip(10_000_000, 200_000_000), -5)

    thoi_han_vay = np.random.choice([6, 12, 18, 24, 36, 48, 60], size=n)
    he_so_vay    = np.random.uniform(2, 16, n)
    so_tien_vay  = np.round((thu_nhap * he_so_vay), -6).clip(10_000_000, 3_000_000_000)

    diem_tin_dung = (
        480 + (thu_nhap / 1_000_000) * 0.7 + np.random.normal(0, 50, n)
    ).clip(300, 850).astype(int)

    lai_suat_nam   = 0.08 + (1 - diem_tin_dung / 850) * 0.15
    lai_suat_thang = lai_suat_nam / 12
    tks            = (1 + lai_suat_thang) ** thoi_han_vay
    tra_hang_thang = (so_tien_vay * lai_suat_thang * tks / (tks - 1)).astype(int)

    ty_le_tra_no   = tra_hang_thang / thu_nhap
    diem_rui_ro    = ty_le_tra_no * 4 - diem_tin_dung / 600 + np.random.normal(0, 0.5, n)
    lich_su_no_xau = (diem_rui_ro > np.percentile(diem_rui_ro, 85)).astype(int)

    df = pd.DataFrame({
        "thu_nhap":       thu_nhap.astype(int),
        "so_tien_vay":    so_tien_vay.astype(int),
        "thoi_han_vay":   thoi_han_vay,
        "diem_tin_dung":  diem_tin_dung,
        "tra_hang_thang": tra_hang_thang,
        TARGET_COL:       lich_su_no_xau,
    })

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        df[PRED_COL] = model.predict(df[FEATURE_COLS])
    else:
        df[PRED_COL] = df[TARGET_COL]

    df.to_csv(current_path, index=False)


def load_data(reference_path: Path, current_path: Path):
    if not reference_path.exists():
        raise FileNotFoundError(f"Khong tim thay reference data: {reference_path}")
    if not current_path.exists():
        _generate_mock_logs(current_path)

    reference_data = pd.read_csv(reference_path)
    current_data   = pd.read_csv(current_path)
    print(f"[OK] Da tai du lieu:\n     - {reference_path}\n     - {current_path}")
    return reference_data, current_data


# =============================================================================
# TIEN XU LY
# =============================================================================
def _calc_tra_hang_thang(df: pd.DataFrame) -> pd.Series:
    lai_suat_nam   = 0.08 + (1 - df["diem_tin_dung"] / 850) * 0.15
    lai_suat_thang = lai_suat_nam / 12
    tks            = (1 + lai_suat_thang) ** df["thoi_han_vay"]
    return (df["so_tien_vay"] * lai_suat_thang * tks / (tks - 1)).astype(int)


def preprocess(reference_data: pd.DataFrame, current_data: pd.DataFrame, model):
    ref_df = reference_data.dropna().copy()
    print("[...] Dang tinh bu cot 'tra_hang_thang' cho du lieu train...")
    ref_df["tra_hang_thang"] = _calc_tra_hang_thang(ref_df)
    ref_df = ref_df[FEATURE_COLS + [TARGET_COL]].copy()
    ref_df[PRED_COL] = model.predict(ref_df[FEATURE_COLS])

    cur_df = current_data[FEATURE_COLS + [TARGET_COL, PRED_COL]].dropna().copy()

    data_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target=TARGET_COL,
                prediction_labels=PRED_COL,
            )
        ]
    )

    ref_evidently = Dataset.from_pandas(ref_df, data_definition=data_definition)
    cur_evidently = Dataset.from_pandas(cur_df, data_definition=data_definition)
    return ref_evidently, cur_evidently


# =============================================================================
# TAO & LUU REPORT
# Evidently 0.7+: report.run() tra ve result object, goi save_html tren do
# =============================================================================
def generate_report(reference_data, current_data):
    report = Report([DataDriftPreset(), ClassificationPreset()])
    result = report.run(reference_data=reference_data, current_data=current_data)
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
# DANH GIA MODEL — in ket qua test sau retrain
# =============================================================================
def evaluate_model(model, test_df: pd.DataFrame) -> dict:
    """Chay model tren tap test, tra ve dict cac chi so danh gia."""
    X = test_df[FEATURE_COLS]
    y_true = test_df[TARGET_COL]
    y_pred = model.predict(X)

    # predict_proba de tinh ROC-AUC (neu model ho tro)
    try:
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    except AttributeError:
        auc = None

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   auc,
        "confusion_matrix": cm,
        "n_samples": len(y_true),
    }


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"\n  [{label}]")
    print(f"    Accuracy  : {metrics['accuracy']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    F1-Score  : {metrics['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"    ROC-AUC   : {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"      FN={cm[1,0]}  TP={cm[1,1]}")


# =============================================================================
# RETRAIN — ham chinh
# =============================================================================
def _ask_user_retrain() -> bool:
    """Hoi nguoi dung co muon retrain khong. Tra ve True neu dong y."""
    print(f"\n{'─' * 60}")
    print("  CANH BAO: Data drift vuot nguong cho phep!")
    print("  Mo hinh hien tai co the khong con chinh xac.")
    print(f"{'─' * 60}")
    print("  Ban co muon thuc hien RETRAIN mo hinh khong?")
    print("    [1] Co  — Tien hanh retrain voi du lieu moi")
    print("    [2] Khong — Giu nguyen mo hinh hien tai, dung lai")
    print(f"{'─' * 60}")

    while True:
        choice = input("  Lua chon cua ban (1/2): ").strip()
        if choice == "1":
            return True
        elif choice == "2":
            return False
        else:
            print("  [!] Vui long nhap 1 hoac 2.")


def _backup_model() -> None:
    """Luu ban sao model cu truoc khi ghi de."""
    if not MODEL_PATH.exists():
        return
    os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = MODEL_BACKUP_DIR / f"model_backup_{ts}.pkl"
    import shutil
    shutil.copy2(MODEL_PATH, backup_path)
    print(f"  [OK] Da backup model cu tai: {backup_path}")


def _merge_datasets() -> tuple:
    """
    Gop train_data va inference_logs thanh retrain_data.
    - Chi lay features + target, bo PRED_COL.
    - KHONG drop_duplicates: giu nguyen ca 2 nguon, chi loai NaN.
      drop_duplicates co the xoa sach 1 nguon khi 2 tap co phan phoi
      giong nhau (nhu khi log duoc sinh ra tu cung seed voi train).
    - Tra ve (train_clean, log_clean) lam tap test doc lap.
    """
    print("\n  [1/4] Dang gop du lieu train + inference logs...")
 
    train_df = pd.read_csv(TRAIN_PATH)
    log_df   = pd.read_csv(LOGS_PATH)
 
    # Chuan hoa: dam bao log co cot tra_hang_thang
    if "tra_hang_thang" not in log_df.columns:
        log_df["tra_hang_thang"] = _calc_tra_hang_thang(log_df)
 
    keep_cols   = FEATURE_COLS + [TARGET_COL]
    train_clean = train_df[[c for c in keep_cols if c in train_df.columns]].dropna()
    log_clean   = log_df[[c for c in keep_cols if c in log_df.columns]].dropna()
 
    # Gop don gian: concat + reset index --- KHONG drop_duplicates
    merged = pd.concat([train_clean, log_clean], ignore_index=True)
 
    os.makedirs(RETRAIN_PATH.parent, exist_ok=True)
    merged.to_csv(RETRAIN_PATH, index=False)
 
    print(f"  [OK] retrain_data.csv da tao xong:")
    print(f"       - Train goc : {len(train_clean):,} dong")
    print(f"       - Inference : {len(log_clean):,} dong")
    print(f"       - Sau gop   : {len(merged):,} dong  →  {RETRAIN_PATH}")
 
    # Tra ve rieng biet de dung lam test set doc lap
    return train_clean, log_clean


def _run_train_script() -> bool:
    """
    Goi train.py qua subprocess, truyen RETRAIN_DATA_PATH qua bien moi truong.
    Tra ve True neu thanh cong.
    """
    print("\n  [2/4] Dang chay lai train.py voi retrain_data...")

    if not TRAIN_SCRIPT.exists():
        print(f"  [ERROR] Khong tim thay train script: {TRAIN_SCRIPT}")
        return False

    env = os.environ.copy()
    env["RETRAIN_DATA_PATH"] = str(RETRAIN_PATH)   # train.py co the doc bien nay

    try:
        proc = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            env=env,
            capture_output=False,   # cho phep output hien thi truc tiep
            text=True,
        )
        if proc.returncode != 0:
            print(f"\n  [ERROR] train.py ket thuc voi ma loi: {proc.returncode}")
            return False
        return True
    except Exception as e:
        print(f"  [ERROR] Loi khi chay train.py: {e}")
        return False


def _run_evaluation(train_df: pd.DataFrame, log_df: pd.DataFrame) -> None:
    """
    Load model moi vua train xong, chay test tren 2 tap RIENG BIET:
      - train_df : du lieu goc truoc drift (baseline)
      - log_df   : du lieu production da bi drift (muc tieu chinh)
    KHONG test tren retrain_data vi do la train set (tranh data leakage).
    """
    if 'tra_hang_thang' not in train_df.columns:
        # Tính toán lại lãi suất và số tiền trả hàng tháng (EMI) giống bên train.py
        lai_suat_nam = 0.08 + (1 - (train_df['diem_tin_dung'] / 850)) * 0.15
        lai_suat_thang = lai_suat_nam / 12
        tks = (1 + lai_suat_thang) ** train_df['thoi_han_vay']
        
        train_df['tra_hang_thang'] = (train_df['so_tien_vay'] * lai_suat_thang * tks / (tks - 1)).astype(int)
    
    print("\n  [3/4] Dang danh gia model moi...")
 
    if not MODEL_PATH.exists():
        print("  [ERROR] Khong tim thay model sau khi retrain.")
        return
 
    new_model = load_model(MODEL_PATH)
 
    # Tap 1: train_data goc (baseline truoc drift)
    train_test = train_df[FEATURE_COLS + [TARGET_COL]].dropna()
 
    # Tap 2: inference_logs (du lieu production bi drift)
    log_test = log_df[FEATURE_COLS + [TARGET_COL]].dropna()
 
    metrics_train = evaluate_model(new_model, train_test)
    metrics_log   = evaluate_model(new_model, log_test)
 
    print(f"\n  [4/4] KET QUA DANH GIA MODEL MOI")
    print(f"  {'─' * 50}")
    _print_metrics(f"Tren train_data goc  ({metrics_train['n_samples']:,} mau / baseline)", metrics_train)
    _print_metrics(f"Tren inference_logs  ({metrics_log['n_samples']:,} mau / sau drift) ", metrics_log)
 
    # So sanh delta F1
    delta_f1 = metrics_log["f1"] - metrics_train["f1"]
    symbol   = "+" if delta_f1 >= 0 else ""
    verdict  = "on dinh" if abs(delta_f1) < 0.05 else "can theo doi them"
    print(f"\n  [DELTA] F1 (logs vs train): {symbol}{delta_f1:.4f}  [{verdict}]")
 
    print(f"\n  {'─' * 50}")
    print("  [INFO] Mo report drift de xem chi tiet:")
    print(f"         {REPORT_PATH}")
    print(f"  {'─' * 50}")


def proactive_retrain() -> None:
    """
    Quy trinh retrain tuong tac:
      1. Hoi nguoi dung
      2. Gop du lieu
      3. Backup model cu
      4. Chay lai train.py
      5. Danh gia va bao cao ket qua
    """
    # --- Buoc 0: Hoi nguoi dung ---
    if not _ask_user_retrain():
        print("\n  [INFO] Nguoi dung chon khong retrain. Giu nguyen mo hinh hien tai.")
        print(f"  [INFO] Hay theo doi drift va chay lai monitor.py khi can thiet.")
        return
 
    print(f"\n{SEP}")
    print("  BAT DAU QUA TRINH RETRAIN")
    print(SEP)
 
    # --- Buoc 1: Gop du lieu ---
    train_clean, log_clean = _merge_datasets()
 
    # --- Buoc 2: Backup model cu ---
    _backup_model()
 
    # --- Buoc 3: Chay train.py ---
    success = _run_train_script()
 
    if not success:
        print(f"\n  [THAT BAI] Retrain khong thanh cong.")
        print(f"  [INFO] Model cu van duoc giu nguyen (xem backup tai {MODEL_BACKUP_DIR})")
        return
 
    print(f"\n  [OK] train.py chay thanh cong. Model moi da duoc luu tai: {MODEL_PATH}")
 
    # --- Buoc 4: Danh gia model moi (2 tap doc lap, tranh data leakage) ---
    _run_evaluation(train_clean, log_clean)
 
    print(f"\n{SEP}")
    print("  RETRAIN HOAN TAT THANH CONG")
    print(SEP)

# =============================================================================
# PHAN TICH DRIFT — goi retrain neu can
# =============================================================================
def analyze_drift(result, drift_threshold: float = 0.2) -> None:
    drift_share = _extract_drift_share(result)

    print(f"\n{'─' * 60}")
    print(f"  KET QUA PHAN TICH DRIFT")
    print(f"{'─' * 60}")
    print(f"  Ty le cot bi drift : {drift_share:.2%}")
    print(f"  Nguong canh bao    : {drift_threshold:.0%}")

    if drift_share >= drift_threshold:
        proactive_retrain()
    else:
        print(f"\n  [OK] He thong on dinh — khong can retrain.")
        print(f"{'─' * 60}")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main() -> None:
    print(SEP)
    print("   BAT DAU QUA TRINH KIEM TRA DATA DRIFT")
    print(SEP)

    reference_data, current_data = load_data(TRAIN_PATH, LOGS_PATH)
    model                        = load_model(MODEL_PATH)
    ref_evidently, cur_evidently = preprocess(reference_data, current_data, model)
    result                       = generate_report(ref_evidently, cur_evidently)
    save_report(result, REPORT_PATH)
    analyze_drift(result)

    print(f"\n{SEP}")
    print("   HOAN THANH QUA TRINH KIEM TRA")
    print(SEP)


if __name__ == "__main__":
    main()