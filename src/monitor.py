import json
import os
import shutil
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.core.datasets import BinaryClassification
from evidently.presets import ClassificationPreset, DataDriftPreset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# =============================================================================
# DUONG DAN
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train_data.csv"
LOGS_PATH = BASE_DIR / "logs" / "inference_logs.csv"

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
MODEL_BACKUP_DIR = BASE_DIR / "models" / "backups"

REPORT_PATH = BASE_DIR / "reports" / "drift_report.html"
BEFORE_RETRAIN_REPORT = BASE_DIR / "reports" / "before_retrain_report.html"
AFTER_RETRAIN_REPORT = BASE_DIR / "reports" / "after_retrain_report.html"

RETRAIN_SCRIPT = BASE_DIR / "src" / "retrain.py"

FEATURE_COLS = [
    "thu_nhap",
    "so_tien_vay",
    "thoi_han_vay",
    "diem_tin_dung",
    "tra_hang_thang",
]
TARGET_COL = "lich_su_no_xau"
PRED_COL = "prediction"

SEP = "=" * 60


# =============================================================================
# LOAD DU LIEU VA MO HINH
# =============================================================================
def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Khong tim thay du lieu tai: {data_path}")

    data = pd.read_csv(data_path)
    print(f"[OK] Da tai du lieu:\n     - {data_path}")
    return data


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay model tai: {model_path}")

    model = joblib.load(model_path)
    print(f"[OK] Da tai model tu {model_path}")
    return model


# =============================================================================
# TIEN XU LY
# =============================================================================
def calc_tra_hang_thang(df: pd.DataFrame) -> pd.Series:
    so_tien_vay = df["so_tien_vay"]
    thoi_han_vay = df["thoi_han_vay"].replace(0, pd.NA)
    diem_tin_dung = df["diem_tin_dung"]

    lai_suat_nam = 0.08 + (1 - (diem_tin_dung / 850)) * 0.15
    lai_suat_thang = lai_suat_nam / 12

    tks = (1 + lai_suat_thang) ** thoi_han_vay
    return so_tien_vay * lai_suat_thang * tks / (tks - 1)


def prepare_eval_frame(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    if "tra_hang_thang" not in work.columns:
        work["tra_hang_thang"] = calc_tra_hang_thang(work)
    return work


def preprocess_for_report(data: pd.DataFrame):
    required_cols = FEATURE_COLS + [TARGET_COL, PRED_COL]
    report_df = data[required_cols].dropna().copy()

    data_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target=TARGET_COL,
                prediction_labels=PRED_COL,
            )
        ]
    )

    return Dataset.from_pandas(report_df, data_definition=data_definition)


def attach_predictions(data: pd.DataFrame, model) -> pd.DataFrame:
    work = prepare_eval_frame(data)
    work[PRED_COL] = model.predict(work[FEATURE_COLS])
    return work


# =============================================================================
# REPORT
# =============================================================================
def generate_comp_report(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    ref_evidently = preprocess_for_report(reference_data)
    cur_evidently = preprocess_for_report(current_data)

    report = Report([DataDriftPreset(), ClassificationPreset()])
    return report.run(reference_data=ref_evidently, current_data=cur_evidently)


def save_report(result, save_path: Path) -> None:
    os.makedirs(save_path.parent, exist_ok=True)
    result.save_html(str(save_path))
    print(f"[OK] Report da duoc luu tai: {save_path}")


def generate_and_open_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_path: Path,
) -> object:
    result = generate_comp_report(reference_data, current_data)
    save_report(result, report_path)
    webbrowser.open(f"file://{report_path.absolute()}")
    return result


# =============================================================================
# DRIFT SHARE
# =============================================================================
def _count_drift_from_columns(metric_block: dict) -> float:
    for key in ("value", "result"):
        cols = metric_block.get(key, {}).get("drift_by_columns", {})
        if cols:
            drifted = sum(1 for value in cols.values() if value.get("drift_detected", False))
            return drifted / len(cols)
    return 0.0


def _extract_drift_share(result) -> float:
    try:
        raw = json.loads(result.json())
    except Exception as exc:
        print(f"[WARN] Khong the parse result.json(): {exc}")
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
                value = path_fn(metric_block)
                if value is not None:
                    return float(value)
            except (KeyError, TypeError):
                continue

    print("\n[DEBUG] Khong tim thay drift share. Cau truc metrics[0]:")
    print(json.dumps(metrics[0] if metrics else {}, indent=2, default=str)[:2000])
    return 0.0


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_model(model, data: pd.DataFrame) -> dict:
    eval_df = prepare_eval_frame(data)[FEATURE_COLS + [TARGET_COL]].dropna().copy()
    y_true = eval_df[TARGET_COL]
    y_pred = model.predict(eval_df[FEATURE_COLS])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_samples": len(eval_df),
    }


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n  [{label}]")
    print(f"    So mau    : {metrics['n_samples']:,}")
    print(f"    Accuracy  : {metrics['accuracy']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    F1-Score  : {metrics['f1']:.4f}")


# =============================================================================
# RETRAIN
# =============================================================================
def _backup_model() -> Path | None:
    if not MODEL_PATH.exists():
        return None

    os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = MODEL_BACKUP_DIR / f"model_backup_{ts}.pkl"
    shutil.copy2(MODEL_PATH, backup_path)
    print(f"  [OK] Da backup model cu tai: {backup_path}")
    return backup_path


def _restore_model(backup_path: Path | None) -> None:
    if backup_path is None or not backup_path.exists():
        return

    shutil.copy2(backup_path, MODEL_PATH)
    print(f"  [OK] Da phuc hoi model cu tu: {backup_path}")


def _ask_user_retrain() -> bool:
    try:
        response = input("\n[?] Phat hien Data Drift vuot nguong! Ban co muon thuc hien retrain model khong? (y/n): ")
        return response.strip().lower() in ["y", "yes"]
    except Exception:
        return False


def _run_retrain_script(train_data_path: Path) -> bool:
    if not RETRAIN_SCRIPT.exists():
        print(f"  [ERROR] Khong tim thay retrain script: {RETRAIN_SCRIPT}")
        return False

    env = os.environ.copy()
    env["TRAIN_DATA_PATH"] = str(train_data_path)

    try:
        proc = subprocess.run(
            [sys.executable, str(RETRAIN_SCRIPT)],
            env=env,
            capture_output=False,
            text=True,
        )
        if proc.returncode != 0:
            print(f"\n  [ERROR] retrain.py ket thuc voi ma loi: {proc.returncode}")
            return False
        print("\n  [OK] Retrain thanh cong.")
        return True
    except Exception as exc:
        print(f"\n  [ERROR] Loi khi chay retrain.py: {exc}")
        return False


def proactive_retrain(reference_data: pd.DataFrame, current_data: pd.DataFrame, old_model) -> None:
    print(SEP)
    print("  BAT DAU QUA TRINH RETRAIN")
    print(SEP)

    old_metrics = evaluate_model(old_model, current_data)
    print_metrics("Model hien tai tren inference logs", old_metrics)

    backup_path = _backup_model()

    print("\n  [1/2] Dang retrain model bang inference logs...")
    success = _run_retrain_script(LOGS_PATH)
    if not success:
        print("\n  [THAT BAI] Qua trinh chay retrain.py gap loi.")
        print(f"  [INFO] Model cu van duoc giu nguyen (xem backup tai {MODEL_BACKUP_DIR})")
        return

    new_model = load_model(MODEL_PATH)
    new_metrics = evaluate_model(new_model, current_data)
    print_metrics("Model moi tren inference logs", new_metrics)

    deployed_model = new_model
    if (
        new_metrics["accuracy"] < old_metrics["accuracy"]
        or (
            new_metrics["accuracy"] == old_metrics["accuracy"]
            and new_metrics["f1"] < old_metrics["f1"]
        )
    ):
        print("\n  [WARN] Model moi khong tot hon tren inference logs. Phuc hoi model cu.")
        _restore_model(backup_path)
        deployed_model = load_model(MODEL_PATH)
    else:
        print("\n  [OK] Model moi tot hon tren inference logs. Giu model moi.")

    print("\n  [2/2] Tao report sau retrain...")
    reference_after = attach_predictions(reference_data, deployed_model)
    current_after = attach_predictions(current_data, deployed_model)
    generate_and_open_report(reference_after, current_after, AFTER_RETRAIN_REPORT)
    shutil.copy2(AFTER_RETRAIN_REPORT, REPORT_PATH)

    print(f"\n{SEP}")
    print("  RETRAIN HOAN TAT THANH CONG")
    print(SEP)


# =============================================================================
# PHAN TICH DRIFT
# =============================================================================
def analyze_drift(
    result,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    model,
    drift_threshold: float = 0.3,
) -> None:
    drift_share = _extract_drift_share(result)

    print(f"\n{'-' * 60}")
    print("  KET QUA PHAN TICH DATA DRIFT")
    print(f"{'-' * 60}")
    print(f"  Ty le cot bi drift : {drift_share:.2%}")
    print(f"  Nguong canh bao    : {drift_threshold:.0%}")

    if drift_share >= drift_threshold:
        if not _ask_user_retrain():
            print("\n  [INFO] Nguoi dung chon khong retrain. Giu nguyen mo hinh hien tai.")
            print("  [INFO] Hay theo doi drift va chay lai monitor.py khi can thiet.")
            return
        proactive_retrain(reference_data, current_data, model)
    else:
        print("\n  [OK] He thong on dinh - Khong can thuc hien retrain.")
        print(f"{'-' * 60}")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main() -> None:
    print(SEP)
    print("   BAT DAU QUA TRINH KIEM TRA DATA DRIFT")
    print(SEP)

    reference_data = load_data(TRAIN_PATH)
    current_data = load_data(LOGS_PATH)
    model = load_model(MODEL_PATH)

    reference_before = attach_predictions(reference_data, model)
    current_before = attach_predictions(current_data, model)

    result = generate_and_open_report(reference_before, current_before, BEFORE_RETRAIN_REPORT)
    shutil.copy2(BEFORE_RETRAIN_REPORT, REPORT_PATH)

    analyze_drift(
        result,
        reference_data=reference_data,
        current_data=current_data,
        model=model,
        drift_threshold=0.3,
    )


if __name__ == "__main__":
    main()
