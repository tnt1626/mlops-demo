import argparse
import csv
import json
from pathlib import Path

import joblib
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.core.datasets import BinaryClassification
from evidently.presets import ClassificationPreset, DataDriftPreset


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_REFERENCE_PATH = BASE_DIR / "data" / "train_data.csv"
DEFAULT_LOGS_PATH = BASE_DIR / "logs" / "inference_logs.csv"
DEFAULT_BASE_MODEL_PATH = BASE_DIR / "models" / "base_model.pkl"
DEFAULT_REPORT_PATH = BASE_DIR / "reports" / "drift_report.html"
DEFAULT_METRICS_PATH = BASE_DIR / "reports" / "drift_metrics.json"
DEFAULT_DECISION_PATH = BASE_DIR / "reports" / "retrain_decision.json"
DEFAULT_RETRAIN_DATA_PATH = BASE_DIR / "data" / "retrain_data.csv"

FEATURE_COLS = [
    "thu_nhap",
    "so_tien_vay",
    "thoi_han_vay",
    "diem_tin_dung",
    "tra_hang_thang",
]
TARGET_COL = "lich_su_no_xau"
PRED_COL = "prediction"
API_LOG_COLS = ["timestamp"] + FEATURE_COLS + [TARGET_COL, PRED_COL]
DRIFT_DATA_COLS = FEATURE_COLS + [TARGET_COL]
DRIFT_DATA_WITH_PRED_COLS = FEATURE_COLS + [TARGET_COL, PRED_COL]


def load_data(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Khong tim thay file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_inference_logs(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(
            f"{path} dang rong hoac chua ton tai. Hay chay API bang `make serve`, "
            "roi chay `make simulate-drift` truoc khi monitor/retrain."
        )

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"{path} khong co dong du lieu nao. Hay chay API bang `make serve`, "
            "roi chay `make simulate-drift` truoc khi monitor/retrain."
        ) from exc
    except pd.errors.ParserError:
        pass

    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None:
            raise ValueError(
                f"{path} khong co header. Hay chay API bang `make serve`, "
                "roi chay `make simulate-drift` truoc khi monitor/retrain."
            )

        for row in reader:
            if not row or row == header:
                continue
            if len(row) == len(DRIFT_DATA_COLS):
                rows.append(dict(zip(DRIFT_DATA_COLS, row)))
            elif len(row) == len(DRIFT_DATA_WITH_PRED_COLS):
                rows.append(dict(zip(DRIFT_DATA_WITH_PRED_COLS, row)))
            elif len(row) == len(API_LOG_COLS):
                rows.append(dict(zip(API_LOG_COLS, row)))
            else:
                raise ValueError(
                    f"Dong log khong dung schema: expected 6, 7, or 8 fields; got {len(row)} fields."
                )

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError(
            f"{path} khong co log hop le. Hay chay API bang `make serve`, "
            "roi chay `make simulate-drift` truoc khi monitor/retrain."
        )
    for col in DRIFT_DATA_WITH_PRED_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay model: {path}")
    return joblib.load(path)


def calc_tra_hang_thang(df: pd.DataFrame) -> pd.Series:
    so_tien_vay = df["so_tien_vay"]
    thoi_han_vay = df["thoi_han_vay"].replace(0, pd.NA)
    diem_tin_dung = df["diem_tin_dung"]

    lai_suat_nam = 0.08 + (1 - (diem_tin_dung / 850)) * 0.15
    lai_suat_thang = lai_suat_nam / 12
    tks = (1 + lai_suat_thang) ** thoi_han_vay
    return so_tien_vay * lai_suat_thang * tks / (tks - 1)


def normalize_data(data: pd.DataFrame, model=None, *, require_prediction: bool = True) -> pd.DataFrame:
    if data.empty:
        return data

    normalized = data.copy()
    if "tra_hang_thang" not in normalized.columns:
        normalized["tra_hang_thang"] = calc_tra_hang_thang(normalized)

    required_cols = FEATURE_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in normalized.columns]
    if missing_cols:
        raise ValueError(f"Thieu cot bat buoc: {missing_cols}")

    normalized = normalized[required_cols + ([PRED_COL] if PRED_COL in normalized.columns else [])]
    normalized = normalized.dropna().copy()

    if model is not None:
        normalized[PRED_COL] = model.predict(normalized[FEATURE_COLS])
    elif require_prediction and PRED_COL not in normalized.columns:
        raise ValueError(f"Thieu cot {PRED_COL} va khong co model de sinh prediction.")

    return normalized


def to_evidently_dataset(data: pd.DataFrame) -> Dataset:
    data_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target=TARGET_COL,
                prediction_labels=PRED_COL,
            )
        ]
    )
    return Dataset.from_pandas(data, data_definition=data_definition)


def generate_report(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    report = Report([DataDriftPreset(), ClassificationPreset()])
    return report.run(
        reference_data=to_evidently_dataset(reference_data),
        current_data=to_evidently_dataset(current_data),
    )


def _count_drift_from_columns(metric_block: dict) -> float | None:
    for key in ("value", "result"):
        cols = metric_block.get(key, {}).get("drift_by_columns", {})
        if cols:
            drifted = sum(1 for value in cols.values() if value.get("drift_detected", False))
            return drifted / len(cols)
    return None


def extract_drift_share(result) -> float:
    try:
        raw = json.loads(result.json())
    except Exception as exc:
        print(f"[WARN] Khong parse duoc Evidently JSON: {exc}")
        return 0.0

    for metric_block in raw.get("metrics", []):
        candidates = [
            metric_block.get("value", {}).get("share_of_drifted_columns"),
            metric_block.get("value", {}).get("share"),
            metric_block.get("result", {}).get("share_of_drifted_columns"),
            metric_block.get("result", {}).get("share"),
            _count_drift_from_columns(metric_block),
        ]
        for value in candidates:
            if value is not None:
                return float(value)
    return 0.0


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_monitor(args: argparse.Namespace) -> None:
    reference_path = Path(args.reference_path)
    logs_path = Path(args.logs_path)
    model_path = Path(args.model_path)
    report_path = Path(args.report_path)
    metrics_path = Path(args.metrics_path)
    decision_path = Path(args.decision_path)

    model = load_model(model_path)
    reference_data = normalize_data(load_data(reference_path), model=model)
    current_data = normalize_data(load_inference_logs(logs_path), model=model)

    result = generate_report(reference_data, current_data)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(report_path))

    drift_share = extract_drift_share(result)
    should_retrain = drift_share >= args.drift_threshold
    payload = {
        "drift_share": drift_share,
        "drift_threshold": args.drift_threshold,
        "should_retrain": should_retrain,
        "reference_rows": int(len(reference_data)),
        "current_rows": int(len(current_data)),
    }
    save_json(metrics_path, payload)
    save_json(decision_path, payload)

    print(f"[OK] Drift report: {report_path}")
    print(f"[OK] Drift metrics: {metrics_path}")
    print(f"[OK] Retrain decision: {decision_path}")
    print(f"[INFO] Drift share: {drift_share:.2%}; should_retrain={should_retrain}")


def prepare_retrain_data(args: argparse.Namespace) -> None:
    reference_path = Path(args.reference_path)
    logs_path = Path(args.logs_path)
    decision_path = Path(args.decision_path)
    output_path = Path(args.output_path)

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    should_retrain = bool(decision.get("should_retrain", False))

    reference_data = normalize_data(load_data(reference_path), require_prediction=False)
    reference_data = reference_data[FEATURE_COLS + [TARGET_COL]]

    if should_retrain:
        logs_data = normalize_data(load_inference_logs(logs_path), require_prediction=False)
        logs_data = logs_data[FEATURE_COLS + [TARGET_COL]]
        retrain_data = pd.concat([reference_data, logs_data], ignore_index=True)
        print(f"[INFO] Drift vuot nguong, merge train + logs: {len(retrain_data)} rows.")
    else:
        retrain_data = reference_data
        print("[INFO] Drift chua vuot nguong, dung lai train data hien tai.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    retrain_data.to_csv(output_path, index=False)
    print(f"[OK] Retrain data: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor drift and prepare DVC retrain inputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    monitor = subparsers.add_parser("run", help="Generate drift report and retrain decision.")
    monitor.add_argument("--reference-path", default=str(DEFAULT_REFERENCE_PATH))
    monitor.add_argument("--logs-path", default=str(DEFAULT_LOGS_PATH))
    monitor.add_argument("--model-path", default=str(DEFAULT_BASE_MODEL_PATH))
    monitor.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    monitor.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    monitor.add_argument("--decision-path", default=str(DEFAULT_DECISION_PATH))
    monitor.add_argument("--drift-threshold", type=float, default=0.3)
    monitor.set_defaults(func=run_monitor)

    prepare = subparsers.add_parser("prepare-retrain-data", help="Create retrain dataset from decision.")
    prepare.add_argument("--reference-path", default=str(DEFAULT_REFERENCE_PATH))
    prepare.add_argument("--logs-path", default=str(DEFAULT_LOGS_PATH))
    prepare.add_argument("--decision-path", default=str(DEFAULT_DECISION_PATH))
    prepare.add_argument("--output-path", default=str(DEFAULT_RETRAIN_DATA_PATH))
    prepare.set_defaults(func=prepare_retrain_data)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
