import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


FEATURES = [
    "thu_nhap",
    "so_tien_vay",
    "thoi_han_vay",
    "diem_tin_dung",
    "tra_hang_thang",
]
TARGET = "lich_su_no_xau"


def train_model(data: pd.DataFrame, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print(f"Da thanh cong luu model tai: {model_path}")
    print(f"Accuracy: {model.score(X_test, y_test) * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit scoring model.")
    parser.add_argument(
        "--data-path",
        default=os.getenv("TRAIN_DATA_PATH", "data/train_data.csv"),
        help="CSV training data path.",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", "models/model.pkl"),
        help="Output model path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_data = pd.read_csv(Path(args.data_path))
    train_model(train_data, Path(args.model_path))


if __name__ == "__main__":
    main()
