import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "thu_nhap",
    "so_tien_vay",
    "thoi_han_vay",
    "diem_tin_dung",
    "tra_hang_thang",
]
TARGET = "lich_su_no_xau"
RANDOM_STATE = 42


def build_candidates():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = [
        (
            "logistic_regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            solver="lbfgs",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 3.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        (
            "random_forest",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [8, 12, None],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced", "balanced_subsample"],
            },
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [None, 6, 10],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [20, 40],
            },
        ),
    ]

    return cv, candidates


def train_model(data: pd.DataFrame, model_output_path: Path):
    data = data.dropna(subset=FEATURES + [TARGET]).copy()
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv, candidates = build_candidates()
    best_result = None

    for model_name, estimator, param_grid in candidates:
        print(f"[INFO] Dang toi uu {model_name} theo accuracy...")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        y_pred = search.best_estimator_.predict(X_test)
        result = {
            "name": model_name,
            "search": search,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "report": classification_report(y_test, y_pred, zero_division=0),
        }

        print(
            f"[INFO] {model_name}: "
            f"cv_accuracy={search.best_score_:.4f}, "
            f"test_accuracy={result['accuracy']:.4f}, "
            f"test_f1={result['f1']:.4f}, "
            f"test_recall={result['recall']:.4f}"
        )

        if best_result is None:
            best_result = result
            continue

        if result["accuracy"] > best_result["accuracy"]:
            best_result = result
        elif result["accuracy"] == best_result["accuracy"] and result["f1"] > best_result["f1"]:
            best_result = result

    best_model = best_result["search"].best_estimator_

    os.makedirs(model_output_path.parent, exist_ok=True)
    joblib.dump(best_model, model_output_path)

    print(f"[OK] Da luu model tot nhat tai: {model_output_path}")
    print(f"[OK] Mo hinh duoc chon: {best_result['name']}")
    print(f"[OK] Accuracy tren tap test: {best_result['accuracy'] * 100:.2f}%")
    print(f"[OK] F1 tren tap test: {best_result['f1']:.4f}")
    print(f"[OK] Recall lop no xau: {best_result['recall']:.4f}")
    print("[INFO] Bao cao phan loai:")
    print(best_result["report"])


def main():
    base_dir = Path(__file__).resolve().parent.parent
    env_path = os.environ.get("TRAIN_DATA_PATH")

    train_path = Path(env_path) if env_path else base_dir / "data" / "retrain_data.csv"
    model_output_path = base_dir / "models" / "model.pkl"

    if not train_path.exists():
        print(f"[ERROR] Khong tim thay file du lieu huan luyen tai: {train_path}")
        sys.exit(1)

    print(f"[OK] Dang doc du lieu tai huan luyen tu: {train_path}")
    train_data = pd.read_csv(train_path)

    train_model(train_data, model_output_path)


if __name__ == "__main__":
    main()
