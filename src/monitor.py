'''
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from evidently import Dataset, DataDefinition, Report
from evidently.presets import (
    DataDriftPreset,
    ClassificationPreset
)

from evidently.core.datasets import BinaryClassification


# =========================================================
# 1. Generate reference dataset
# =========================================================

X, y = make_classification(
    n_samples=2000,
    n_features=5,
    n_informative=3,
    random_state=42
)

reference_data = pd.DataFrame(
    X,
    columns=[
        "age",
        "income",
        "score",
        "debt",
        "balance"
    ]
)

reference_data["target"] = y


# =========================================================
# 2. Train RandomForest model
# =========================================================

X_train = reference_data.drop(columns=["target"])
y_train = reference_data["target"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# =========================================================
# 3. Create drifted current dataset
# =========================================================

current_data = reference_data.copy()

# create obvious drift
current_data["income"] += 5
current_data["score"] *= 1.8
current_data["balance"] += (
    current_data["balance"] * 0.5
)


# =========================================================
# 4. Generate predictions
# =========================================================

reference_X = reference_data.drop(columns=["target"])
current_X = current_data.drop(columns=["target"])

reference_data["prediction"] = model.predict(reference_X)
current_data["prediction"] = model.predict(current_X)


# =========================================================
# 5. Evidently dataset configuration
# =========================================================

data_definition = DataDefinition(
    classification=[
        BinaryClassification(
            target="target",
            prediction_labels="prediction"
        )
    ]
)

reference_dataset = Dataset.from_pandas(
    reference_data,
    data_definition=data_definition
)

current_dataset = Dataset.from_pandas(
    current_data,
    data_definition=data_definition
)


# =========================================================
# 6. Build Evidently report
# =========================================================

report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])

my_eval = report.run(reference_dataset,current_dataset)
print("Tao report thanh cong!!!")


# =========================================================
# 7. Save report
# =========================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 
report_file = BASE_DIR / "reports" / "drift_report.html"

my_eval.save_html(str(report_file))

print(f"✅ Report đã được lưu an toàn tại: {report_file}")
'''

#phan tren la de demo co chay duoc hay khong
#phan duoi la de lien ket voi dev1 va dev2

''''''
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import (
    DataDriftPreset,
    ClassificationPreset
)

from evidently.core.datasets import BinaryClassification
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

#load model tu file train.py
model_path = BASE_DIR / "models" / "model.pkl"
model = joblib.load(model_path)
print("Đã load model thành công!")

#load 2 dataset tu dev1 va dev2
train_path = BASE_DIR / "data" / "train_data.csv"
logs_path = BASE_DIR / "logs" / "inferences_logs.csv"

reference_data = pd.read_csv(train_path)
current_data = pd.read_csv(logs_path)

reference_X = reference_data.drop(columns=["target"])
current_X = current_data.drop(columns=["target"])

reference_data["prediction"] = model.predict(reference_X)
current_data["prediction"] = model.predict(current_X)

#chuan bi data cho Evidently hieu duoc
data_definition = DataDefinition(
    classification=[
        BinaryClassification(
            target="target",
            prediction_labels="prediction"
        )
    ]
)

reference_dataset = Dataset.from_pandas(
    reference_data,
    data_definition=data_definition
)

current_dataset = Dataset.from_pandas(
    current_data,
    data_definition=data_definition
)

#tao va luu report
report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])

my_eval = report.run(reference_dataset,current_dataset)



report_file = BASE_DIR / "reports" / "drift_report.html"


print(f"✅ Report đã được lưu an toàn tại: {report_file}")

