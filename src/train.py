import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Data đơn giản
data = pd.DataFrame({
    "thu_nhap": [10, 20, 30, 40],
    "so_tien_vay": [5, 25, 10, 50],
    "lich_su_no_xau": [0, 1, 0, 1],
    "label": [1, 0, 1, 0]
})

# X và y
X = data[["thu_nhap", "so_tien_vay", "lich_su_no_xau"]]
y = data["label"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Tạo folder models nếu chưa có
Path("models").mkdir(exist_ok=True)

# Lưu model
joblib.dump(model, "models/model.pkl")

print("Saved model to models/model.pkl")