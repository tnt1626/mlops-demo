import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path



def train_model(data: pd.DataFrame):
    
    os.makedirs('models', exist_ok=True)

    features = ['thu_nhap', 'so_tien_vay', 'thoi_han_vay', 'diem_tin_dung', 'tra_hang_thang']
    X = data[features]
    y = data['lich_su_no_xau']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    # Sử dụng class_weight='balanced' vì nợ xấu là nhóm thiểu số
    model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'models/model.pkl')
    print("Đã thành công lưu model tại: models/model.pkl")
    print(f"Accuracy: {model.score(X_test, y_test)*100:.2f}%")
    


def main():
    train_path = Path("data") / "train_data.csv"
    train_data = pd.read_csv(train_path)
    train_model(train_data)
if __name__ == "__main__":
    main()