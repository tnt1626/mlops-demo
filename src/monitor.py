import os
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import (
    DataDriftPreset,
    ClassificationPreset
)

from sklearn.ensemble import RandomForestClassifier
from evidently.core.datasets import BinaryClassification
from pathlib import Path
import joblib

#declaring paths
# BASE_DIR = Path(__file__).resolve().parent.parent
train_path = Path("data") / "train_data.csv"
logs_path = Path("logs") / "inference_logs.csv"
model_path = Path("models") / "model.pkl"
report_path = Path("reports") / "drift_report.html"


#defining methods
def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ Không tìm thấy model: {model_path}"
        )
    model = joblib.load(model_path)
    print(f"✅ THÀNH CÔNG: Đã tải model từ {model_path}")
    return model

def load_data(reference_path, current_path):
    if not reference_path.exists():
        raise FileNotFoundError(
            f"❌ Không tìm thấy reference data: {reference_path}"
        )
    
    if not current_path.exists():
        raise FileNotFoundError(
            f"❌ Không tìm thấy inference logs: {current_path}"
        )
    
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)
    print(f"Đã tải dữ liệu thành công từ {reference_path} và {current_path}")

    return reference_data, current_data

def preprocess(reference_data, current_data, model):
    feature_cols = [
        "thu_nhap",
        "so_tien_vay",
        "thoi_han_vay",
        "diem_tin_dung",
        "tra_hang_thang",
        "lich_su_no_xau"
    ]
    # Thêm .copy() và .dropna() để code an toàn tuyệt đối
    ref_df = reference_data[feature_cols].dropna().copy()

    pred_col_name = "ket_qua_du_doan"
    cur_cols = feature_cols + [pred_col_name]
    cur_df = current_data[cur_cols].dropna().copy()

    # Xử lý Prediction cho tập Reference (vì data train chưa có dự đoán)
    # Trích xuất đúng 5 features theo đúng thứ tự mà model yêu cầu để thực hiện tính toán
    model_features = ["thu_nhap", "so_tien_vay", "thoi_han_vay", "diem_tin_dung", "tra_hang_thang"]
    ref_X = ref_df[model_features]
    ref_df[pred_col_name] = model.predict(ref_X)

    # chuẩn bị data cho Evidently hiểu được
    data_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target="lich_su_no_xau",
                prediction_labels=pred_col_name
            )
        ]
    )

    reference_evidently = Dataset.from_pandas(
        ref_df,
        data_definition=data_definition
    )

    current_evidently = Dataset.from_pandas(
        cur_df,
        data_definition=data_definition
    )

    return reference_evidently, current_evidently

def generate_report(reference_data, current_data):
    #tao va luu report
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])
    my_eval = report.run(reference_data,current_data)

    return my_eval

def save_report(report, save_path):
    os.makedirs(save_path, exist_ok= True)
    report.save_html(str(save_path))

    print(f"✅ Report đã được lưu an toàn tại: {save_path}")

def analyze_drift(report, drift_threshold=0.3):
    result = report.dict()
    drift_share = result["metrics"][0]["value"]["share"]

    if drift_share >= drift_threshold:
        print(f"\n🚨 ALERT! Tỷ lệ drift vượt ngưỡng ({drift_share:.2%} >= {drift_threshold:.0%})")
        
        # GỌI HÀM RETRAIN
        #proactive_retrain()
        #ham nay khong the dong nhat voi dev1 nen khong them vao

    else:
        print(f"\n✅ Model đang hoạt động ổn định ({drift_share:.2%}).")     


def main():
    """
    Hàm điều phối chính cho luồng theo dõi (monitoring workflow).
    """
    print("=== BẮT ĐẦU QUÁ TRÌNH KIỂM TRA DRIFT ===")

    # 1. Load data từ dev1 và dev2

    reference_data, current_data = load_data(train_path, logs_path)

    # 2. Load model từ file train.py
    model = load_model(model_path)

    # 3. Chuẩn bị định dạng của data để tạo report
    reference_evidently, current_evidently = preprocess(reference_data, current_data, model)

    # 4. Tạo report
    report = generate_report(reference_evidently, current_evidently)

    # 5. Lưu report
    save_report(report, report_path)

    # 6. Phân tích drift để đưa ra cảnh báo
    analyze_drift(report)

    print("=== HOÀN THÀNH QUÁ TRÌNH KIỂM TRA ===")

# Khối lệnh này đảm bảo hàm main() chỉ chạy KHI VÀ CHỈ KHI file này được chạy trực tiếp
if __name__ == "__main__":
    main()


# Các cột trong file inference_logs.csv
# timestamp,thu_nhap,so_tien_vay,thoi_han_vay,diem_tin_dung,tra_hang_thang,lich_su_no_xau,prediction