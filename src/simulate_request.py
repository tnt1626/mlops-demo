import pandas as pd
import requests
import time

# Đọc dữ liệu drift
path = "data/drift_data.csv"
df = pd.read_csv(path)

# URL API
API_URL = "http://127.0.0.1:8000/predict"

print("Bắt đầu simulate requests...")

# Gửi từng dòng vào API
for index, row in df.iterrows():

    payload = {
        "thu_nhap": int(row["thu_nhap"]),
        "so_tien_vay": int(row["so_tien_vay"]),
        "thoi_han_vay": int(row["thoi_han_vay"]),
        "diem_tin_dung": int(row["diem_tin_dung"]),
        "tra_hang_thang": int(row["tra_hang_thang"]),
        "lich_su_no_xau": int(row["lich_su_no_xau"])
    }

    try:
        response = requests.post(API_URL, json=payload)

        print(f"[{index + 1}] Status: {response.status_code}")
        print(response.json())

    except Exception as e:
        print(f"Lỗi request: {e}")

    # Delay nhẹ để giống traffic thật
    time.sleep(0.001)

print("\n-------Hoàn tất simulate requests-------\n\n")