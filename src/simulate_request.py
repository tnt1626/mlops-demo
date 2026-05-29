"""
Mô-đun Mô Phỏng Request (Request Simulator)
=============================================
Mục đích: Gửi các request dự đoán đến API để tạo ra dữ liệu logs phục vụ monitoring.

Luồng:
  1. Đọc dữ liệu drift từ data/drift_data.csv
  2. Lặp qua từng hàng dữ liệu
  3. Gửi POST request đến http://127.0.0.1:8000/predict
  4. In ra kết quả (status code, response)
  5. Yêu cầu: API phải đang chạy (uv run uvicorn src.api:app --reload)

Dependencies: pandas, requests

Chạy: python src/simulate_request.py
     (Trước tiên phải chạy: uv run uvicorn src.api:app --reload trong terminal khác)
"""

import time
import pandas as pd
import requests


# ============================================================================
# CẤU HÌNH
# ============================================================================

DRIFT_DATA_PATH = "data/drift_data.csv"
API_URL = "http://127.0.0.1:8000/predict"

# Thời gian chờ giữa các request (giây)
REQUEST_DELAY = 0.001  # 1ms để mô phỏng traffic thực


# ============================================================================
# HÀM GỬII REQUEST
# ============================================================================

def simulate_requests() -> None:
    """
    Gửi các request dự đoán từ dữ liệu drift đến API.

    Quá trình:
    1. Đọc file data/drift_data.csv
    2. Với mỗi hàng dữ liệu:
       - Tạo payload JSON
       - Gửi POST request đến /predict
       - In ra kết quả
       - Chờ thời gian REQUEST_DELAY
    3. Mỗi request tự động được ghi log vào logs/inference_logs.csv

    Yêu cầu:
    -------
    • API phải đang chạy (uvicorn src.api:app --reload)
    • Phải có file data/drift_data.csv
    """

    # Đọc dữ liệu
    try:
        df = pd.read_csv(DRIFT_DATA_PATH)
    except FileNotFoundError:
        print(f"\n[LỖI] Không tìm thấy file: {DRIFT_DATA_PATH}")
        print("Vui lòng chạy: python src/generate_drift_data.py")
        return

    print("\n" + "=" * 70)
    print(" MÔ PHỎNG GỬII REQUEST ĐẾN API")
    print("=" * 70)
    print(f"📤 Sẽ gửi {len(df)} request đến {API_URL}")
    print(f"⏱️  Delay giữa các request: {REQUEST_DELAY*1000:.1f}ms")
    print("=" * 70 + "\n")

    successful = 0
    failed = 0

    # Gửi từng request
    for index, row in df.iterrows():
        # Tạo payload
        payload = {
            "thu_nhap": int(row["thu_nhap"]),
            "so_tien_vay": int(row["so_tien_vay"]),
            "thoi_han_vay": int(row["thoi_han_vay"]),
            "diem_tin_dung": int(row["diem_tin_dung"]),
            "lich_su_no_xau": int(row["lich_su_no_xau"])
        }

        try:
            # Gửi request
            response = requests.post(API_URL, json=payload, timeout=5)

            if response.status_code == 200:
                result = response.json()
                print(f"[{index + 1:4d}] ✓ Status 200 | Dự đoán: {result['prediction']} "
                      f"| Độ tin cậy: {result['confidence']:.2%}")
                successful += 1
            else:
                print(f"[{index + 1:4d}] ✗ Status {response.status_code} | {response.text[:50]}")
                failed += 1

        except requests.exceptions.ConnectionError:
            print(f"[{index + 1:4d}] ✗ Lỗi kết nối (API không chạy?)")
            failed += 1
            break  # Dừng nếu không thể kết nối

        except Exception as e:
            print(f"[{index + 1:4d}] ✗ Lỗi: {type(e).__name__}: {str(e)[:40]}")
            failed += 1

        # Delay
        time.sleep(REQUEST_DELAY)

    # Tóm tắt
    print("\n" + "=" * 70)
    print(" KẾT QUẢ MÔ PHỎNG")
    print("=" * 70)
    print(f"✓ Thành công: {successful:,} request")
    print(f"✗ Thất bại:   {failed:,} request")
    print(f"📊 Tổng cộng:  {len(df):,} request")
    print("=" * 70 + "\n")

    if successful > 0:
        print(f"✓ Dữ liệu đã được ghi log vào: logs/inference_logs.csv")
        print(f"  Bạn có thể chạy: python src/monitor.py để kiểm tra drift")
    else:
        print(f"⚠️  Không có request nào thành công.")
        print(f"   Hãy đảm bảo API đang chạy: uv run uvicorn src.api:app --reload")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    """Hàm chính để khởi chạy mô phỏng."""
    simulate_requests()


if __name__ == "__main__":
    main()
