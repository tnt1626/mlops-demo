from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
import csv
from datetime import datetime
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
model = joblib.load("models/model.pkl")

# 3. Định nghĩa dữ liệu đầu vào
class PredictRequest(BaseModel):
    thu_nhap: float = Field(..., ge=0)
    so_tien_vay: float = Field(..., ge=0)
    thoi_han_vay: int = Field(..., ge=6, le=60)    # Kỳ hạn thực tế từ 6 đến 60 tháng
    diem_tin_dung: int = Field(..., ge=300, le=850) # Điểm tín dụng thực tế từ 300 đến 850
    lich_su_no_xau: bool = Field(...)

# 4. Endpoint dự đoán
@app.post("/predict")
def predict(request: PredictRequest):

    # Tính lãi suất năm và lãi suất tháng dựa trên điểm tín dụng của người dùng
    lai_suat_nam = 0.08 + (1 - (request.diem_tin_dung / 850)) * 0.15 
    lai_suat_thang = lai_suat_nam / 12

    # Tính số tiền trả hàng tháng (EMI)
    tks = (1 + lai_suat_thang) ** request.thoi_han_vay
    tra_hang_thang = request.so_tien_vay * lai_suat_thang * tks / (tks - 1)

    # Tạo bảng mới, gom đủ 5 cột
    input_data = pd.DataFrame([{
        'thu_nhap': int(request.thu_nhap),
        'so_tien_vay': int(request.so_tien_vay),
        'thoi_han_vay': int(request.thoi_han_vay),
        'diem_tin_dung': int(request.diem_tin_dung),
        'tra_hang_thang': int(tra_hang_thang)
    }])

    # 5. Tiến hành dự đoán nợ xấu (0: Tốt, 1: Nợ xấu)
    prediction = model.predict(input_data)[0]

    # 6. Ghi log lịch sử dự đoán
    Path("logs").mkdir(exist_ok=True)
    log_file = "logs/inference_logs.csv"

    file_exists = Path(log_file).exists()
    file_empty = not file_exists or Path(log_file).stat().st_size == 0

    with open(log_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if file_empty:
            writer.writerow([
                "timestamp", "thu_nhap", "so_tien_vay", 
                "thoi_han_vay", "diem_tin_dung", "tra_hang_thang", "lich_su_no_xau", "prediction"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            request.thu_nhap, 
            request.so_tien_vay,   
            request.thoi_han_vay, 
            request.diem_tin_dung, 
            int(tra_hang_thang), 
            int(request.lich_su_no_xau),
            int(prediction)
        ])

    # 7. Trả về kết quả
    return {
        "prediction": int(prediction)
    }
    
# Phần này không đụng đến
# =========================
# FRONTEND
# =========================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html"
)