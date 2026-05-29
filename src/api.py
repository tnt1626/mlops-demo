"""
Mô-đun API Phục vụ (Serving API Module)
=============================================
Mục đích: Triển khai một API FastAPI để phục vụ dự đoán nợ xấu theo thời gian thực.

Luồng hoạt động:
  1. Khởi tạo FastAPI server
  2. Load mô hình đã huấn luyện từ models/model.pkl
  3. Định nghĩa endpoint POST /predict để nhận dữ liệu đầu vào
  4. Tính toán các đặc trưng phụ trợ (lãi suất, tiền trả hàng tháng)
  5. Dự đoán nợ xấu bằng mô hình
  6. Ghi log toàn bộ request và response vào logs/inference_logs.csv
  7. Trả về kết quả dự đoán

Dependencies: fastapi, uvicorn, joblib, pandas, pydantic

Tác giả: MLOps Team - HCMUS Intro2DS Project

Chạy: uv run uvicorn src.api:app --reload
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import csv

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import joblib
import pandas as pd


# ============================================================================
# CẤU HÌNH
# ============================================================================

MODEL_PATH = "models/model.pkl"
LOG_PATH = Path("logs") / "inference_logs.csv"
TEMPLATES_DIR = "templates"

# Các đặc trưng mô hình cần
FEATURE_COLUMNS = [
    "thu_nhap",
    "so_tien_vay",
    "thoi_han_vay",
    "diem_tin_dung",
    "tra_hang_thang"
]

# Cấu hình lãi suất
MIN_INTEREST_RATE = 0.08  # 8% mỗi năm (base rate)
MAX_INTEREST_ADJUSTMENT = 0.15  # Thêm tối đa 15% dựa trên điểm tín dụng
CREDIT_SCORE_MAX = 850

# Cấu hình thời gian Việt Nam
VIETNAM_TZ = "Asia/Ho_Chi_Minh"


# ============================================================================
# KHỞI TẠO
# ============================================================================

app = FastAPI(
    title="MLOps Demo: Credit Scoring API",
    description="API phục vụ dự đoán nợ xấu cho hệ thống tín dụng",
    version="0.1.0"
)

# Load templates (giao diện frontend)
try:
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
except Exception:
    templates = None

# Load mô hình
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Không tìm thấy mô hình tại {MODEL_PATH}. "
        f"Vui lòng chạy: python src/train.py"
    )


# ============================================================================
# ĐỊNH NGHĨA SCHEMA DỮ LIỆU
# ============================================================================

class PredictRequest(BaseModel):
    """Schema cho request dự đoán nợ xấu."""

    thu_nhap: float = Field(
        ...,
        ge=0,
        description="Thu nhập hàng tháng (VND)"
    )
    so_tien_vay: float = Field(
        ...,
        ge=0,
        description="Số tiền vay (VND)"
    )
    thoi_han_vay: int = Field(
        ...,
        ge=6,
        le=60,
        description="Thời hạn vay (tháng, 6-60)"
    )
    diem_tin_dung: int = Field(
        ...,
        ge=300,
        le=850,
        description="Điểm tín dụng (300-850)"
    )
    lich_su_no_xau: bool = Field(
        ...,
        description="Lịch sử nợ xấu trước đó (True/False)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "thu_nhap": 35000000,
                "so_tien_vay": 200000000,
                "thoi_han_vay": 24,
                "diem_tin_dung": 650,
                "lich_su_no_xau": False
            }
        }


class PredictResponse(BaseModel):
    """Schema cho response dự đoán."""

    prediction: int = Field(
        ...,
        description="Kết quả dự đoán (0: khách hàng tốt, 1: nợ xấu)"
    )
    confidence: float = Field(
        ...,
        description="Độ tin cậy của dự đoán (0-1)"
    )


# ============================================================================
# HÀM TRỢ GIÚP
# ============================================================================

def calculate_monthly_payment(
    loan_amount: float,
    loan_term_months: int,
    credit_score: int
) -> float:
    """
    Tính số tiền trả hàng tháng (EMI - Equated Monthly Installment).

    Công thức: P * r * (1+r)^n / ((1+r)^n - 1)
    Trong đó:
      - P: Số tiền vay
      - r: Lãi suất hàng tháng
      - n: Số tháng

    Thông số:
    ----------
    loan_amount : float
        Số tiền vay (VND)
    loan_term_months : int
        Thời hạn vay (tháng)
    credit_score : int
        Điểm tín dụng (300-850)

    Trả về:
    -------
    float
        Số tiền trả hàng tháng (VND)
    """
    # Tính lãi suất năm dựa trên điểm tín dụng
    # Điểm cao -> lãi thấp, điểm thấp -> lãi cao
    annual_rate = MIN_INTEREST_RATE + (
        1 - (credit_score / CREDIT_SCORE_MAX)
    ) * MAX_INTEREST_ADJUSTMENT

    # Chuyển đổi sang lãi suất hàng tháng
    monthly_rate = annual_rate / 12

    # Tính toán EMI
    compounding_factor = (1 + monthly_rate) ** loan_term_months
    monthly_payment = (
        loan_amount * monthly_rate * compounding_factor
        / (compounding_factor - 1)
    )

    return monthly_payment


def log_prediction(
    input_data: dict,
    prediction: int,
    monthly_payment: float
) -> None:
    """
    Ghi log request dự đoán vào file CSV.

    Thông số:
    ----------
    input_data : dict
        Dữ liệu đầu vào từ request
    prediction : int
        Kết quả dự đoán (0 hoặc 1)
    monthly_payment : float
        Số tiền trả hàng tháng được tính toán
    """
    # Tạo thư mục logs nếu chưa tồn tại
    Path("logs").mkdir(exist_ok=True)

    # Kiểm tra xem file đã tồn tại và có dữ liệu chưa
    file_exists = LOG_PATH.exists()
    file_is_empty = not file_exists or LOG_PATH.stat().st_size == 0

    # Lấy thời gian hiện tại (Múi giờ Việt Nam)
    now = datetime.now(ZoneInfo(VIETNAM_TZ))
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Milliseconds

    # Ghi vào file CSV
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Ghi header nếu file mới
        if file_is_empty:
            writer.writerow([
                "timestamp",
                "thu_nhap",
                "so_tien_vay",
                "thoi_han_vay",
                "diem_tin_dung",
                "tra_hang_thang",
                "lich_su_no_xau",
                "prediction"
            ])

        # Ghi dữ liệu
        writer.writerow([
            timestamp,
            int(input_data["thu_nhap"]),
            int(input_data["so_tien_vay"]),
            int(input_data["thoi_han_vay"]),
            int(input_data["diem_tin_dung"]),
            int(monthly_payment),
            int(input_data["lich_su_no_xau"]),
            int(prediction)
        ])


# ============================================================================
# ENDPOINT API
# ============================================================================

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Endpoint dự đoán nợ xấu.

    Nhận dữ liệu khách hàng và trả về dự đoán nợ xấu + độ tin cậy.

    Thông số:
    ----------
    request : PredictRequest
        Dữ liệu khách hàng gồm: thu_nhap, so_tien_vay, thoi_han_vay,
        diem_tin_dung, lich_su_no_xau

    Trả về:
    -------
    PredictResponse
        prediction: 0 (khách hàng tốt) hoặc 1 (nợ xấu)
        confidence: Độ tin cậy dự đoán

    Quá trình:
    ----------
    1. Tính số tiền trả hàng tháng dựa trên lãi suất điểm tín dụng
    2. Tạo DataFrame từ 5 đặc trưng
    3. Dự đoán bằng mô hình
    4. Tính độ tin cậy từ xác suất
    5. Ghi log request + response
    6. Trả về kết quả
    """

    # Tính toán đặc trưng phụ trợ: tiền trả hàng tháng
    monthly_payment = calculate_monthly_payment(
        loan_amount=request.so_tien_vay,
        loan_term_months=request.thoi_han_vay,
        credit_score=request.diem_tin_dung
    )

    # Chuẩn bị dữ liệu đầu vào cho mô hình
    input_df = pd.DataFrame([{
        "thu_nhap": request.thu_nhap,
        "so_tien_vay": request.so_tien_vay,
        "thoi_han_vay": request.thoi_han_vay,
        "diem_tin_dung": request.diem_tin_dung,
        "tra_hang_thang": monthly_payment
    }])

    # Dự đoán
    prediction = int(model.predict(input_df)[0])

    # Tính độ tin cậy (xác suất tối đa từ predict_proba)
    probabilities = model.predict_proba(input_df)[0]
    confidence = float(max(probabilities))

    # Ghi log
    log_prediction(
        input_data={
            "thu_nhap": request.thu_nhap,
            "so_tien_vay": request.so_tien_vay,
            "thoi_han_vay": request.thoi_han_vay,
            "diem_tin_dung": request.diem_tin_dung,
            "lich_su_no_xau": request.lich_su_no_xau,
        },
        prediction=prediction,
        monthly_payment=monthly_payment
    )

    return PredictResponse(
        prediction=prediction,
        confidence=confidence
    )


# ============================================================================
# GIAO DIỆN FRONTEND (Tùy chọn)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> str:
    """Giao diện web (nếu có file template)."""
    if templates:
        return templates.TemplateResponse(
            request=request,
            name="index.html"
        )
    return """
    <html>
        <body style="font-family: Arial; margin: 50px;">
            <h1>🚀 MLOps Demo: Credit Scoring API</h1>
            <p>Đây là API dự đoán nợ xấu cho hệ thống tín dụng.</p>
            <p><a href="/docs">📖 Xem tài liệu API (Swagger)</a></p>
        </body>
    </html>
    """


@app.get("/health")
def health_check() -> dict:
    """Endpoint kiểm tra sức khỏe API."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now(ZoneInfo(VIETNAM_TZ)).isoformat()
    }
