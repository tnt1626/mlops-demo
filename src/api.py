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


# =========================
# Request Schema
# =========================
class PredictRequest(BaseModel):

    thu_nhap: float = Field(..., ge=0) #...: Bắt buộc
                                        #ge: greater than or Equal 

    so_tien_vay: float = Field(..., ge=0)

    lich_su_no_xau: int = Field(..., ge=0)

# =========================
# Predict Endpoint
# =========================
@app.post("/predict")
def predict(request: PredictRequest):

    # Convert thành DataFrame
    input_data = pd.DataFrame([{
        "thu_nhap": request.thu_nhap,
        "so_tien_vay": request.so_tien_vay,
        "lich_su_no_xau": request.lich_su_no_xau
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    # =========================
    # Ghi log inference
    # =========================

    # Tạo folder logs nếu chưa có
    Path("logs").mkdir(exist_ok=True)

    log_file = "logs/inference_logs.csv"

    # Kiểm tra file tồn tại/chưa có dữ liệu
    file_exists = Path(log_file).exists()
    file_empty = (
        not file_exists or #Ngược với file exists
        Path(log_file).stat().st_size == 0  #Hoặc file rỗng
    )

    # Append log
    with open(log_file, mode="a", newline="") as file:

        writer = csv.writer(file)

        # Nếu file rỗng -> ghi header
        if file_empty:
            writer.writerow([
                "timestamp",
                "thu_nhap",
                "so_tien_vay",
                "lich_su_no_xau",
                "prediction"
            ])

        # Ghi dữ liệu
        writer.writerow([
            datetime.now(),
            request.thu_nhap,
            request.so_tien_vay,
            request.lich_su_no_xau,
            int(prediction)
        ])

    # Return response
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