# 🚀 MLOps Demo: Credit Scoring Pipeline (Lean MLOps)

Chào mừng team đến với dự án demo MLOps! Dự án này mô phỏng một vòng đời hoàn chỉnh của một mô hình Học máy: từ lúc **Huấn luyện**, **Triển khai** (thành API), đến **Giám sát** dữ liệu thực tế và **Bảo trì**.

Để đảm bảo gọn nhẹ và chạy mượt trên máy cá nhân, chúng ta áp dụng kiến trúc **Lean MLOps**:

- **Môi trường:** `uv` (Nhanh hơn pip)
    
- **Pipeline & Versioning:** `DVC`
    
- **Serving API:** `FastAPI`
    
- **Monitoring:** `Evidently AI` (Batch monitoring, xuất ra file HTML tĩnh)
    

## 📂 1. Cấu trúc Dự án (Quy định bắt buộc)

Mọi thành viên **KHÔNG** tự ý thay đổi cấu trúc thư mục này để đảm bảo code của người này gọi được file của người kia.

```
mlops-demo/
│
├── data/                   # [Track bằng DVC] Chứa file train_data.csv (Dữ liệu gốc)
├── models/                 # [Track bằng DVC] Chứa file model.pkl (Sau khi train)
├── logs/                   # [Không Git/DVC] Chứa file inference_logs.csv (Ghi log từ API)
├── reports/                # [Không Git/DVC] Chứa file drift_report.html (Evidently gen ra)
│
├── src/                    # 📌 THƯ MỤC LÀM VIỆC CHÍNH CỦA TEAM
│   ├── train.py            # Code train model (Nhiệm vụ của Hồi 1)
│   ├── api.py              # Code chạy FastAPI (Nhiệm vụ của Hồi 2 & 3)
│   └── monitor.py          # Code gen report bằng Evidently (Nhiệm vụ của Hồi 4)
│
├── dvc.yaml                # Cấu hình pipeline huấn luyện
├── Makefile                # Tập hợp các lệnh chạy tắt (make train, make serve...)
└── requirements.txt        # (Hoặc pyproject.toml) Quản lý thư viện
```

## ⚙️ 2. Hướng dẫn Setup ban đầu (Dành cho mọi thành viên)

Sau khi `git clone` dự án về máy, mỗi người cần làm các bước sau:

**Bước 1: Cài đặt `uv` (Nếu chưa có)**

```
pip install uv
```

**Bước 2: Cài đặt môi trường ảo và thư viện (Chạy lệnh Make)**

```
make install
```

_(Nếu Windows không chạy được `make`, hãy chạy thủ công: `uv venv` -> Kích hoạt môi trường -> `uv pip install fastapi uvicorn scikit-learn pandas evidently joblib dvc`)_

**Bước 3: Kích hoạt môi trường ảo**

- Mac/Linux: `source .venv/bin/activate`
    
- Windows: `.venv\Scripts\activate`
    

## 👨‍💻 3. Phân công Công việc (Chia theo Hồi)

Mỗi người sẽ nhận 1 file trong thư mục `src/` để code. Vui lòng đọc kỹ **Input/Output** để biết code của mình cần nhận cái gì và trả ra cái gì.

### 📍 Dev 1: Kỹ sư Dữ liệu & Huấn luyện (Phụ trách Hồi 1)

- **File làm việc:** `src/train.py` (và `dvc.yaml`)
    
- **Nhiệm vụ:**
    
    1. Dùng `sklearn.datasets.make_classification` sinh ra dữ liệu giả lập (hoặc tự tạo dataframe) gồm 3 cột: `thu_nhap`, `so_tien_vay`, `lich_su_no_xau`.
        
    2. **(QUAN TRỌNG):** Lưu dữ liệu vừa sinh ra vào `data/train_data.csv` (Để Dev 3 lấy làm mốc so sánh).
        
    3. Train một model đơn giản (vd: RandomForest).
        
    4. Lưu mô hình vào `models/model.pkl`.
        
        
- **Lệnh để test cục bộ:** `make train` (Đảm bảo folder `models` và `data` có sinh ra file).
    

### 📍 Dev 2: Kỹ sư Triển khai API (Phụ trách Hồi 2 & 3)

- **File làm việc:** `src/api.py`
    
- **Nhiệm vụ:**
    
    1. Viết code FastAPI. Load model từ đường dẫn tĩnh: `models/model.pkl` (File do Dev 1 tạo ra).
        
    2. Tạo endpoint `POST /predict` nhận 3 tham số: `thu_nhap`, `so_tien_vay`, `lich_su_no_xau`.
        
    3. **(QUAN TRỌNG):** Mỗi lần user gọi API dự đoán, phải mở file `logs/inference_logs.csv` ở chế độ append (`"a"`) và ghi lại vào đó: `thời gian`, `thu_nhap`, `so_tien_vay`, `lich_su_no_xau`, `kết quả dự đoán`. (Nếu file chưa có thì tạo mới kèm header).
        
- **Lệnh để test cục bộ:** `make serve` (Lên `localhost:8000/docs` thử gửi request xem file log trong folder `logs` có được sinh ra không).
    

### 📍 Dev 3: Kỹ sư Giám sát & Đánh giá (Phụ trách Hồi 4)

- **File làm việc:** `src/monitor.py`
    
- **Nhiệm vụ:**
    
    1. Đọc file dữ liệu gốc do Dev 1 tạo ra ở `data/train_data.csv` (Reference Data).
        
    2. Đọc file log người dùng do Dev 2 tạo ra ở `logs/inference_logs.csv` (Current Data).
        
    3. Dùng thư viện `Evidently` (cụ thể là `DataDriftPreset`) để so sánh 2 tập dữ liệu này.
        
    4. **(QUAN TRỌNG):** Xuất báo cáo HTML ra đường dẫn `reports/drift_report.html`.
        
- **Lệnh để test cục bộ:** `make monitor` (Vào thư mục `reports/` mở file HTML lên xem giao diện có đẹp không).
    

## 🤝 4. Quy trình Git & Ghép Code

Vì mỗi người làm một file riêng rẽ nên việc conflict sẽ rất ít xảy ra. Tuy nhiên, hãy tuân thủ luồng này:

1. Chuyển sang nhánh mới: `git checkout -b feature/ten-file-ban-lam` (vd: `feature/api`)
    
2. Code, test trên máy bằng các lệnh `make ...`
    
3. Commit và Push:
    
    ```
    git add src/ten_file_cua_ban.py
    git commit -m "feat: hoan thien module [ten_module]"
    git push origin feature/ten-file-ban-lam
    ```
    
4. Tạo Pull Request trên GitHub để Leader gộp vào nhánh `main`.
    

**Lưu ý:** Không ai được commit các thư mục `models/`, `logs/`, `reports/`, `data/` lên Git. Hãy kiểm tra `.gitignore` cẩn thận!

_Chúc cả team code mượt mà, ghép code một phát ăn luôn và Demo thành công rực rỡ! 🔥_