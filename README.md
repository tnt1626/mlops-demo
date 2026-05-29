# 🚀 MLOps Demo: Credit Scoring Pipeline
> Dự án Nhập môn Khoa học Dữ liệu - HCMUS (Intro2DS)

## 📋 Mục Lục
- [Giới thiệu dự án](#giới-thiệu-dự-án)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Chuẩn bị môi trường](#chuẩn-bị-môi-trường)
- [Luồng làm việc (Workflow)](#luồng-làm-việc-workflow)
- [Hướng dẫn chạy từng module](#hướng-dẫn-chạy-từng-module)
- [Phân công công việc](#phân-công-công-việc)
- [Quy trình Git & PR](#quy-trình-git--pr)

---

## 🎯 Giới Thiệu Dự Án

### Mục Đích
Đây là một dự án **Lean MLOps** mô phỏng vòng đời hoàn chỉnh của một hệ thống Machine Learning sản xuất:
1. **Huấn luyện mô hình** từ dữ liệu tín dụng giả lập
2. **Triển khai API** để phục vụ dự đoán nợ xấu theo thời gian thực
3. **Giám sát** dữ liệu thực tế (Data Drift Detection)
4. **Retrain tự động** khi phát hiện drift vượt ngưỡng

### Mục Tiêu Học Tập
- Hiểu rõ vòng đời ML từ train → serve → monitor
- Làm quen với công cụ MLOps: **DVC, FastAPI, Evidently AI**
- Thực hành version control, testing, và automation
- Áp dụng kiến trúc **Lean** (gọn nhẹ, chạy trên máy cá nhân)

### Tại sao gọi là "Lean MLOps"?
- ✅ Không dùng Cloud phức tạp (GCP, AWS, etc)
- ✅ Không dùng Kubernetes, Docker swarm
- ✅ Chỉ dùng: **Python + FastAPI + DVC + Evidently**
- ✅ Chạy tốt trên laptop, phù hợp demo & learning

---

## 🏗️ Kiến Trúc Hệ Thống

### Cấu Trúc Thư Mục (Bắt buộc - Không thay đổi!)
```
mlops-demo/
│
├── data/                   # [Quản lý bởi DVC] Dữ liệu thô
│   ├── train_data.csv      # Dữ liệu huấn luyện gốc
│   └── drift_data.csv      # Dữ liệu mô phỏng drift (để test)
│
├── models/                 # [Quản lý bởi DVC] Mô hình đã huấn luyện
│   ├── model.pkl           # Mô hình RandomForest
│   └── backups/            # Các phiên bản cũ
│
├── logs/                   # [Không tracking] Logs từ API
│   └── inference_logs.csv  # Record từng request dự đoán
│
├── reports/                # [Không tracking] Báo cáo từ Evidently
│   └── drift_report.html   # Report HTML hiển thị chi tiết drift
│
├── src/                    # 📌 THƯ MỤC LÀM VIỆC CHÍNH
│   ├── train.py            # Huấn luyện mô hình
│   ├── api.py              # FastAPI server (phục vụ dự đoán)
│   ├── monitor.py          # Phát hiện drift + retrain
│   ├── generate_train_data.py     # Tạo dữ liệu train
│   ├── generate_drift_data.py     # Tạo dữ liệu drift
│   └── simulate_request.py        # Gửi test requests tới API
│
├── templates/              # Giao diện HTML (tùy chọn)
│   └── index.html
│
├── dvc.yaml                # Cấu hình DVC pipeline
├── dvc.lock                # Lock file của DVC
├── Makefile                # Lệnh tắt (make train, make serve, etc)
├── pyproject.toml          # Cấu hình uv
└── README.md               # File này
```

### Quy Trình Dữ Liệu (Data Flow)
```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  [1] GENERATE DATA              [2] TRAIN MODEL               │
│  └─ generate_train_data.py      └─ train.py                   │
│     ↓                              ↓                           │
│  data/train_data.csv ──────────→ models/model.pkl            │
│  (5000 mẫu)                      (RandomForest)              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  [3] SERVE API                                               │
│  └─ api.py + uvicorn                                         │
│     ├─ Load: models/model.pkl                                │
│     ├─ Listen: http://127.0.0.1:8000/predict               │
│     └─ Log:    logs/inference_logs.csv (append)             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
        ↓                                       ↓
   [USER REQUESTS]                   [LOGS ACCUMULATE]
        ↓                                       ↓
   POST /predict ────────────────→ logs/inference_logs.csv
   (từng khách hàng)                 (dự đoán logs)
        ↓
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  [4] MONITOR & DETECT DRIFT                                  │
│  └─ monitor.py (chạy hàng ngày)                              │
│     ├─ Compare: train_data.csv vs inference_logs.csv        │
│     ├─ Tool: Evidently (DataDrift + ClassificationPreset)   │
│     └─ Output: reports/drift_report.html                    │
│                                                               │
│  [5] AUTO RETRAIN (nếu drift > 30%)                         │
│  └─ merge data + retrain (train.py)                          │
│     ├─ Backup mô hình cũ                                    │
│     ├─ Huấn luyện mô hình mới                                │
│     └─ API tự động reload mô hình mới                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Chuẩn Bị Môi Trường

### Yêu Cầu Hệ Thống
- **OS**: Windows / macOS / Linux
- **Python**: 3.13+
- **RAM**: 4GB+ (tối thiểu)

### Bước 1: Cài Đặt `uv` (Package Manager nhanh hơn pip)
```bash
# Cách 1: Dùng pip
pip install uv

# Cách 2: Hoặc download từ https://astral.sh/uv
```

### Bước 2: Clone Dự Án
```bash
git clone <repo-url>
cd mlops-demo
```

### Bước 3: Tạo Môi Trường Ảo & Cài Thư Viện
```bash
# Chạy make (Linux/Mac)
make install

# Hoặc Windows / thủ công:
uv venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate      # Windows

uv pip install -r requirements.txt
```

### Bước 4: Kiểm Tra Cài Đặt
```bash
python --version  # Phải ≥ 3.13
uv --version
source .venv/bin/activate  # Kiểm tra môi trường ảo
```

---

## 📊 Luồng Làm Việc (Workflow)

### Quy Trình Chính (5 Bước)
```
BƯỚC 1: TẠO DỮ LIỆU HUẤN LUYỆN
       ↓
       python src/generate_train_data.py
       Outputs: data/train_data.csv (5000 mẫu)
       
       ↓
BƯỚC 2: HUẤN LUYỆN MÔ HÌNH
       ↓
       python src/train.py
       Outputs: models/model.pkl (RandomForest)
       
       ↓
BƯỚC 3: KHỞI ĐỘNG API SERVER
       ↓
       uv run uvicorn src.api:app --reload
       Listen: http://127.0.0.1:8000
       📖 Docs: http://127.0.0.1:8000/docs (Swagger UI)
       
       ↓ (Mở terminal khác)
BƯỚC 4: GỬI TEST REQUESTS (TẠO LOGS)
       ↓
       python src/generate_drift_data.py      # Tạo drift data
       python src/simulate_request.py         # Gửi 1000 requests
       Outputs: logs/inference_logs.csv (~1000 hàng)
       
       ↓ (Dừng API server, hoặc mở terminal thứ 3)
BƯỚC 5: GIÁM SÁT & PHÁT HIỆN DRIFT
       ↓
       python src/monitor.py
       Outputs: reports/drift_report.html
       Nếu drift > 30%: Hỏi retrain → chạy train.py → cập nhật model
```

### Sơ Đồ Quyết Định Monitoring
```
┌─ START: python src/monitor.py
│
├─ Đọc: train_data.csv (Reference)
├─ Đọc: inference_logs.csv (Current)
├─ Load: model.pkl
│
├─ Chạy Evidently DataDrift
│
├─ Tính: drift_share = % cột bị drift
│
├─ Kiểm tra: drift_share ≥ 30% ?
│
│   ├─ YES → Hỏi: "Retrain không?"
│   │          ├─ User chọn: YES
│   │          │   └─→ Merge data → Backup model → Retrain → Reload
│   │          │
│   │          └─ User chọn: NO
│   │              └─→ Log cảnh báo, thoát
│   │
│   └─ NO → System ổn định, thoát
│
└─ END
```

---

## 🔧 Hướng Dẫn Chạy Từng Module

### 📌 Module 1: Tạo Dữ Liệu Huấn Luyện
**File**: `src/generate_train_data.py`

**Mục đích**: Sinh dữ liệu giả lập gồm 5000 khách hàng với 6 đặc trưng

**Chạy**:
```bash
# Cách 1: Dùng make
make generate-train

# Cách 2: Trực tiếp
python src/generate_train_data.py
```

**Output**:
```
data/train_data.csv
├─ 5000 hàng
├─ Cột: thu_nhap, so_tien_vay, thoi_han_vay, diem_tin_dung, 
│       tra_hang_thang, lich_su_no_xau
└─ Tỷ lệ nợ xấu: ~10%
```

**Ý nghĩa các cột**:
| Cột | Ý nghĩa | Phạm vi |
|-----|---------|--------|
| `thu_nhap` | Thu nhập hàng tháng (VND) | 10-200 triệu |
| `so_tien_vay` | Số tiền vay (VND) | 10 triệu - 3 tỷ |
| `thoi_han_vay` | Thời hạn vay (tháng) | 6, 12, 18, ..., 60 |
| `diem_tin_dung` | Điểm tín dụng | 300-850 |
| `tra_hang_thang` | Tiền trả hàng tháng (VND) | Tính từ lãi suất |
| `lich_su_no_xau` | Nhãn nợ xấu | 0 (tốt), 1 (nợ xấu) |

---

### 📌 Module 2: Huấn Luyện Mô Hình
**File**: `src/train.py`

**Mục đích**: Huấn luyện RandomForest để phân loại nợ xấu

**Chạy**:
```bash
# Cách 1: Dùng make
make train

# Cách 2: Hoặc trực tiếp
python src/train.py
```

**Quá trình**:
1. Đọc `data/train_data.csv`
2. Tách 5 cột features: `[thu_nhap, so_tien_vay, thoi_han_vay, diem_tin_dung, tra_hang_thang]`
3. Chia: 80% train, 20% test
4. Huấn luyện `RandomForestClassifier` (100 cây, max_depth=10, balanced class_weight)
5. Đánh giá trên tập test
6. Lưu mô hình → `models/model.pkl`

**Output**:
```
[OK] Độ chính xác: 85.23%

Chi tiết phân loại:
              precision    recall  f1-score   support
  Khách tốt      0.8741    0.8923    0.8831      456
  Nợ xấu         0.7821    0.7456    0.7634      144
```

---

### 📌 Module 3: Khởi Động API Server
**File**: `src/api.py`

**Mục đích**: Chạy FastAPI server để phục vụ dự đoán nợ xấu theo thời gian thực

**Chạy**:
```bash
# Cách 1: Dùng make
make serve

# Cách 2: Trực tiếp
uv run uvicorn src.api:app --reload
```

**Output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

**Truy cập**:
- 🌐 **Frontend (nếu có)**: http://127.0.0.1:8000/
- 📖 **Swagger UI (Docs)**: http://127.0.0.1:8000/docs
- ❤️ **Health check**: http://127.0.0.1:8000/health

**Endpoint `/predict`**:
```bash
# Ví dụ request
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "thu_nhap": 35000000,
    "so_tien_vay": 200000000,
    "thoi_han_vay": 24,
    "diem_tin_dung": 650,
    "lich_su_no_xau": false
  }'

# Response
{
  "prediction": 0,
  "confidence": 0.89
}
```

**Tính năng**:
- ✅ Tự động tính `tra_hang_thang` (tiền trả hàng tháng)
- ✅ Tự động ghi log vào `logs/inference_logs.csv`
- ✅ Trả về dự đoán + độ tin cậy (confidence score)

---

### 📌 Module 4: Tạo Dữ Liệu Drift (Test Data)
**File**: `src/generate_drift_data.py`

**Mục đích**: Sinh dữ liệu với phân bố **khác** so với tập train (để test monitoring)

**Chạy**:
```bash
python src/generate_drift_data.py
```

**Khác biệt so với train data**:
| Đặc trưng | Train | Drift | Tác động |
|-----------|-------|-------|---------|
| Thu nhập TB | 35 triệu | 25 triệu | ↓ THẤP HƠN |
| Thời hạn vay | Đều | Ưu tiên 18-24 tháng | Phân bố khác |
| Điểm tín dụng | 500 | 480 | ↓ THẤP HƠN |
| Tỷ lệ nợ xấu | ~10% | ~20% | ↑ CAO HƠN |

**Output**:
```
data/drift_data.csv (1000 mẫu)
```

---

### 📌 Module 5: Gửi Test Requests
**File**: `src/simulate_request.py`

**Mục đích**: Mô phỏng người dùng gửi requests → tạo logs

**Yêu cầu**:
- ✅ API phải đang chạy (`make serve` trong terminal khác)
- ✅ Phải có `data/drift_data.csv`

**Chạy**:
```bash
python src/simulate_request.py
```

**Quá trình**:
1. Đọc `data/drift_data.csv` (1000 hàng)
2. Gửi POST request tới API từng hàng
3. Mỗi request được API ghi log vào `logs/inference_logs.csv`
4. In ra status code + dự đoán

**Output**:
```
[   1] ✓ Status 200 | Dự đoán: 0 | Độ tin cậy: 89.12%
[   2] ✓ Status 200 | Dự đoán: 1 | Độ tin cậy: 76.45%
...
logs/inference_logs.csv (~1000 hàng)
```

---

### 📌 Module 6: Phát Hiện Drift & Retrain
**File**: `src/monitor.py`

**Mục đích**: So sánh train data vs logs → phát hiện drift → retrain nếu cần

**Chạy**:
```bash
# Dùng make
make monitor

# Hoặc trực tiếp
python src/monitor.py
```

**Quá trình**:
```
1. Đọc: data/train_data.csv (reference/gốc)
2. Đọc: logs/inference_logs.csv (current/hiện tại)
3. Load: models/model.pkl
4. Dùng Evidently AI để so sánh:
   - DataDriftPreset: Phát hiện thay đổi phân bố
   - ClassificationPreset: Đánh giá hiệu suất
5. Tính: drift_share = % cột bị drift
6. Nếu drift_share ≥ 30%:
   → Hỏi người dùng: "Retrain không?"
   → YES: Merge data + backup + retrain + reload
   → NO: Log cảnh báo, thoát
7. Output: reports/drift_report.html
```

**Output**:
```
70 ─────────────────────────────────────
  KẾT QUẢ PHÂN TÍCH DATA DRIFT
 ─────────────────────────────────────
  📊 Tỷ lệ cột bị drift: 45.00%
  ⚠️  Ngưỡng cảnh báo:   30%
  🚨 PHÁT HIỆN DRIFT VƯỢT NGƯỠNG!
 ─────────────────────────────────────

[?] Data Drift vượt ngưỡng! Bạn có muốn RETRAIN mô hình không? (y/n): y

[OK] Merge dữ liệu:
     - Reference: 5,000 hàng
     - Logs:      1,000 hàng
     - Tổng:      6,000 hàng → Ghi vào data/train_data.csv

[OK] Backup mô hình cũ: models/backups/model_backup_20250529_120000.pkl
[3/5] Huấn luyện RandomForest...
[4/5] Đánh giá mô hình trên tập kiểm tra...
✓ Độ chính xác: 87.45%

[5/5] Lưu mô hình...
✓ Thành công! Mô hình đã được lưu tại: models/model.pkl

[OK] Báo cáo đã được lưu: reports/drift_report.html
✓ RETRAIN HOÀN THÀNH THÀNH CÔNG
```

**Báo cáo HTML**:
- Mở `reports/drift_report.html` trong browser
- Xem chi tiết drift từng cột
- Xem biểu đồ phân bố dữ liệu

---

## 👥 Phân Công Công Việc

### Mô Hình Phân Công (4 Nhóm)

#### 🔴 **Nhóm 1: Kỹ sư Dữ liệu & Huấn luyện**
**Phụ trách**: `src/train.py` + `src/generate_train_data.py` + `dvc.yaml`

**Nhiệm vụ**:
- [x] Sinh dữ liệu huấn luyện (5000 mẫu, 6 cột)
- [x] **[QUAN TRỌNG]** Lưu vào `data/train_data.csv`
- [x] Huấn luyện RandomForest model
- [x] Lưu model → `models/model.pkl`
- [x] Báo cáo: Accuracy, Precision, Recall, F1

**Kiểm tra**:
```bash
make train
# Kiểm tra: Có file models/model.pkl & data/train_data.csv không?
```

**Deliverable**:
- Mô hình đã train
- Dữ liệu huấn luyện sạch
- Documentation về logic nợ xấu

---

#### 🟢 **Nhóm 2-3: Kỹ sư Triển khai API**
**Phụ trách**: `src/api.py`

**Nhiệm vụ**:
- [x] Viết FastAPI endpoint `POST /predict`
- [x] Load model từ `models/model.pkl`
- [x] Nhận 5 input: `thu_nhap, so_tien_vay, thoi_han_vay, diem_tin_dung, lich_su_no_xau`
- [x] **[QUAN TRỌNG]** Ghi log mỗi request vào `logs/inference_logs.csv`
- [x] Trả về: `{prediction: 0/1, confidence: float}`

**Kiểm tra**:
```bash
make serve
# Truy cập: http://127.0.0.1:8000/docs
# Test endpoint /predict
# Kiểm tra: logs/inference_logs.csv có dữ liệu không?
```

**Deliverable**:
- API endpoint working
- Logs được ghi chính xác
- API documentation (Swagger)

---

#### 🔵 **Nhóm 4: Kỹ sư Giám sát & Đánh giá**
**Phụ trách**: `src/monitor.py` + `src/generate_drift_data.py` + `src/simulate_request.py`

**Nhiệm vụ**:
- [x] Đọc: `data/train_data.csv` (Reference)
- [x] Đọc: `logs/inference_logs.csv` (Current)
- [x] Dùng **Evidently DataDriftPreset** để so sánh
- [x] **[QUAN TRỌNG]** Xuất HTML report → `reports/drift_report.html`
- [x] Nếu drift > 30%: Hỏi retrain → chạy `train.py` → cập nhật model

**Kiểm tra**:
```bash
# Tạo drift data + test requests
python src/generate_drift_data.py
make serve &  # Chạy API background
python src/simulate_request.py
# Chạy monitor
make monitor
# Kiểm tra: reports/drift_report.html có không?
```

**Deliverable**:
- Drift detection system working
- HTML reports chi tiết
- Auto-retrain logic

---

### Timeline & Dependencies
```
┌─ Nhóm 1 (Data)
│   └─ [DONE] Tạo train_data.csv
│
├─ Nhóm 2-3 (API)
│   └─ [DEPENDS ON Nhóm 1] model.pkl sẵn sàng
│       └─ [DONE] API endpoint
│           └─ [OUTPUT] logs/inference_logs.csv
│
└─ Nhóm 4 (Monitoring)
    └─ [DEPENDS ON Nhóm 1 + 2] train_data + logs + model sẵn sàng
        └─ [DONE] Monitor + Retrain
```

---

## 📝 Quy Trình Git & PR

### Quy Tắc Bắt Buộc
1. **Không ai được commit** các thư mục:
   - ❌ `data/` (trừ `.gitkeep`)
   - ❌ `models/` (trừ `.gitkeep`)
   - ❌ `logs/` (trừ `.gitkeep`)
   - ❌ `reports/` (trừ `.gitkeep`)
   - ✅ DVC sẽ track các file này

2. **Mỗi người làm một file riêng** (ít conflict)

3. **Test cục bộ trước khi push**

### Quy Trình Chi Tiết

**Bước 1: Tạo branch**
```bash
git checkout -b feature/[tên-module]
# Ví dụ:
# git checkout -b feature/train
# git checkout -b feature/api
# git checkout -b feature/monitor
```

**Bước 2: Code & Test**
```bash
# Chỉnh sửa file của bạn
nano src/train.py

# Test cục bộ
make train  # (hoặc make serve, make monitor)

# Commit từng bước
git add src/train.py
git commit -m "feat: hoàn thiện module train"
```

**Bước 3: Push & Tạo PR**
```bash
git push origin feature/[tên-module]

# Vào GitHub → Tạo Pull Request
# Title: "feat: hoàn thiện [module-name]"
# Description: Mô tả cụ thể công việc
```

**Bước 4: Review & Merge**
```
Leader review code → Approve or Request Changes
→ GitHub auto-merge khi có ≥1 approve
```

### Commit Message Format
```
feat: mô tả công việc          # Tính năng mới
fix: mô tả lỗi sửa             # Sửa lỗi
refactor: mô tả tái cấu trúc   # Cải thiện code
docs: mô tả cập nhật doc       # Cập nhật documentation
test: mô tả test               # Thêm test
```

**Ví dụ**:
```
git commit -m "feat: hoàn thiện train.py với docstring đầy đủ"
git commit -m "fix: xử lý edge case khi chia dữ liệu"
git commit -m "docs: cập nhật README với hướng dẫn chi tiết"
```

---

## 🚀 Quick Start (Tóm Tắt)

```bash
# 1. Setup
make install

# 2. Tạo dữ liệu
python src/generate_train_data.py

# 3. Train
python src/train.py

# 4. Chạy API (terminal 1)
make serve

# 5. Test API (terminal 2)
python src/generate_drift_data.py
python src/simulate_request.py

# 6. Monitor (terminal 2 hoặc 3)
python src/monitor.py

# 7. Xem báo cáo
open reports/drift_report.html  # macOS
# hoặc
xdg-open reports/drift_report.html  # Linux
start reports/drift_report.html     # Windows
```

---

## 📚 Tài Liệu Tham Khảo

- **DVC**: https://dvc.org/doc
- **FastAPI**: https://fastapi.tiangolo.com/
- **Evidently AI**: https://docs.evidentlyai.com/
- **Scikit-learn**: https://scikit-learn.org/
- **Pandas**: https://pandas.pydata.org/

---

## ❓ FAQ & Troubleshooting

### Q1: API không start được
```
Error: Model không found
→ Chạy: python src/train.py trước
```

### Q2: simulate_request.py bị connection error
```
Error: ConnectionRefusedError
→ Mở terminal khác & chạy: make serve
```

### Q3: Monitor.py không tìm thấy logs
```
Error: FileNotFoundError: logs/inference_logs.csv
→ Chạy: python src/simulate_request.py để tạo logs
```

### Q4: DVC pull/push slow
```
→ DVC mặc định dùng local cache
→ Nếu dùng GCS: cần config credentials
→ Xem: dvc.yaml & .dvc/config
```

---

## 👨‍💻 Cách Đóng Góp

1. Fork dự án
2. Tạo branch: `git checkout -b feature/[tên]`
3. Code & test cục bộ
4. Commit: `git commit -m "feat: ..."`
5. Push: `git push origin feature/[tên]`
6. Tạo Pull Request

---

## 📄 License

MIT License - Tự do sử dụng cho mục đích học tập

---

## 👥 Tác Giả

**MLOps Team - HCMUS Intro2DS Project**

Chúc cả team code mượt mà, ghép code không conflict, và demo thành công rực rỡ! 🔥

---

**Cập nhật lần cuối**: 29/05/2025 - v0.2.0 (Refactored)
