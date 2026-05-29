"""
Mô-đun Tạo Dữ Liệu Huấn Luyện (Training Data Generator)
===========================================================
Mục đích: Tạo dữ liệu giả lập cho huấn luyện mô hình phân loại nợ xấu.

Luồng:
  1. Sinh dữ liệu ngẫu nhiên: thu nhập, thời hạn vay, điểm tín dụng, ...
  2. Tính toán các đặc trưng phụ trợ: lãi suất, tiền trả hàng tháng
  3. Phân loại khách hàng nợ xấu dựa trên tỷ lệ nợ trên thu nhập (DTI)
  4. Lưu dữ liệu vào data/train_data.csv

Dependencies: numpy, pandas

Chạy: python src/generate_train_data.py
"""

import os
import numpy as np
import pandas as pd


# ============================================================================
# CẤU HÌNH
# ============================================================================

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_data.csv")

NUM_SAMPLES = 5000
RANDOM_STATE = 42

# Phạm vi dữ liệu
MIN_INCOME = 10_000_000  # 10 triệu VND
MAX_INCOME = 200_000_000  # 200 triệu VND
INCOME_MODE = 35_000_000  # Mô-đun (median) ~ 35 triệu

MIN_LOAN_TERM = 6  # tháng
MAX_LOAN_TERM = 60  # tháng

MIN_CREDIT_SCORE = 300
MAX_CREDIT_SCORE = 850

# Cấu hình lãi suất
MIN_INTEREST_RATE = 0.08  # 8%
MAX_INTEREST_ADJUSTMENT = 0.15  # Tối đa +15%

# Ngưỡng phân loại nợ xấu (90 percentile)
BAD_DEBT_PERCENTILE = 90


# ============================================================================
# HÀM SINH DỮ LIỆU
# ============================================================================

def generate_advanced_credit_data(
    n_samples: int = NUM_SAMPLES,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Sinh dữ liệu tín dụng giả lập với logic nợ xấu thực tế.

    Thông số:
    ----------
    n_samples : int
        Số mẫu dữ liệu cần sinh (mặc định: 5000)
    random_state : int
        Seed để tái tạo lại dữ liệu giống nhau (mặc định: 42)

    Trả về:
    -------
    pd.DataFrame
        DataFrame với 6 cột: thu_nhap, so_tien_vay, thoi_han_vay,
        diem_tin_dung, tra_hang_thang, lich_su_no_xau

    Chi tiết cột:
    -----------
    - thu_nhap: Thu nhập hàng tháng (VND), phân bố log-normal
    - so_tien_vay: Số tiền vay (VND), tỷ lệ với thu nhập
    - thoi_han_vay: Thời hạn vay (tháng), phân bố rời rạc {6,12,18,24,36,48,60}
    - diem_tin_dung: Điểm tín dụng (300-850), phụ thuộc vào thu nhập
    - tra_hang_thang: Tiền trả hàng tháng (VND), tính theo công thức EMI
    - lich_su_no_xau: Nhãn (0: tốt, 1: nợ xấu), dựa trên tỷ lệ DTI
    """

    # Cố định seed cho tái tạo lại dữ liệu
    np.random.seed(random_state)

    # ─────────────────────────────────────────────────────────────
    # 1. THU NHẬP (Log-normal distribution: tập trung quanh mode)
    # ─────────────────────────────────────────────────────────────
    mu = np.log(INCOME_MODE)
    sigma = 0.5
    thu_nhap = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)
    thu_nhap = np.round(thu_nhap.clip(MIN_INCOME, MAX_INCOME), -5)

    # ─────────────────────────────────────────────────────────────
    # 2. THỜI HẠN VAY (rời rạc)
    # ─────────────────────────────────────────────────────────────
    thoi_han_vay = np.random.choice(
        [6, 12, 18, 24, 36, 48, 60],
        size=n_samples
    )

    # ─────────────────────────────────────────────────────────────
    # 3. SỐ TIỀN VAY (gấp 1-15 lần thu nhập hàng tháng)
    # ─────────────────────────────────────────────────────────────
    he_so_vay = np.random.uniform(1, 15, n_samples)
    so_tien_vay = np.round(
        (thu_nhap * he_so_vay),
        -6
    ).clip(10_000_000, 3_000_000_000)

    # ─────────────────────────────────────────────────────────────
    # 4. ĐIỂM TÍN DỤNG (phụ thuộc vào thu nhập + noise)
    # ─────────────────────────────────────────────────────────────
    thu_nhap_trieu = thu_nhap / 1_000_000
    diem_tin_dung = (
        500
        + (thu_nhap_trieu * 0.7)
        + np.random.normal(0, 50, n_samples)
    )
    diem_tin_dung = diem_tin_dung.clip(
        MIN_CREDIT_SCORE,
        MAX_CREDIT_SCORE
    ).astype(int)

    # ─────────────────────────────────────────────────────────────
    # 5. LÃI SUẤT & TIỀN TRẢ HÀNG THÁNG (EMI)
    # ─────────────────────────────────────────────────────────────
    # Lãi suất năm: 8% + điểm tín dụng thấp = lãi cao
    lai_suat_nam = (
        MIN_INTEREST_RATE
        + (1 - (diem_tin_dung / MAX_CREDIT_SCORE))
        * MAX_INTEREST_ADJUSTMENT
    )
    lai_suat_thang = lai_suat_nam / 12

    # Công thức EMI: P * r * (1+r)^n / ((1+r)^n - 1)
    tks = (1 + lai_suat_thang) ** thoi_han_vay
    tra_hang_thang = (
        so_tien_vay * lai_suat_thang * tks / (tks - 1)
    )

    # ─────────────────────────────────────────────────────────────
    # 6. PHÂN LOẠI NỢ XẤU (dựa trên tỷ lệ DTI)
    # ─────────────────────────────────────────────────────────────
    # DTI (Debt-to-Income Ratio) = tiền trả / thu nhập hàng tháng
    # Nếu DTI cao -> khả năng nợ xấu cao
    ty_le_tra_no = tra_hang_thang / thu_nhap
    diem_rui_ro = (
        (ty_le_tra_no * 4)
        - (diem_tin_dung / 600)
        + np.random.normal(0, 0.5, n_samples)
    )

    # Ngưỡng: 90 percentile của điểm rủi ro
    nguong_no_xau = np.percentile(diem_rui_ro, BAD_DEBT_PERCENTILE)
    lich_su_no_xau = (diem_rui_ro > nguong_no_xau).astype(int)

    # ─────────────────────────────────────────────────────────────
    # 7. GỘP LẠI THÀNH DATAFRAME
    # ─────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "thu_nhap": thu_nhap.astype(int),
        "so_tien_vay": so_tien_vay.astype(int),
        "thoi_han_vay": thoi_han_vay,
        "diem_tin_dung": diem_tin_dung,
        "tra_hang_thang": tra_hang_thang.astype(int),
        "lich_su_no_xau": lich_su_no_xau
    })

    return df


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    """Hàm chính để tạo và lưu dữ liệu huấn luyện."""

    # Tạo thư mục đầu ra
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print(" TẠO DỮ LIỆU HUẤN LUYỆN GIẢI PHÓNG")
    print("=" * 70)
    print(f"[1/3] Sinh dữ liệu {NUM_SAMPLES:,} mẫu...")

    # Sinh dữ liệu
    df = generate_advanced_credit_data(
        n_samples=NUM_SAMPLES,
        random_state=RANDOM_STATE
    )

    print(f"[2/3] Lưu dữ liệu vào: {OUTPUT_FILE}")
    # Lưu vào file CSV
    df.to_csv(OUTPUT_FILE, index=False)

    # Thống kê
    print(f"\n{'─' * 70}")
    print("📊 THỐNG KÊ DỮ LIỆU:")
    print(f"{'─' * 70}")
    print(f"  Tổng mẫu: {len(df):,}")
    print(f"  Thu nhập TB: {df['thu_nhap'].mean():,.0f} VND")
    print(f"  Tiền vay TB: {df['so_tien_vay'].mean():,.0f} VND")
    print(f"  Tiền trả TB: {df['tra_hang_thang'].mean():,.0f} VND/tháng")
    print(f"  Điểm tín dụng TB: {df['diem_tin_dung'].mean():.1f}")
    print(f"  Tỷ lệ nợ xấu: {df['lich_su_no_xau'].mean() * 100:.1f}%")
    print(f"{'─' * 70}\n")

    print(f"[3/3] ✓ Hoàn tất!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
