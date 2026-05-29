"""
Mô-đun Tạo Dữ Liệu Drift (Data Drift Generator)
=================================================
Mục đích: Sinh dữ liệu giả lập với phân bố khác biệt so với tập huấn luyện
để kiểm tra khả năng phát hiện drift của hệ thống monitoring.

Luồng:
  1. Sinh dữ liệu với tập hợp tham số khác (drift scenario)
  2. Ví dụ: Thu nhập thấp hơn, ưu tiên vay ngắn hạn, điểm tín dụng thấp hơn
  3. Lưu vào data/drift_data.csv
  4. Dữ liệu này dùng để test quá trình monitoring bằng Evidently

Dependencies: numpy, pandas, os

Chạy: python src/generate_drift_data.py
"""

import os
import numpy as np
import pandas as pd


# ============================================================================
# CẤU HÌNH
# ============================================================================

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "drift_data.csv")

NUM_SAMPLES = 1000
RANDOM_STATE = 100  # Seed khác để có phân bố khác

# Phạm vi DỮ LIỆU DRIFT (khác so với tập train)
MIN_INCOME = 8_000_000  # Thấp hơn
MAX_INCOME = 200_000_000
INCOME_MODE = 25_000_000  # ~25 triệu (thấp hơn train ~35 triệu)

# Phân bố thời hạn VAY ưu tiên NGẮN HẠN
LOAN_TERMS = [6, 12, 18, 24, 36, 48, 60]
LOAN_TERM_WEIGHTS = [0.05, 0.1, 0.3, 0.15, 0.2, 0.15, 0.05]  # Ưu tiên 18-24 tháng

MIN_CREDIT_SCORE = 300
MAX_CREDIT_SCORE = 850
CREDIT_SCORE_BASE = 480  # Base điểm thấp hơn (train ~500)

# Cấu hình lãi suất
MIN_INTEREST_RATE = 0.08
MAX_INTEREST_ADJUSTMENT = 0.15

# Xác suất nợ xấu cao hơn (logic Sigmoid)
BAD_DEBT_THRESHOLD = 1.5035


# ============================================================================
# HÀM SINH DỮ LIỆU DRIFT
# ============================================================================

def generate_drifted_credit_data(
    n_samples: int = NUM_SAMPLES,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Sinh dữ liệu tín dụng với DRIFT (phân bố khác biệt).

    Khác biệt so với generate_advanced_credit_data:
    ──────────────────────────────────────────────
    • Thu nhập thấp hơn (~25 triệu thay vì 35 triệu)
    • Ưu tiên vay ngắn hạn (16-24 tháng)
    • Điểm tín dụng thấp hơn (~480 thay vì 500)
    • Tỷ lệ nợ xấu cao hơn

    Thông số:
    ----------
    n_samples : int
        Số mẫu dữ liệu (mặc định: 1000)
    random_state : int
        Seed (mặc định: 100, khác với 42 để có drift)

    Trả về:
    -------
    pd.DataFrame
        DataFrame với 6 cột giống như tập train
    """

    np.random.seed(random_state)

    # ─────────────────────────────────────────────────────────────
    # 1. THU NHẬP THẤP HƠN
    # ─────────────────────────────────────────────────────────────
    mu = np.log(INCOME_MODE)
    sigma = 0.4  # Sigma nhỏ hơn = phân bố tập trung hơn
    thu_nhap = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)
    thu_nhap = np.round(thu_nhap.clip(MIN_INCOME, MAX_INCOME), -5)

    # ─────────────────────────────────────────────────────────────
    # 2. THỜI HẠN VAY NGẮN HƠN
    # ─────────────────────────────────────────────────────────────
    thoi_han_vay = np.random.choice(
        LOAN_TERMS,
        size=n_samples,
        p=LOAN_TERM_WEIGHTS
    )

    # ─────────────────────────────────────────────────────────────
    # 3. SỐ TIỀN VAY (tỷ lệ 3-12 lần, cao hơn train 1-15)
    # ─────────────────────────────────────────────────────────────
    he_so_vay = np.random.uniform(3, 12, n_samples)
    so_tien_vay = np.round(
        (thu_nhap * he_so_vay),
        -6
    ).clip(10_000_000, 3_000_000_000)

    # ─────────────────────────────────────────────────────────────
    # 4. ĐIỂM TÍN DỤNG THẤP HƠN
    # ─────────────────────────────────────────────────────────────
    thu_nhap_trieu = thu_nhap / 1_000_000
    diem_tin_dung = (
        CREDIT_SCORE_BASE
        + (thu_nhap_trieu * 0.5)  # Hệ số thấp hơn
        + np.random.normal(0, 60, n_samples)
    )
    diem_tin_dung = diem_tin_dung.clip(
        MIN_CREDIT_SCORE,
        MAX_CREDIT_SCORE
    ).astype(int)

    # ─────────────────────────────────────────────────────────────
    # 5. LÃI SUẤT & TIỀN TRẢ HÀNG THÁNG
    # ─────────────────────────────────────────────────────────────
    lai_suat_nam = (
        MIN_INTEREST_RATE
        + (1 - (diem_tin_dung / MAX_CREDIT_SCORE))
        * MAX_INTEREST_ADJUSTMENT
    )
    lai_suat_thang = lai_suat_nam / 12

    tks = (1 + lai_suat_thang) ** thoi_han_vay
    tra_hang_thang = (
        so_tien_vay * lai_suat_thang * tks / (tks - 1)
    )

    # ─────────────────────────────────────────────────────────────
    # 6. PHÂN LOẠI NỢ XẤU (XÁC SUẤT CAO HƠN)
    # ─────────────────────────────────────────────────────────────
    ty_le_tra_no = tra_hang_thang / thu_nhap
    diem_rui_ro = (
        (ty_le_tra_no * 4)
        - (diem_tin_dung / 600)
        + np.random.normal(0, 0.5, n_samples)
    )

    # Dùng Sigmoid để tính xác suất (thay vì ngưỡng cứng)
    xac_suat_no_xau = 1 / (1 + np.exp(-(diem_rui_ro - BAD_DEBT_THRESHOLD)))
    lich_su_no_xau = np.random.binomial(1, xac_suat_no_xau)

    # ─────────────────────────────────────────────────────────────
    # 7. GỘP LẠI
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
    """Hàm chính để tạo và lưu dữ liệu drift."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print(" TẠO DỮ LIỆU DRIFT ĐỂ KIỂM ĐỊNH MONITORING")
    print("=" * 70)
    print(f"[1/3] Sinh dữ liệu {NUM_SAMPLES:,} mẫu (seed={RANDOM_STATE})...")

    # Sinh dữ liệu drift
    df_drift = generate_drifted_credit_data(
        n_samples=NUM_SAMPLES,
        random_state=RANDOM_STATE
    )

    print(f"[2/3] Lưu dữ liệu vào: {OUTPUT_FILE}")
    # Lưu vào file CSV
    df_drift.to_csv(OUTPUT_FILE, index=False)

    # Thống kê
    print(f"\n{'─' * 70}")
    print("📊 THỐNG KÊ DỮ LIỆU DRIFT:")
    print(f"{'─' * 70}")
    print(f"  Tổng mẫu: {len(df_drift):,}")
    print(f"  Thu nhập TB: {df_drift['thu_nhap'].mean():,.0f} VND (THẤP HƠN)")
    print(f"  Tiền vay TB: {df_drift['so_tien_vay'].mean():,.0f} VND")
    print(f"  Tiền trả TB: {df_drift['tra_hang_thang'].mean():,.0f} VND/tháng")
    print(f"  Điểm tín dụng TB: {df_drift['diem_tin_dung'].mean():.1f} (THẤP HƠN)")
    print(f"  Tỷ lệ nợ xấu: {df_drift['lich_su_no_xau'].mean() * 100:.1f}% (CAO HƠN)")
    print(f"{'─' * 70}\n")

    print(f"[3/3] ✓ Hoàn tất!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
