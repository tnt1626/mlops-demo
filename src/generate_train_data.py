import numpy as np
import pandas as pd
import os


def generate_advanced_credit_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)
    
    # Thu nhập (10tr - 200tr)
    mu = np.log(35000000)
    thu_nhap = np.random.lognormal(mean=mu, sigma=0.5, size=n_samples)
    thu_nhap = np.round(thu_nhap.clip(10000000, 200000000), -5)
    
    # Thời hạn vay (6 tháng đến 60 tháng)
    thoi_han_vay = np.random.choice([6, 12, 18, 24, 36, 48, 60], size=n_samples)
    
    # Số tiền vay (Gấp 1 - 15 lần thu nhập tháng)
    he_so_vay = np.random.uniform(1, 15, n_samples)
    so_tien_vay = np.round((thu_nhap * he_so_vay), -6).clip(10000000, 3000000000)

    # Điểm tín dụng (300 - 850)
    thu_nhap_trieu = thu_nhap / 1000000
    diem_tin_dung = 500 + (thu_nhap_trieu * 0.7) + np.random.normal(0, 50, n_samples)
    diem_tin_dung = diem_tin_dung.clip(300, 850).astype(int)

    # TÍNH LÃI SUẤT Logic: Thu nhập cao, điểm cao -> Lãi thấp
    lai_suat_nam = 0.08 + (1 - (diem_tin_dung / 850)) * 0.15 
    lai_suat_thang = lai_suat_nam / 12

    # TÍNH SỐ TIỀN TRẢ HÀNG THÁNG (EMI - Dư nợ giảm dần)
    # Công thức: P * r * (1+r)^n / ((1+r)^n - 1)
    tks = (1 + lai_suat_thang) ** thoi_han_vay
    tra_hang_thang = so_tien_vay * lai_suat_thang * tks / (tks - 1)
    
    # LOGIC NỢ XẤU: Dựa trên tỷ lệ nợ trên thu nhập hàng tháng (DTI)
    # Nếu tiền trả hàng tháng > 50% thu nhập -> Rất dễ nợ xấu
    ty_le_tra_no = tra_hang_thang / thu_nhap
    diem_rui_ro = (ty_le_tra_no * 4) - (diem_tin_dung / 600) + np.random.normal(0, 0.5, n_samples)
    
    nguong_no_xau = np.percentile(diem_rui_ro, 90)
    lich_su_no_xau = (diem_rui_ro > nguong_no_xau).astype(int)

    # Trả về DataFrame
    df = pd.DataFrame({
        'thu_nhap': thu_nhap.astype(int),
        'so_tien_vay': so_tien_vay.astype(int),
        'thoi_han_vay': thoi_han_vay,
        'diem_tin_dung': diem_tin_dung,
        'tra_hang_thang': tra_hang_thang.astype(int), 
        'lich_su_no_xau': lich_su_no_xau
    })
    
    return df


def main():
    os.makedirs('data', exist_ok=True)
    # Lấy toàn bộ data từ generator
    df = generate_advanced_credit_data(n_samples=2000)

    cols_to_save = ['thu_nhap', 'so_tien_vay', 'thoi_han_vay', 'diem_tin_dung', 'tra_hang_thang', 'lich_su_no_xau']
    df[cols_to_save].to_csv('data/train_data.csv', index=False)
    
    print(f"Da luu data vao: data/train_data.csv")
    print(f"Vi du tra hang thang: {df['tra_hang_thang'].iloc[0]:,} VND")
    print(f"Ti le no xau: {df['lich_su_no_xau'].mean()*100:.1f}%")
if __name__ == "__main__":
    main()


