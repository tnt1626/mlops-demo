"""
Mô-đun Huấn luyện Mô hình (Training Module)
"""
def train_model(data: pd.DataFrame) -> None:
    """
    Huấn luyện mô hình RandomForest để phân loại nợ xấu.

    Thông số:
    ----------
    data : pd.DataFrame
        DataFrame chứa dữ liệu huấn luyện với các cột trong FEATURE_COLUMNS
        và TARGET_COLUMN.

    Quá trình:
    ----------
    1. Chuẩn bị thư mục đầu ra (models/)
    2. Tách features (X) và target (y) từ dữ liệu
    3. Chia dữ liệu: 80% huấn luyện, 20% kiểm tra
    4. Khởi tạo RandomForest với:
       - 100 cây (n_estimators=100)
       - Giới hạn độ sâu 10 (max_depth=10)
       - Cân bằng trọng số lớp (class_weight='balanced')
       - Seed cố định (random_state=42)
    5. Huấn luyện mô hình
    6. Đánh giá trên tập kiểm tra
    7. Lưu mô hình vào models/model.pkl

    Đầu ra:
    -------
    - File model.pkl được lưu tại MODEL_OUTPUT_PATH
    - In ra màn hình: accuracy score và classification report
    """

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(MODEL_OUTPUT_PATH.parent, exist_ok=True)

    # Chuẩn bị features (X) và target (y)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        stratify=y  # Đảm bảo phân bố lớp đều trong cả hai tập
    )

    print("\n" + "=" * 70)
    print(" BẮTĐẦU QUÁ TRÌNH HUẤN LUYỆN MÔ HÌNH")
    print("=" * 70)
    print(f"[1/5] Đọc dữ liệu: {len(data)} mẫu")
    print(f"[2/5] Chia dữ liệu: {len(X_train)} train, {len(X_test)} test")

    # Khởi tạo mô hình
    # class_weight='balanced' giúp xử lý trường hợp nợ xấu là nhóm thiểu số
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1  # Sử dụng tất cả CPU cores
    )

    # Huấn luyện mô hình
    print(f"[3/5] Huấn luyện RandomForest...")
    model.fit(X_train, y_train)

    # Đánh giá mô hình
    print(f"[4/5] Đánh giá mô hình trên tập kiểm tra...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'─' * 70}")
    print(f"  ✓ Độ chính xác: {accuracy * 100:.2f}%")
    print(f"{'─' * 70}")
    print("\nChi tiết phân loại:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Khách hàng tốt", "Nợ xấu"],
        digits=4
    ))

    # Lưu mô hình
    print(f"[5/5] Lưu mô hình...")
    joblib.dump(model, str(MODEL_OUTPUT_PATH))
    print(f"\n✓ Thành công! Mô hình đã được lưu tại: {MODEL_OUTPUT_PATH}")
    print("=" * 70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Hàm chính để khởi chạy quá trình huấn luyện.
    """
    try:
        # Đọc dữ liệu huấn luyện
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu: {DATA_PATH}\n"
                f"Vui lòng chạy: python src/generate_train_data.py"
            )

        print(f"Đang đọc dữ liệu từ: {DATA_PATH}")
        train_data = pd.read_csv(DATA_PATH)

        # Kiểm tra các cột cần thiết
        missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(train_data.columns)
        if missing_cols:
            raise ValueError(
                f"Dữ liệu thiếu các cột: {missing_cols}"
            )

        # Huấn luyện mô hình
        train_model(train_data)

    except Exception as e:
        print(f"\n[LỖI] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
