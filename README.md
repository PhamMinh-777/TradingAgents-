# Hệ thống vn-llm-agents

Hệ thống mã nguồn phục vụ đề tài Nghiên cứu Khoa học Sinh viên:
**"Ứng dụng LLM và hệ thống đa tác tử hỗ trợ ra quyết định tham khảo trong phân tích tài chính và giao dịch chứng khoán"** (Thị trường chứng khoán Việt Nam).

## 1. Cấu trúc thư mục

```
vn-llm-agents/
├── config/             # File cấu hình tập trung config.yaml
├── src/                # Mã nguồn chính
│   ├── data/           # Thu thập và chuẩn hóa dữ liệu (giá, BCTC, sự kiện, tin tức)
│   ├── features/       # Tính toán chỉ số định lượng & sentiment
│   ├── llm/            # Module gọi LLM, caching đĩa, schemas
│   ├── agents/         # Các tác tử (kỹ thuật, cơ bản, tin tức, tổng hợp)
│   ├── regime/         # Nhận diện chế độ thị trường (HMM, rule-based)
│   ├── eventstudy/     # Nghiên cứu phản ứng thị trường quanh sự kiện (Event Study)
│   ├── backtest/       # Engine mô phỏng giao dịch point-in-time
│   ├── experiments/    # Chạy baseline, ablation study
│   ├── reporting/      # Xuất báo cáo, bảng biểu, đồ thị
│   └── utils/          # Tiện ích chung, assert_no_future, logging, seed
├── prompts/            # Lưu trữ system prompts và template tác tử
├── tests/              # Bộ unit tests cho từng module
├── data/               # Dữ liệu (raw, interim, processed) - được gitignore
├── outputs/            # Kết quả (tables, figures, logs) - được gitignore
├── notebooks/          # Notebooks thử nghiệm & setup cho Colab
├── requirements.txt    # Danh sách thư viện phụ thuộc
├── .env.example        # Mẫu khai báo biến môi trường
└── README.md
```

## 2. Cài đặt và Chạy thử

### 2.1 Môi trường cục bộ (Local)
1. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Tạo file `.env` từ `.env.example` và điền key:
   ```bash
   cp .env.example .env
   ```
3. Kiểm tra môi trường:
   ```bash
   python -m src.utils.check_env
   ```
4. Chạy unit tests:
   ```bash
   pytest tests/
   ```

### 2.2 Google Colab
Mở và chạy file [notebooks/00_setup.ipynb](file:///C:/TDTU 2024-2028/Chính thức/Năm 3/NCKH/notebooks/00_setup.ipynb).

## 3. Nguyên tắc vận hành bất biến
- **Point-in-time**: Chỉ sử dụng dữ liệu có sẵn tại thời điểm quyết định $t$.
- **Không nhìn trước tương lai (No Look-ahead)**: Được kiểm tra tự động qua hàm `assert_no_future`.
- **Tách tập thời gian nghiêm ngặt**:
  - Train: 2020-01-01 đến 2024-12-31
  - Validation: 2025-01-01 đến 2025-12-31
  - Test: 2026-01-01 đến 2026-09-30
