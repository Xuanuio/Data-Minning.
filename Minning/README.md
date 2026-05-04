# Dự đoán tuổi thọ (Life Expectancy) với Machine Learning

Dự án này ứng dụng các kỹ thuật Khai phá dữ liệu (Data Mining) và Học máy (Machine Learning) để dự đoán tuổi thọ trung bình của các quốc gia dựa trên các chỉ số về kinh tế, y tế, và môi trường (GDP, dân số, nước sạch, năng lượng, lượng phát thải...).

## Nội dung thư mục
- `Bao_cao.md`: Báo cáo chi tiết của dự án viết theo định dạng học thuật.
- `Model/`: Thư mục chứa toàn bộ mã nguồn xử lý dữ liệu, huấn luyện mô hình và hệ thống giao diện web demo.
  - `Code1.ipynb` / `Code2.ipynb`: Các notebook phân tích và thử nghiệm mô hình.
  - `web_demo/`: Thư mục chứa giao diện web bằng Streamlit và mã nguồn thuật toán cây quyết định (ExtraTrees) được xây dựng tùy chỉnh.

## Cách chạy ứng dụng Web Demo (Streamlit)

1. Mở terminal và trỏ đường dẫn tới thư mục web:
   ```bash
   cd Model/web_demo
   ```

2. Cài đặt các thư viện cần thiết (nếu chưa có):
   ```bash
   pip install -r requirements.txt
   ```

3. (Tùy chọn) Khởi tạo lại mô hình bằng cách huấn luyện:
   ```bash
   python train_model.py
   ```

4. Chạy ứng dụng web Streamlit:
   ```bash
   streamlit run app.py
   ```
   Hoặc chạy trực tiếp qua file bash:
   ```bash
   ./run_demo.bat
   ```

## Mô hình dự đoán
- Mô hình chính được sử dụng là **Extra Trees Regressor** (Extremely Randomized Trees) được cài đặt và tối ưu bằng tay (Custom model).
- Các chỉ số đánh giá của mô hình trên tập Test:
  - **MAE**: ~ 0.464 tuổi
  - **RMSE**: ~ 1.009 tuổi
  - **R²**: ~ 0.986
- Tính năng quan trọng nhất chi phối tuổi thọ (Feature Importance): Sự kết hợp giữa việc tiếp cận Điện và Nước sạch, và tuổi thọ năm trước.
