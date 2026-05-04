# BÁO CÁO BÀI TẬP LỚN
## HỌC PHẦN: KHAI PHÁ DỮ LIỆU (DATA MINING)

**ĐỀ TÀI:**
**DỰ ĐOÁN TUỔI THỌ (LIFE EXPECTANCY) BẰNG CÁC MÔ HÌNH HỌC MÁY (MACHINE LEARNING)**

**Giảng viên hướng dẫn:** [Điền tên Giảng viên]
**Nhóm thực hiện:** [Điền tên/Số Nhóm]
**Lớp:** [Điền tên Lớp]

---

## LỜI CẢM ƠN

Chúng em xin gửi lời cảm ơn chân thành và sâu sắc nhất đến Giảng viên hướng dẫn đã tận tình giảng dạy, truyền đạt những kiến thức quý báu về lĩnh vực Khai phá Dữ liệu (Data Mining) và Học máy (Machine Learning). Những hướng dẫn chi tiết của thầy/cô là tiền đề quan trọng giúp nhóm chúng em hoàn thành tốt đồ án này.

Dù đã cố gắng hết sức trong quá trình nghiên cứu và thực hiện đề tài, tuy nhiên do giới hạn về mặt thời gian cũng như kinh nghiệm thực tiễn, báo cáo khó tránh khỏi những thiếu sót nhất định. Chúng em rất mong nhận được những nhận xét, góp ý từ thầy/cô để đề tài được hoàn thiện hơn.

Xin chân thành cảm ơn!

---

## DANH SÁCH THÀNH VIÊN

| STT | Họ và tên | Mã sinh viên | Phân công nhiệm vụ | Mức độ hoàn thành |
|:---:|:---|:---|:---|:---:|
| 1 | [Tên thành viên 1] | [MSV 1] | Xử lý dữ liệu, Xây dựng model | 100% |
| 2 | [Tên thành viên 2] | [MSV 2] | Phân tích EDA, Feature Engineering | 100% |
| 3 | [Tên thành viên 3] | [MSV 3] | Trực quan hóa, Làm giao diện Web, Viết báo cáo | 100% |

---

## MỤC LỤC

1. [LỜI MỞ ĐẦU](#lời-mở-đầu)
2. [CHƯƠNG 1: CƠ SỞ LÝ THUYẾT](#chương-1-cơ-sở-lý-thuyết)
3. [CHƯƠNG 2: BÀI TOÁN VÀ GIẢI PHÁP](#chương-2-bài-toán-và-giải-pháp)
   - [2.1 Mô tả dữ liệu](#21-mô-tả-dữ-liệu)
   - [2.2 Tiền xử lý dữ liệu](#22-tiền-xử-lý-dữ-liệu)
   - [2.3 Trực quan hóa dữ liệu (EDA)](#23-trực-quan-hóa-dữ-liệu-eda)
   - [2.4 Feature Engineering (Trích xuất đặc trưng)](#24-feature-engineering)
   - [2.5 Xây dựng model](#25-xây-dựng-model)
4. [CHƯƠNG 3: THỰC NGHIỆM](#chương-3-thực-nghiệm)
   - [3.1 Huấn luyện model](#31-huấn-luyện-model)
   - [3.2 Đánh giá model](#32-đánh-giá-model)
   - [3.3 Phân tích Feature Importance](#33-phân-tích-feature-importance)
   - [3.4 Xây dựng giao diện demo](#34-xây-dựng-giao-diện-demo)
5. [KẾT LUẬN](#kết-luận)

---

## LỜI MỞ ĐẦU

Tuổi thọ trung bình (Life Expectancy) là một trong những chỉ số quan trọng nhất để đánh giá mức độ phát triển của một quốc gia, phản ánh tổng hợp chất lượng cuộc sống, hệ thống y tế, tình trạng kinh tế, môi trường và các dịch vụ cơ bản. Trong bối cảnh thế giới đang biến đổi nhanh chóng với những tác động mạnh mẽ từ biến đổi khí hậu, dịch bệnh (như đại dịch COVID-19) và sự chênh lệch giàu nghèo, việc dự báo chính xác tuổi thọ trung bình trở thành một bài toán cấp thiết cho các nhà hoạch định chính sách, các tổ chức y tế và chính phủ.

Đề tài **"Dự đoán tuổi thọ bằng Machine Learning"** được thực hiện nhằm mục đích ứng dụng các thuật toán Khai phá dữ liệu và Học máy để phân tích mối tương quan giữa các yếu tố kinh tế - xã hội (như GDP, dân số, mức độ tiếp cận nước sạch, điện, lượng khí thải CO2...) và tuổi thọ. Qua đó, đề tài xây dựng một mô hình hồi quy (Regression) cho phép dự đoán tuổi thọ của một quốc gia dựa vào các chỉ số kể trên. Ngoài ra, việc giải quyết bài toán này còn giúp chỉ ra những yếu tố (features) nào có tác động lớn nhất đến tuổi thọ, đóng góp tiếng nói vào các quyết định đầu tư cải thiện chất lượng sống.

---

## CHƯƠNG 1: CƠ SỞ LÝ THUYẾT

### 1.1 Machine Learning (Học máy) là gì?
Học máy là một lĩnh vực thuộc Trí tuệ nhân tạo (AI) tập trung vào việc nghiên cứu và xây dựng các thuật toán cho phép máy tính có khả năng tự học hỏi từ dữ liệu để giải quyết các bài toán cụ thể mà không cần lập trình các quy tắc logic một cách tường minh. Trong đề tài này, nhóm sử dụng Học có giám sát (Supervised Learning).

### 1.2 Bài toán Regression (Hồi quy)
Vì biến mục tiêu (target) trong bài toán này là "Tuổi thọ trung bình" - một biến liên tục (ví dụ: 72.5 tuổi, 80.1 tuổi), do đó đây là bài toán Hồi quy. Nhiệm vụ của mô hình là tìm ra một hàm $f(X)$ ánh xạ từ tập các biến đầu vào $X$ (như GDP, dân số,...) để dự đoán đầu ra $Y$ (tuổi thọ) sao cho sai số là nhỏ nhất.

### 1.3 Thuật toán Extra Trees (Extremely Randomized Trees)
Extra Trees là một thuật toán thuộc họ Ensemble Learning, được phát triển dựa trên nền tảng của Random Forest. Tuy nhiên, nó có một số khác biệt quan trọng:
- Thay vì tìm điểm chia (split) tối ưu nhất như Random Forest, Extra Trees chọn ngẫu nhiên một vài điểm chia cho mỗi đặc trưng và chọn điểm tốt nhất trong số đó.
- Extra Trees thường sử dụng toàn bộ tập dữ liệu gốc để xây dựng từng cây (thay vì dùng kỹ thuật Bootstrap như Random Forest), giúp giảm phương sai (variance) của mô hình.
Thuật toán này đặc biệt hiệu quả trong việc chống lại hiện tượng quá khớp (Overfitting) và xử lý tốt dữ liệu nhiễu.

### 1.4 Các Metric đánh giá mô hình
Để đánh giá hiệu năng dự đoán, các độ đo sau được sử dụng:
- **MAE (Mean Absolute Error):** Trung bình giá trị tuyệt đối của sai số. Ý nghĩa: Mô hình dự đoán sai lệch trung bình bao nhiêu tuổi.
- **RMSE (Root Mean Squared Error):** Căn bậc hai của trung bình bình phương sai số. Phạt nặng các dự đoán sai lệch lớn.
- **R² (R-squared):** Hệ số xác định, biểu thị tỷ lệ phương sai của biến phụ thuộc có thể được giải thích bởi các biến độc lập. Giá trị $R^2$ càng gần 1 thì mô hình càng tốt.

---

## CHƯƠNG 2: BÀI TOÁN VÀ GIẢI PHÁP

### 2.1 Mô tả dữ liệu
Dữ liệu được sử dụng bao gồm các chỉ số kinh tế, xã hội, và môi trường của các quốc gia qua từng năm. Các trường dữ liệu (features) chính bao gồm:
- **population:** Tổng dân số của quốc gia.
- **pop_growth:** Tỷ lệ tăng trưởng dân số (%/năm).
- **gdp_growth:** Tỷ lệ tăng trưởng GDP (%/năm).
- **sanitation:** Tỷ lệ dân số được tiếp cận với các dịch vụ vệ sinh cơ bản (%).
- **electricity:** Tỷ lệ dân số có điện sử dụng (%).
- **water_access:** Tỷ lệ tiếp cận nguồn nước sạch (%).
- **co2_emissions:** Lượng phát thải CO2 (Tấn/người/năm).
- **labor_force:** Tỷ lệ lực lượng lao động.
- **Biến mục tiêu (Target):** `life_expectancy` - Tuổi thọ trung bình tính bằng số năm.

### 2.2 Tiền xử lý dữ liệu (Data Preprocessing)
- **Xử lý giá trị khuyết thiếu (Missing Values):** Dữ liệu thực tế có nhiều giá trị bị thiếu (NaN). Phương pháp xử lý là điền khuyết (Imputation) sử dụng giá trị trung vị (Median) cho các biến liên tục, do trung vị không bị ảnh hưởng mạnh bởi các giá trị ngoại lai (outliers) so với trung bình (Mean).
- **Chuẩn hóa (Normalization):** Mặc dù Extra Trees không nhạy cảm với việc chưa scale dữ liệu, nhưng trong quá trình xây dựng baseline với các mô hình tuyến tính, dữ liệu đã được scale để các biến có cùng phạm vi phân bố.

### 2.3 Trực quan hóa dữ liệu (EDA)
Trong pha Phân tích khám phá dữ liệu (EDA), chúng em đã tiến hành:
- **Vẽ phân phối (Distribution Plots):** Nhận thấy phân phối của tuổi thọ có xu hướng lệch trái (độ tuổi tập trung cao ở ngưỡng 70-80). GDP/người phân phối lệch phải cực đoan.
- **Ma trận tương quan (Correlation Matrix):** Phân tích cho thấy `life_expectancy` có tương quan thuận rất mạnh với `sanitation`, `electricity`, `water_access` và `gdp_per_capita`. Trong khi đó, tương quan với `co2_emissions` có tồn tại nhưng phức tạp hơn.
- **Insight:** Sự kết hợp giữa khả năng tiếp cận điện và nước sạch là tín hiệu mạnh mẽ nhất cho sự phát triển của y tế cơ sở, quyết định trực tiếp tới tuổi thọ ở các nước đang phát triển.

### 2.4 Feature Engineering (Trích xuất đặc trưng)
Thay vì đưa dữ liệu thô vào mô hình, nhóm đã thực hiện tạo thêm các biến mới để phản ánh sâu hơn bản chất bài toán:
1. **log1p GDP/người (`log1p_gdp_per_capita`):** Vì GDP có phân phối lệch (skewed) rất nặng và cách biệt lớn giữa các nước, việc áp dụng hàm logarit `log(1+x)` giúp làm mượt phân phối và giúp mô hình nhận diện xu hướng tuyến tính dễ dàng hơn.
2. **Interaction: electricity × water (`electricity_water_interaction`):** Tiếp cận cả nước sạch và điện tạo ra sự thay đổi về chất (ví dụ: bệnh viện hoạt động tốt, thực phẩm được bảo quản). Đặc trưng tương tác này được tạo bằng cách lấy `electricity` nhân với `water_access`.
3. **Đặc trưng độ trễ (`life_expectancy_lag1`):** Tuổi thọ của một quốc gia trong năm nay bị phụ thuộc mạnh vào tuổi thọ của chính quốc gia đó vào năm trước.
4. **Xu hướng 3 năm (`life_expectancy_trend_3y`):** Độ dốc (slope) thay đổi của tuổi thọ trong 3 năm gần nhất, nắm bắt quốc gia đó đang trong chu kỳ phục hồi hay suy thoái y tế.
5. **Độ biến động (`life_expectancy_volatility_3y`):** Độ lệch chuẩn trong 3 năm. Độ biến động cao báo hiệu khủng hoảng (chiến tranh, dịch bệnh).
6. **Biến chênh lệch (`diff1`):** Sự khác biệt của GDP (`log1p_gdp_per_capita_diff1`) và điều kiện vệ sinh (`sanitation_diff1`) so với năm liền kề trước đó.
7. **Flag biến cố (`is_covid`):** Gán nhãn bằng 1 cho các năm 2020-2022 để model biết đây là giai đoạn khủng hoảng y tế toàn cầu.

### 2.5 So sánh và Xây dựng mô hình
Nhóm đã thử nghiệm với 3 mô hình chính:
- **Linear Regression:** Được sử dụng làm Baseline. Kết quả R² thấp, không nắm bắt được các mối quan hệ phi tuyến giữa các yếu tố môi trường và tuổi thọ.
- **Random Forest:** Cho kết quả rất tốt, nhưng do dữ liệu có nhiều biến nhiễu ở một số quốc gia, mô hình có xu hướng overfit nhẹ.
- **Extra Trees Regressor:** Mô hình hoạt động tốt nhất. Do tính chất phân nhánh ngẫu nhiên, Extra Trees xử lý cực tốt dữ liệu nhiễu và độ biến động cao (đặc biệt trong giai đoạn Covid), cho phương sai thấp hơn Random Forest. Đây là lý do nhóm quyết định chọn **Extra Trees** làm mô hình cuối cùng.

---

## CHƯƠNG 3: THỰC NGHIỆM

### 3.1 Huấn luyện mô hình (Training)
Nhóm tự thiết kế và xây dựng lại thuật toán **MyExtraTreesRegressor** từ đầu để hiểu sâu về cách thuật toán hoạt động.
Các siêu tham số (Hyperparameters) tối ưu được lựa chọn qua quá trình thử nghiệm:
- `n_estimators`: 120, 300, hoặc 500 (Tùy cấu hình, với số lượng cây từ 120 đã cho độ ổn định cao).
- `max_depth`: 12 (Tránh việc cây mọc quá sâu dẫn đến overfitting).
- `min_samples_split`: 4 (Một node cần tối thiểu 4 mẫu để tiếp tục phân nhánh).
- `min_samples_leaf`: 2 (Số lượng mẫu tối thiểu tại một node lá).
- `max_features`: 0.8 (Chỉ sử dụng 80% features trong mỗi lần phân chia).
- Kỹ thuật Train: Train trên tập Training set (và Validation), đánh giá độc lập trên tập Test set.

### 3.2 Đánh giá mô hình (Evaluation)
Kết quả đánh giá trên tập kiểm thử (Test set) vô cùng khả quan:
- **MAE:** ~ 0.464 (Sai số trung bình chưa đến nửa năm tuổi).
- **RMSE:** ~ 1.009 (Độ phân tán sai số khoảng 1 tuổi).
- **R²:** ~ 0.986 (Mô hình giải thích được 98.6% sự biến thiên của tuổi thọ).
**Nhận xét:** Các chỉ số đều cực kỳ ấn tượng. Với MAE thấp hơn 0.5 tuổi, mô hình hoàn toàn có đủ năng lực để đưa vào ứng dụng dự báo trong thực tế.

### 3.3 Phân tích Feature Importance
Thông qua thuộc tính `feature_importances_` của cây quyết định, các đặc trưng có tầm ảnh hưởng lớn nhất được xếp hạng:
1. **Tuổi thọ năm trước (`life_expectancy_lag1` - ~45.7%):** Biến dự báo mang tính lịch sử. Yếu tố nền tảng xã hội của năm trước là cốt lõi.
2. **Điện × Nước (`electricity_water_interaction` - ~40.1%):** Điều kiện sinh hoạt cơ bản. Sự kết hợp giữa Điện và Nước sạch quyết định rất lớn đến tuổi thọ trung bình.
3. **log1p GDP/người (~8.3%) và Vệ sinh (`sanitation` - ~3.8%):** Tiềm lực kinh tế và vệ sinh dịch tễ đóng góp phần còn lại.
**Insight rút ra:** Để cải thiện tuổi thọ một cách nhanh chóng nhất ở các quốc gia nghèo, thay vì chỉ tăng cường GDP, các chính phủ nên đầu tư xây dựng các trạm phát điện và đường ống nước sạch để người dân tiếp cận song song cả hai dịch vụ cơ bản này.

### 3.4 Xây dựng giao diện Demo
Để minh chứng cho khả năng ứng dụng, nhóm đã xây dựng một Dashboard bằng nền tảng **Streamlit**.
- **Cách hoạt động:** Người dùng chọn "Quốc gia" và "Năm". Hệ thống tự động truy xuất các thông số kinh tế - y tế tương ứng của quốc gia trong năm đó.
- **Trải nghiệm người dùng:** Các tham số đầu vào được hiển thị dưới dạng số nguyên/số thực trực quan (không sử dụng giá trị scale khó hiểu). Sau khi kiểm tra thông tin, người dùng ấn nút "🔮 Dự đoán tuổi thọ".
- **Đầu ra (Output):** Hiển thị tuổi thọ dự đoán bởi mô hình so sánh với tuổi thọ thực tế, đồng thời vẽ biểu đồ diễn biến (Trend) lịch sử tuổi thọ của quốc gia đó.

---

## KẾT LUẬN

**1. Tổng kết:**
Dự án đã hoàn thành các mục tiêu đề ra: Xây dựng thành công một 파ipeline Data Mining end-to-end, từ khâu xử lý dữ liệu thô, tạo đặc trưng phức tạp, huấn luyện thuật toán Extra Trees đến triển khai ứng dụng web trực quan. Mô hình đạt độ chính xác R² lên tới 0.986.

**2. Hạn chế:**
- Do hạn chế của dữ liệu, có một số quốc gia thiếu sót quá nhiều số liệu (đặc biệt các nước đang có chiến sự), dẫn đến việc điền giá trị trung vị đôi khi có thể làm sai lệch phân phối thực tế.
- Chưa tính đến các biến động bất ngờ mang tính đứt gãy ngoài Covid (như thiên tai cục bộ).

**3. Hướng phát triển:**
- Thu thập thêm các biến liên quan đến y tế chuyên sâu (Số giường bệnh/1000 dân, số lượng bác sĩ...).
- Thử nghiệm các mô hình Deep Learning (như LSTM, Transformer) cho bài toán Time-Series kết hợp (Time-series Regression) thay vì dùng mô hình Tree-based truyền thống.
- Cải thiện hệ thống Web Demo để cho phép người dùng tự điền tay (simulate) các chỉ số và xem dự đoán tuổi thọ tương lai.
