# Life Expectancy Web Demo - Train trước, dự đoán khi bấm nút

Bản này khác bản cache toàn bộ dự đoán ở điểm:

- `train_model.py` train model một lần và lưu file `model_artifacts/trained_extra_trees.pkl`.
- `train_model.py` cũng chuẩn bị sẵn `feature_rows_for_web.csv`, tức là dữ liệu đã làm feature engineering cho từng quốc gia/năm.
- `app.py` không train lại model.
- Khi người dùng chọn quốc gia/năm và bấm **Dự đoán tuổi thọ**, web mới gọi `model.predict()` cho đúng một dòng được chọn.

## Cách đặt thư mục

Nên đặt thư mục này như sau:

```text
Mining/
├── Data/
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── Model/
    └── web_demo/
        ├── app.py
        ├── train_model.py
        └── extra_trees_model.py
```

## Cách chạy

```bash
cd Mining/Model/web_demo
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Nếu chỉ muốn kiểm tra nhanh luồng train trước khi chạy đủ 500 cây:

```bash
python train_model.py --n-estimators 5
streamlit run app.py
```

Khi nộp/demo chính thức, chạy lại:

```bash
python train_model.py
streamlit run app.py
```

## File sinh ra sau khi train

```text
model_artifacts/
├── trained_extra_trees.pkl       # model đã train + medians + danh sách feature
├── feature_rows_for_web.csv      # dữ liệu feature đã chuẩn bị sẵn cho web
├── test_metrics.json             # MAE, RMSE, R2 trên test
└── feature_importances.csv       # độ quan trọng feature
```

## Luồng hoạt động

1. Offline: train model bằng `python train_model.py`.
2. Web app load model và feature rows một lần bằng cache của Streamlit.
3. Người dùng chọn quốc gia và năm.
4. Web hiển thị thông số của quốc gia/năm đó.
5. Khi bấm **Dự đoán tuổi thọ**, web lấy đúng 10 feature, điền thiếu bằng median đã lưu, rồi gọi `model.predict()`.
