"""
Streamlit web demo: predict life expectancy only after the user clicks Confirm.

Important workflow:
1. Train once outside the web app:
       python train_model.py
2. Start web:
       streamlit run app.py
3. User selects country/year and clicks "Dự đoán tuổi thọ".
   Only then does the app call model.predict() for that one row.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Import is needed so pickle can restore MyExtraTreesRegressor objects.
import extra_trees_model  # noqa: F401
from config import (
    DISPLAY_LABELS,
    FEATURE_LABELS,
    FEATURE_ROWS_PATH,
    IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    SELECTED_FEATURES,
    TARGET_COL,
    YEAR_COL,
)


def inject_css():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        .hero {
            padding: 30px 32px;
            border-radius: 30px;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 54%, #38bdf8 100%);
            color: white;
            box-shadow: 0 22px 60px rgba(15, 23, 42, 0.26);
            margin-bottom: 24px;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.18rem;
            line-height: 1.15;
        }
        .hero p {
            margin-top: 11px;
            margin-bottom: 0;
            opacity: 0.94;
            font-size: 1.02rem;
        }
        .section-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 24px;
            padding: 22px 24px;
            background: rgba(255, 255, 255, 0.80);
            box-shadow: 0 18px 46px rgba(15, 23, 42, 0.08);
            margin-bottom: 20px;
        }
        .result-card {
            border: 1px solid rgba(59, 130, 246, 0.24);
            border-radius: 28px;
            padding: 24px 26px;
            background: linear-gradient(180deg, #ffffff 0%, #eff6ff 100%);
            box-shadow: 0 20px 52px rgba(37, 99, 235, 0.13);
            margin-bottom: 20px;
        }
        .metric-card {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 22px;
            padding: 18px 20px;
            background: #ffffff;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
            min-height: 118px;
        }
        .metric-label {
            color: #64748b;
            font-size: 0.88rem;
            margin-bottom: 8px;
        }
        .metric-value {
            color: #0f172a;
            font-size: 2rem;
            font-weight: 850;
            line-height: 1.1;
        }
        .metric-caption {
            color: #64748b;
            font-size: 0.85rem;
            margin-top: 8px;
        }
        .small-note {
            color: #64748b;
            font-size: 0.92rem;
        }
        .pill {
            display: inline-block;
            padding: 6px 11px;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            font-weight: 750;
            font-size: 0.82rem;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        .status-ok {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
            font-weight: 800;
            font-size: 0.86rem;
        }
        .status-warn {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            background: #fff7ed;
            color: #c2410c;
            border: 1px solid #fed7aa;
            font-weight: 800;
            font-size: 0.86rem;
        }
        div.stButton > button:first-child {
            width: 100%;
            border-radius: 18px;
            padding: 0.72rem 1rem;
            font-weight: 850;
            border: 0;
            background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 55%, #38bdf8 100%);
            color: white;
            box-shadow: 0 14px 30px rgba(37, 99, 235, 0.22);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chưa có model đã train. Hãy chạy trước: python train_model.py"
        )
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_feature_rows():
    if not FEATURE_ROWS_PATH.exists():
        raise FileNotFoundError(
            "Chưa có file feature_rows_for_web.csv. Hãy chạy trước: python train_model.py"
        )
    df = pd.read_csv(FEATURE_ROWS_PATH)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").round().astype("Int64")
    return df.dropna(subset=[YEAR_COL]).copy()


@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data(show_spinner=False)
def load_importances():
    if IMPORTANCE_PATH.exists():
        return pd.read_csv(IMPORTANCE_PATH)
    return pd.DataFrame(columns=["feature", "importance"])


def format_number(value, digits=3):
    if pd.isna(value):
        return "Không có dữ liệu"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value) >= 1_000_000:
        return f"{value:,.0f}"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:,.{digits}f}"


def metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_stat_table(row: pd.Series) -> pd.DataFrame:
    rows = []
    for col, label in DISPLAY_LABELS.items():
        raw_col = f"raw_{col}"
        if raw_col in row.index and pd.notna(row[raw_col]):
            val = row[raw_col]
        elif col in row.index:
            val = row[col]
        else:
            continue
        rows.append({"Chỉ số": label, "Giá trị (Thực tế)": format_number(val)})
    return pd.DataFrame(rows)


def make_feature_table(row: pd.Series, feature_columns) -> pd.DataFrame:
    rows = []
    for col in feature_columns:
        label = FEATURE_LABELS.get(col, col)
        if col in row.index:
            val = row[col]
            if pd.isna(val):
                formatted_val = "Không có dữ liệu"
            elif col in ["year", "is_covid"]:
                formatted_val = str(int(float(val)))
            else:
                formatted_val = format_number(val, digits=5)
            rows.append({"Feature đưa vào model": label, "Giá trị (Sau chuẩn hóa)": formatted_val})
    return pd.DataFrame(rows)


def predict_one_row(row: pd.Series, artifact: dict) -> float:
    feature_columns = artifact["feature_columns"]
    medians = artifact["medians"]
    values = []
    for col in feature_columns:
        value = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
        if pd.isna(value):
            value = medians.get(col, 0.0)
        values.append(float(value))
    X = np.asarray([values], dtype=float)
    return float(artifact["model"].predict(X)[0])


def render_actual_trend(country_df: pd.DataFrame, selected_year: int, predicted_value: float | None = None):
    chart_df = country_df[[YEAR_COL, TARGET_COL]].copy()
    chart_df = chart_df.dropna(subset=[TARGET_COL]).sort_values(YEAR_COL)
    if chart_df.empty:
        st.info("Quốc gia này chưa có dữ liệu tuổi thọ thực tế để vẽ biểu đồ.")
        return

    plot_df = chart_df.rename(columns={YEAR_COL: "Năm", TARGET_COL: "Tuổi thọ thực tế"}).set_index("Năm")
    st.line_chart(plot_df)

    if predicted_value is not None:
        st.caption(f"Điểm dự đoán cho năm {selected_year}: {predicted_value:.3f} tuổi.")


def main():
    st.set_page_config(
        page_title="Hệ thống dự đoán tuổi thọ",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>🌍  Hệ thống dự đoán tuổi thọ theo quốc gia và năm</h1>
            <p>
                Model được train trước và lưu thành file. Khi người dùng chọn quốc gia/năm rồi bấm xác nhận,
                ứng dụng mới tạo vector đầu vào cho dòng đó và gọi model.predict().
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        artifact = load_model_artifact()
        df = load_feature_rows()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python train_model.py\nstreamlit run app.py", language="bash")
        st.stop()

    metrics = load_metrics()
    importances = load_importances()
    feature_columns = artifact.get("feature_columns", SELECTED_FEATURES)
    country_col = artifact.get("country_col", "country_code")

    with st.sidebar:
        st.markdown("### ⚙️ Trạng thái hệ thống")
        st.markdown('<span class="status-ok">Model đã được train sẵn</span>', unsafe_allow_html=True)
        st.caption("Web không train lại model khi người dùng truy vấn.")
        st.markdown("---")
        st.markdown("### Bộ tham số")
        st.json(artifact.get("params", {}), expanded=False)
        if metrics:
            st.markdown("### Test metrics")
            st.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
            st.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
            st.metric("R²", f"{metrics.get('R2', 0):.4f}")

    st.markdown(
        """
        <div class="section-card">
            <span class="pill">Bước 1: Chọn quốc gia</span>
            <span class="pill">Bước 2: Chọn năm dự đoán</span>
            <span class="pill">Bước 3: Tiến hành dự đoán tuổi thọ</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    country_labels = sorted(df["country_label"].dropna().unique().tolist())

    c1, c2, c3 = st.columns([2.2, 1.2, 1.1])
    with c1:
        selected_country_label = st.selectbox(
            "Nhập hoặc chọn quốc gia",
            options=country_labels,
            index=0,
            help="Có thể gõ tên quốc gia để tìm nhanh trong danh sách.",
        )

    selected_country_df = df[df["country_label"] == selected_country_label].copy()
    years = sorted(selected_country_df[YEAR_COL].dropna().astype(int).unique().tolist())
    default_year_index = len(years) - 1 if years else 0

    with c2:
        selected_year = st.selectbox(
            "Chọn năm cần dự đoán",
            options=years,
            index=default_year_index,
        )

    with c3:
        st.write("")
        st.write("")
        run_prediction = st.button("🔮 Dự đoán tuổi thọ", type="primary")

    selected_rows = selected_country_df[selected_country_df[YEAR_COL].astype(int) == int(selected_year)]
    if selected_rows.empty:
        st.warning("Không tìm thấy dòng dữ liệu tương ứng với quốc gia và năm đã chọn.")
        st.stop()

    row = selected_rows.iloc[0]

    st.markdown("### Thông số của quốc gia trong năm đã chọn")
    stats_col, feature_col = st.columns([1, 1])
    with stats_col:
        stat_table = make_stat_table(row)
        st.dataframe(stat_table, use_container_width=True, hide_index=True)
    with feature_col:
        feature_table = make_feature_table(row, feature_columns)
        st.dataframe(feature_table, use_container_width=True, hide_index=True)

    if not run_prediction:
        st.info("Sau khi kiểm tra các thông số đầu vào, bấm **Dự đoán tuổi thọ** để model bắt đầu dự đoán.")
        st.markdown("### Diễn biến tuổi thọ thực tế của quốc gia")
        render_actual_trend(selected_country_df, int(selected_year), None)
        return

    predicted = predict_one_row(row, artifact)
    actual = pd.to_numeric(pd.Series([row.get(TARGET_COL, np.nan)]), errors="coerce").iloc[0]
    abs_error = np.nan if pd.isna(actual) else abs(float(actual) - predicted)

    st.markdown("### Kết quả dự đoán")
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        metric_card("Tuổi thọ dự đoán", f"{predicted:.3f}", "Đơn vị: năm tuổi")
    with r2:
        metric_card("Tuổi thọ thực tế", format_number(actual), "Có sẵn trong dữ liệu" if not pd.isna(actual) else "Không có nhãn thực tế")
    with r3:
        metric_card("Sai số tuyệt đối", format_number(abs_error), "|thực tế - dự đoán|" if not pd.isna(abs_error) else "Không tính được")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Diễn biến tuổi thọ thực tế của quốc gia")
    render_actual_trend(selected_country_df, int(selected_year), predicted)

    with st.expander("Xem vector đúng lúc đưa vào model"):
        model_input = pd.DataFrame([{col: row.get(col, np.nan) for col in feature_columns}])
        st.dataframe(model_input, use_container_width=True, hide_index=True)

    if not importances.empty:
        with st.expander("Feature importance của model đã train"):
            plot_df = importances.copy()
            plot_df["feature_label"] = plot_df["feature"].map(FEATURE_LABELS).fillna(plot_df["feature"])
            st.dataframe(plot_df[["feature_label", "importance"]], use_container_width=True, hide_index=True)
            st.bar_chart(plot_df.set_index("feature_label")[["importance"]])


if __name__ == "__main__":
    main()
