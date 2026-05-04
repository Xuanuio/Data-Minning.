"""
Train model once, save it, and prepare feature rows for the Streamlit web demo.

Run from this folder:
    python train_model.py

After this script finishes, the web app will NOT train again. The app only loads:
    model_artifacts/trained_extra_trees.pkl
    model_artifacts/feature_rows_for_web.csv

Then prediction is executed only when the user clicks the prediction button.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    COUNTRY_COL_CANDIDATES,
    COVID_YEARS,
    DISPLAY_COLUMNS,
    IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    OPTIMAL_PARAMS,
    SELECTED_FEATURES,
    TARGET_COL,
    TRAIN_ON_TRAIN_VAL,
    YEAR_COL,
    FEATURE_ROWS_PATH,
    ROOT,
)
from extra_trees_model import MyExtraTreesRegressor


def find_data_dir() -> Path:
    """Find Data/processed regardless of whether the app is placed in Mining/Model or run elsewhere."""
    candidates = [
        ROOT.parents[1] / "Data" / "processed" if len(ROOT.parents) > 1 else None,  # Mining/Data/processed
        ROOT.parent / "Data" / "processed",                                         # fallback
        ROOT / "Data" / "processed",
        ROOT / "processed",
        ROOT.parent / "processed",
        Path.cwd() / "Data" / "processed",
        Path.cwd() / "data" / "processed",
        Path.cwd() / "processed",
        Path("/mnt/data/processed"),
    ]
    for data_dir in candidates:
        if data_dir is None:
            continue
        data_dir = data_dir.resolve()
        if (data_dir / "train.csv").exists() and (data_dir / "val.csv").exists() and (data_dir / "test.csv").exists():
            return data_dir
    raise FileNotFoundError(
        "Không tìm thấy thư mục processed chứa train.csv, val.csv, test.csv.\n"
        "Đặt thư mục web trong Mining/Model/... để script tự tìm Mining/Data/processed, "
        "hoặc chạy script tại thư mục có Data/processed."
    )


def get_country_col(df: pd.DataFrame) -> str:
    for col in COUNTRY_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Không tìm thấy cột quốc gia. Đã thử: {COUNTRY_COL_CANDIDATES}")


def ensure_log1p_gdp_per_capita(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "log1p_gdp_per_capita" not in output.columns:
        if "gdp_per_capita" not in output.columns:
            raise ValueError("Thiếu cả log1p_gdp_per_capita và gdp_per_capita.")
        output["log1p_gdp_per_capita"] = np.log1p(pd.to_numeric(output["gdp_per_capita"], errors="coerce"))
    return output


def add_is_covid(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["is_covid"] = output[YEAR_COL].isin(COVID_YEARS).astype(int)
    return output


def add_or_fix_electricity_water_interaction(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "electricity_water_interaction" not in output.columns:
        missing = [col for col in ["electricity", "water_access"] if col not in output.columns]
        if missing:
            raise ValueError("Thiếu cột để tạo electricity_water_interaction: " + str(missing))
        output["electricity_water_interaction"] = (
            pd.to_numeric(output["electricity"], errors="coerce")
            * pd.to_numeric(output["water_access"], errors="coerce")
            / 100.0
        )
    return output


def apply_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    output = ensure_log1p_gdp_per_capita(df)
    output = add_is_covid(output)
    output = add_or_fix_electricity_water_interaction(output)
    return output


def add_life_expectancy_history_features(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
    country_col: str,
    global_fill_value: float | None = None,
) -> pd.DataFrame:
    """Create lag/trend/volatility using only previous years for the same country."""
    if global_fill_value is None:
        global_fill_value = float(pd.to_numeric(history_df[TARGET_COL], errors="coerce").mean())

    history = history_df[[country_col, YEAR_COL, TARGET_COL]].copy()
    history[YEAR_COL] = pd.to_numeric(history[YEAR_COL], errors="coerce")
    history[TARGET_COL] = pd.to_numeric(history[TARGET_COL], errors="coerce")
    history = history.dropna(subset=[country_col, YEAR_COL, TARGET_COL]).sort_values([country_col, YEAR_COL])

    grouped_history = {country: group for country, group in history.groupby(country_col)}

    output = target_df.copy()
    lag1_values = []
    trend3_values = []
    volatility3_values = []

    for _, row in output.iterrows():
        country = row[country_col]
        year = row[YEAR_COL]

        if country not in grouped_history:
            past_values = np.array([], dtype=float)
        else:
            country_hist = grouped_history[country]
            past_values = country_hist[country_hist[YEAR_COL] < year][TARGET_COL].values.astype(float)

        recent = past_values[-3:]

        if len(recent) == 0:
            lag1 = float(global_fill_value)
            trend3 = 0.0
            volatility3 = 0.0
        elif len(recent) == 1:
            lag1 = float(recent[-1])
            trend3 = 0.0
            volatility3 = 0.0
        elif len(recent) == 2:
            lag1 = float(recent[-1])
            trend3 = float(recent[-1] - recent[-2])
            volatility3 = float(recent.std())
        else:
            lag1 = float(recent[-1])
            trend3 = float((recent[-1] - recent[0]) / 2.0)
            volatility3 = float(recent.std())

        lag1_values.append(lag1)
        trend3_values.append(trend3)
        volatility3_values.append(volatility3)

    output["life_expectancy_lag1"] = lag1_values
    output["life_expectancy_trend_3y"] = trend3_values
    output["life_expectancy_volatility_3y"] = volatility3_values
    return output


def add_diff1_feature(target_df: pd.DataFrame, history_df: pd.DataFrame, feature: str, country_col: str) -> pd.DataFrame:
    output = target_df.copy()
    diff_col = f"{feature}_diff1"

    if feature not in output.columns:
        raise ValueError(f"Không tạo được {diff_col} vì thiếu cột {feature}.")

    history = history_df[[country_col, YEAR_COL, feature]].copy()
    history[YEAR_COL] = pd.to_numeric(history[YEAR_COL], errors="coerce")
    history[feature] = pd.to_numeric(history[feature], errors="coerce")
    history = history.dropna(subset=[country_col, YEAR_COL, feature]).sort_values([country_col, YEAR_COL])
    grouped_history = {country: group for country, group in history.groupby(country_col)}

    diff_values = []
    for _, row in output.iterrows():
        country = row[country_col]
        year = row[YEAR_COL]
        current_value = row[feature]

        if pd.isna(current_value) or country not in grouped_history:
            diff_values.append(0.0)
            continue

        past = grouped_history[country][grouped_history[country][YEAR_COL] < year]
        previous_values = past[feature].dropna().values

        if len(previous_values) == 0:
            diff_values.append(0.0)
        else:
            diff_values.append(float(current_value - previous_values[-1]))

    output[diff_col] = diff_values
    return output


def build_features_for_train_and_web(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = apply_basic_features(train_df)
    val_df = apply_basic_features(val_df)
    test_df = apply_basic_features(test_df)

    country_col = get_country_col(train_df)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_train_df = pd.concat([train_df, val_df], ignore_index=True) if TRAIN_ON_TRAIN_VAL else train_df.copy()
    final_train_df = final_train_df.sort_values([country_col, YEAR_COL]).reset_index(drop=True)
    test_df = test_df.sort_values([country_col, YEAR_COL]).reset_index(drop=True)

    global_fill_value = float(pd.to_numeric(final_train_df[TARGET_COL], errors="coerce").mean())

    final_train_fe = add_life_expectancy_history_features(final_train_df, final_train_df, country_col, global_fill_value)
    test_fe = add_life_expectancy_history_features(test_df, final_train_df, country_col, global_fill_value)

    for feature in ["log1p_gdp_per_capita", "sanitation"]:
        final_train_fe = add_diff1_feature(final_train_fe, final_train_df, feature, country_col)
        test_fe = add_diff1_feature(test_fe, final_train_df, feature, country_col)

    feature_rows = pd.concat([final_train_fe, test_fe], ignore_index=True)
    feature_rows = feature_rows.sort_values([country_col, YEAR_COL]).reset_index(drop=True)

    missing = [col for col in SELECTED_FEATURES if col not in feature_rows.columns]
    if missing:
        raise ValueError("Thiếu feature sau feature engineering: " + str(missing))

    return final_train_fe, test_fe, feature_rows, country_col


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def evaluate(y_true, y_pred):
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score_manual(y_true, y_pred),
    }


def make_country_label(row: pd.Series, country_col: str) -> str:
    name = row.get("country_name", "")
    code = row.get("country_code", "")
    if pd.notna(name) and str(name).strip():
        if pd.notna(code) and str(code).strip():
            return f"{name} ({code})"
        return str(name)
    return str(row[country_col])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=None, help="Ghi đè số cây để test nhanh. Không truyền thì dùng 500.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = find_data_dir()
    print("Data dir:", data_dir)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    final_train_fe, test_fe, feature_rows, country_col = build_features_for_train_and_web(train_df, val_df, test_df)

    X_train_df = final_train_fe[SELECTED_FEATURES].copy()
    y_train = pd.to_numeric(final_train_fe[TARGET_COL], errors="coerce").values.astype(float)
    X_test_df = test_fe[SELECTED_FEATURES].copy()
    y_test = pd.to_numeric(test_fe[TARGET_COL], errors="coerce").values.astype(float)

    medians = X_train_df.median(numeric_only=True).fillna(0.0)
    X_train = X_train_df.fillna(medians).values.astype(float)
    X_test = X_test_df.fillna(medians).values.astype(float)

    params = dict(OPTIMAL_PARAMS)
    if args.n_estimators is not None:
        params["n_estimators"] = int(args.n_estimators)

    print("Country column:", country_col)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Features:", SELECTED_FEATURES)
    print("Params:", params)

    model = MyExtraTreesRegressor(**params)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    metrics = evaluate(y_test, y_pred_test)
    print("\nTEST METRICS")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    artifact = {
        "model": model,
        "feature_columns": SELECTED_FEATURES,
        "medians": medians.to_dict(),
        "country_col": country_col,
        "params": params,
        "test_metrics": metrics,
    }
    with MODEL_PATH.open("wb") as f:
        pickle.dump(artifact, f)

    # Web needs raw values + prepared model features, but not precomputed predictions.
    keep_cols = []
    for col in DISPLAY_COLUMNS + SELECTED_FEATURES:
        if col in feature_rows.columns and col not in keep_cols:
            keep_cols.append(col)
    feature_rows_web = feature_rows[keep_cols].copy()
    feature_rows_web["country_label"] = feature_rows_web.apply(lambda r: make_country_label(r, country_col), axis=1)

    try:
        unscaled_df = pd.read_csv(data_dir / "processed_data.csv")
        raw_cols = [c for c in DISPLAY_COLUMNS if c in unscaled_df.columns and c not in [country_col, YEAR_COL, "country_name", "split"]]
        unscaled_subset = unscaled_df[[country_col, YEAR_COL] + raw_cols].copy()
        
        rename_dict = {c: f"raw_{c}" for c in raw_cols}
        unscaled_subset = unscaled_subset.rename(columns=rename_dict)
        
        feature_rows_web = pd.merge(feature_rows_web, unscaled_subset, on=[country_col, YEAR_COL], how="left")
    except Exception as e:
        print("Không thể load processed_data.csv để lấy giá trị raw:", e)

    feature_rows_web.to_csv(FEATURE_ROWS_PATH, index=False)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if model.feature_importances_ is not None:
        feature_importance_df = pd.DataFrame({
            "feature": SELECTED_FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        feature_importance_df.to_csv(IMPORTANCE_PATH, index=False)

    print("\nSaved:")
    print("-", MODEL_PATH)
    print("-", FEATURE_ROWS_PATH)
    print("-", METRICS_PATH)
    print("-", IMPORTANCE_PATH)
    print("\nBây giờ chạy web bằng lệnh:")
    print("streamlit run app.py")


if __name__ == "__main__":
    main()
