from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model_artifacts"
MODEL_PATH = MODEL_DIR / "trained_extra_trees.pkl"
FEATURE_ROWS_PATH = MODEL_DIR / "feature_rows_for_web.csv"
METRICS_PATH = MODEL_DIR / "test_metrics.json"
IMPORTANCE_PATH = MODEL_DIR / "feature_importances.csv"

TARGET_COL = "life_expectancy"
YEAR_COL = "year"
COVID_YEARS = [2020, 2021, 2022]
TRAIN_ON_TRAIN_VAL = True

SELECTED_FEATURES = [
    "year",
    "is_covid",
    "log1p_gdp_per_capita",
    "sanitation",
    "electricity_water_interaction",
    "life_expectancy_lag1",
    "life_expectancy_trend_3y",
    "life_expectancy_volatility_3y",
    "log1p_gdp_per_capita_diff1",
    "sanitation_diff1",
]

OPTIMAL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": 0.8,
    "bootstrap": False,
    "max_samples": 1.0,
    "n_random_thresholds": 2,
    "min_impurity_decrease": 0.0,
    "random_state": 42,
    "verbose": 1,
}

COUNTRY_COL_CANDIDATES = ["country_code", "country_name", "country", "iso_code"]

DISPLAY_COLUMNS = [
    "country_name",
    "country_code",
    "year",
    "population",
    "pop_growth",
    "life_expectancy",
    "gdp_growth",
    "sanitation",
    "electricity",
    "water_access",
    "co2_emissions",
    "labor_force",
    "log1p_gdp_per_capita",
    "electricity_water_interaction",
    "basic_services_gap",
    "split",
]

DISPLAY_LABELS = {
    "population": "Dân số (Người)",
    "pop_growth": "Tăng trưởng dân số (%/năm)",
    "gdp_growth": "Tăng trưởng GDP (%/năm)",
    "sanitation": "Tiếp cận vệ sinh (%)",
    "electricity": "Tiếp cận điện (%)",
    "water_access": "Tiếp cận nước sạch (%)",
    "co2_emissions": "Phát thải CO₂ (Tấn/Người/Năm)",
    "labor_force": "Lực lượng lao động (%)",
    "log1p_gdp_per_capita": "log1p GDP/người",
    "electricity_water_interaction": "Tương tác điện × nước",
    "basic_services_gap": "Khoảng thiếu dịch vụ cơ bản",
}

FEATURE_LABELS = {
    "year": "Năm",
    "is_covid": "Có thuộc giai đoạn COVID",
    "log1p_gdp_per_capita": "log1p GDP/người",
    "sanitation": "Vệ sinh",
    "electricity_water_interaction": "Điện × nước",
    "life_expectancy_lag1": "Tuổi thọ năm trước",
    "life_expectancy_trend_3y": "Xu hướng tuổi thọ 3 năm",
    "life_expectancy_volatility_3y": "Biến động tuổi thọ 3 năm",
    "log1p_gdp_per_capita_diff1": "Chênh lệch log GDP 1 năm",
    "sanitation_diff1": "Chênh lệch vệ sinh 1 năm",
}
