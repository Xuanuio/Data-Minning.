# Simple final test runner for the tuned custom ExtraTrees model.
# Run: python test_extra_trees_fixed_params.py

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================
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
    "n_estimators": 120,
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

OUTPUT_DIR = Path("model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def find_data_dir():
    candidates = [
        Path("processed"),
        Path("data") / "processed",
        Path("..") / "data" / "processed",
        Path("/mnt/data/processed"),
    ]
    for data_dir in candidates:
        if (data_dir / "train.csv").exists() and (data_dir / "val.csv").exists() and (data_dir / "test.csv").exists():
            return data_dir
    raise FileNotFoundError("Khong tim thay processed/train.csv, processed/val.csv, processed/test.csv.")


def get_country_col(df):
    for col in COUNTRY_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Khong tim thay cot quoc gia. Thu cac cot: {COUNTRY_COL_CANDIDATES}")


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def ensure_log1p_gdp_per_capita(df):
    output = df.copy()
    if "log1p_gdp_per_capita" not in output.columns:
        if "gdp_per_capita" not in output.columns:
            raise ValueError("Thieu ca log1p_gdp_per_capita va gdp_per_capita.")
        output["log1p_gdp_per_capita"] = np.log1p(output["gdp_per_capita"].astype(float))
    return output


def add_is_covid(df):
    output = df.copy()
    output["is_covid"] = output[YEAR_COL].isin(COVID_YEARS).astype(int)
    return output


def add_or_fix_electricity_water_interaction(df):
    output = df.copy()
    if "electricity_water_interaction" not in output.columns:
        missing = [col for col in ["electricity", "water_access"] if col not in output.columns]
        if missing:
            raise ValueError("Thieu cot de tao electricity_water_interaction: " + str(missing))
        output["electricity_water_interaction"] = (
            output["electricity"].astype(float) * output["water_access"].astype(float) / 100.0
        )
    return output


def apply_basic_features(df):
    output = ensure_log1p_gdp_per_capita(df)
    output = add_is_covid(output)
    output = add_or_fix_electricity_water_interaction(output)
    return output


def add_life_expectancy_history_features(target_df, history_df, country_col, global_fill_value=None):
    if global_fill_value is None:
        global_fill_value = history_df[TARGET_COL].mean()

    history = history_df[[country_col, YEAR_COL, TARGET_COL]].copy()
    history = history.dropna(subset=[country_col, YEAR_COL, TARGET_COL])
    history = history.sort_values([country_col, YEAR_COL])

    grouped_history = {
        country: group.sort_values(YEAR_COL)
        for country, group in history.groupby(country_col)
    }

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
            x = np.array([0.0, 1.0, 2.0])
            trend3 = float(np.polyfit(x, recent.astype(float), 1)[0])
            volatility3 = float(recent.std())

        lag1_values.append(lag1)
        trend3_values.append(trend3)
        volatility3_values.append(volatility3)

    output["life_expectancy_lag1"] = lag1_values
    output["life_expectancy_trend_3y"] = trend3_values
    output["life_expectancy_volatility_3y"] = volatility3_values
    return output


def add_diff1_feature(target_df, history_df, feature, country_col):
    output = target_df.copy()
    diff_col = f"{feature}_diff1"

    if feature not in output.columns:
        raise ValueError(f"Khong tao duoc {diff_col} vi thieu cot {feature}.")

    grouped_history = {
        country: group.sort_values(YEAR_COL)
        for country, group in history_df.groupby(country_col)
    }

    diff_values = []
    for _, row in output.iterrows():
        country = row[country_col]
        year = row[YEAR_COL]
        current_value = row[feature]

        if pd.isna(current_value) or country not in grouped_history:
            diff_values.append(0.0)
            continue

        past = grouped_history[country][grouped_history[country][YEAR_COL] < year]
        previous_values = past[feature].dropna().values if feature in past.columns else []

        if len(previous_values) == 0:
            diff_values.append(0.0)
        else:
            diff_values.append(float(current_value - previous_values[-1]))

    output[diff_col] = diff_values
    return output


def build_features_for_final_train_and_test(train_df, val_df, test_df):
    train_df = apply_basic_features(train_df)
    val_df = apply_basic_features(val_df)
    test_df = apply_basic_features(test_df)

    country_col = get_country_col(train_df)

    if TRAIN_ON_TRAIN_VAL:
        final_train_df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        final_train_df = train_df.copy()

    global_fill_value = final_train_df[TARGET_COL].mean()

    final_train_fe = add_life_expectancy_history_features(
        target_df=final_train_df,
        history_df=final_train_df,
        country_col=country_col,
        global_fill_value=global_fill_value,
    )

    test_fe = add_life_expectancy_history_features(
        target_df=test_df,
        history_df=final_train_df,
        country_col=country_col,
        global_fill_value=global_fill_value,
    )

    for feature in ["log1p_gdp_per_capita", "sanitation"]:
        final_train_fe = add_diff1_feature(final_train_fe, final_train_df, feature, country_col)
        test_fe = add_diff1_feature(test_fe, final_train_df, feature, country_col)

    missing = [col for col in SELECTED_FEATURES if col not in final_train_fe.columns]
    if missing:
        raise ValueError("Thieu feature sau feature engineering: " + str(missing))

    return final_train_fe, test_fe, country_col


# ============================================================
# 3. PREPROCESSOR
# ============================================================
class NumericPreprocessor:
    def __init__(self):
        self.feature_columns = None
        self.medians = None

    def fit(self, X_df):
        self.feature_columns = X_df.columns.tolist()
        self.medians = X_df.median(numeric_only=True).fillna(0.0)
        return self

    def transform(self, X_df):
        X = X_df.reindex(columns=self.feature_columns)
        X = X.fillna(self.medians)
        return X.values.astype(float)


# ============================================================
# 4. METRICS
# ============================================================
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


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


# ============================================================
# 5. CUSTOM EXTRA TREES
# ============================================================
class TreeNode:
    def __init__(self, prediction=None, feature_index=None, threshold=None, left=None, right=None, is_leaf=True):
        self.prediction = prediction
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf


class MyExtraTreeRegressor:
    def __init__(
        self,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.8,
        n_random_thresholds=1,
        min_impurity_decrease=0.0,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_random_thresholds = n_random_thresholds
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root = None
        self.rng = np.random.default_rng(random_state)
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.root = self._build_tree(X, y, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total
        return self

    def _sse(self, y):
        if len(y) == 0:
            return 0.0
        return float(np.sum((y - np.mean(y)) ** 2))

    def _sse_from_sum(self, n, total_sum, total_sum2):
        if n <= 0:
            return 0.0
        return float(total_sum2 - (total_sum ** 2) / n)

    def _get_feature_indices(self, n_features):
        if self.max_features is None:
            size = n_features
        elif self.max_features == "sqrt":
            size = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            size = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            size = min(n_features, self.max_features)
        elif isinstance(self.max_features, float):
            size = max(1, int(n_features * self.max_features))
        else:
            size = n_features
        return self.rng.choice(n_features, size=size, replace=False)

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_sse = self._sse(y)
        best_feature = None
        best_threshold = None
        best_gain = 0.0
        best_score = np.inf

        for feature_idx in self._get_feature_indices(n_features):
            x_col = X[:, feature_idx]
            x_min = float(np.min(x_col))
            x_max = float(np.max(x_col))
            if x_min == x_max:
                continue

            thresholds = self.rng.uniform(
                low=x_min,
                high=x_max,
                size=max(1, int(self.n_random_thresholds)),
            )
            thresholds = np.unique(thresholds)

            for threshold in thresholds:
                left_mask = x_col <= threshold
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]

                sse_left = self._sse_from_sum(n_left, float(np.sum(y_left)), float(np.sum(y_left ** 2)))
                sse_right = self._sse_from_sum(n_right, float(np.sum(y_right)), float(np.sum(y_right ** 2)))
                score = sse_left + sse_right
                gain = parent_sse - score

                if score < best_score and gain > best_gain:
                    best_score = score
                    best_gain = float(gain)
                    best_feature = int(feature_idx)
                    best_threshold = float(threshold)

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        prediction = float(np.mean(y))

        if depth >= self.max_depth or len(y) < self.min_samples_split or np.var(y) < 1e-12:
            return TreeNode(prediction=prediction, is_leaf=True)

        feature_idx, threshold, gain = self._find_best_split(X, y)
        gain_per_sample = gain / max(1, len(y))

        if feature_idx is None or threshold is None or gain_per_sample < self.min_impurity_decrease:
            return TreeNode(prediction=prediction, is_leaf=True)

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return TreeNode(prediction=prediction, is_leaf=True)

        self.feature_importances_[feature_idx] += gain
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            prediction=prediction,
            feature_index=feature_idx,
            threshold=threshold,
            left=left_node,
            right=right_node,
            is_leaf=False,
        )

    def _predict_node_vectorized(self, X, node, indices, output):
        if node.is_leaf:
            output[indices] = node.prediction
            return

        selected_X = X[indices]
        left_mask = selected_X[:, node.feature_index] <= node.threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        if len(left_indices) > 0:
            self._predict_node_vectorized(X, node.left, left_indices, output)
        if len(right_indices) > 0:
            self._predict_node_vectorized(X, node.right, right_indices, output)

    def predict(self, X):
        if self.root is None:
            raise RuntimeError("Tree chua fit.")
        X = np.asarray(X, dtype=float)
        output = np.zeros(X.shape[0], dtype=float)
        self._predict_node_vectorized(X, self.root, np.arange(X.shape[0]), output)
        return output


class MyExtraTreesRegressor:
    def __init__(
        self,
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.8,
        bootstrap=False,
        max_samples=1.0,
        n_random_thresholds=1,
        min_impurity_decrease=0.0,
        random_state=42,
        verbose=1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_random_thresholds = n_random_thresholds
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.verbose = verbose
        self.trees = []
        self.rng = np.random.default_rng(random_state)
        self.feature_importances_ = None

    def _get_sample_indices(self, n_samples):
        if self.max_samples is None:
            sample_size = n_samples
        elif isinstance(self.max_samples, float):
            sample_size = max(1, int(n_samples * self.max_samples))
            sample_size = min(n_samples, sample_size)
        else:
            sample_size = min(n_samples, int(self.max_samples))

        if self.bootstrap:
            return self.rng.choice(n_samples, size=sample_size, replace=True)

        if sample_size < n_samples:
            return self.rng.choice(n_samples, size=sample_size, replace=False)

        return np.arange(n_samples)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]
        self.trees = []

        for i in range(self.n_estimators):
            sample_indices = self._get_sample_indices(n_samples)
            tree = MyExtraTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_random_thresholds=self.n_random_thresholds,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.random_state + i,
            )
            tree.fit(X[sample_indices], y[sample_indices])
            self.trees.append(tree)

            if self.verbose and ((i + 1) % 10 == 0 or (i + 1) == self.n_estimators):
                print(f"Da train {i + 1}/{self.n_estimators} cay")

        self._compute_feature_importances()
        return self

    def _compute_feature_importances(self):
        if len(self.trees) == 0:
            self.feature_importances_ = None
            return
        importances = [tree.feature_importances_ for tree in self.trees if tree.feature_importances_ is not None]
        if len(importances) == 0:
            self.feature_importances_ = None
            return
        self.feature_importances_ = np.mean(np.vstack(importances), axis=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total

    def predict(self, X):
        if len(self.trees) == 0:
            raise RuntimeError("ExtraTrees chua fit.")
        X = np.asarray(X, dtype=float)
        all_predictions = np.zeros((len(self.trees), X.shape[0]), dtype=float)
        for i, tree in enumerate(self.trees):
            all_predictions[i] = tree.predict(X)
        return np.mean(all_predictions, axis=0)


# ============================================================
# 6. RUN FINAL TEST
# ============================================================
def main():
    data_dir = find_data_dir()
    print("Data dir:", data_dir.resolve())

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    final_train_fe, test_fe, country_col = build_features_for_final_train_and_test(train_df, val_df, test_df)

    X_train_df = final_train_fe[SELECTED_FEATURES]
    y_train = final_train_fe[TARGET_COL].values.astype(float)

    X_test_df = test_fe[SELECTED_FEATURES]
    y_test = test_fe[TARGET_COL].values.astype(float)

    preprocessor = NumericPreprocessor()
    X_train = preprocessor.fit(X_train_df).transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    print("Country column:", country_col)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Features:", SELECTED_FEATURES)
    print("Params:", OPTIMAL_PARAMS)

    model = MyExtraTreesRegressor(**OPTIMAL_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    print("\nTEST METRICS")
    print("MAE :", round(metrics["MAE"], 6))
    print("RMSE:", round(metrics["RMSE"], 6))
    print("R2  :", round(metrics["R2"], 6))

    prediction_cols = [col for col in [country_col, YEAR_COL, TARGET_COL] if col in test_fe.columns]
    predictions_df = test_fe[prediction_cols].copy()
    predictions_df["y_pred"] = y_pred
    predictions_df["abs_error"] = np.abs(predictions_df[TARGET_COL] - predictions_df["y_pred"])

    predictions_path = OUTPUT_DIR / "test_predictions_fixed_params.csv"
    metrics_path = OUTPUT_DIR / "test_metrics_fixed_params.csv"
    importances_path = OUTPUT_DIR / "feature_importances_fixed_params.csv"

    predictions_df.to_csv(predictions_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    if model.feature_importances_ is not None:
        feature_importance_df = pd.DataFrame({
            "feature": SELECTED_FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        feature_importance_df.to_csv(importances_path, index=False)

    print("\nSaved:")
    print("-", predictions_path)
    print("-", metrics_path)
    print("-", importances_path)


if __name__ == "__main__":
    main()
