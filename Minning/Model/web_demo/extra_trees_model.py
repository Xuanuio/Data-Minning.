"""
Custom Extra Trees Regressor used for the life expectancy demo.

This module intentionally keeps the model implementation in plain NumPy so the
web app can reuse the same tuned model logic instead of switching to sklearn.
"""

from __future__ import annotations

import numpy as np


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
        n_random_thresholds=2,
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

    @staticmethod
    def _sse(y):
        if len(y) == 0:
            return 0.0
        return float(np.sum((y - np.mean(y)) ** 2))

    @staticmethod
    def _sse_from_sum(n, total_sum, total_sum2):
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
            raise RuntimeError("Tree chua duoc fit.")
        X = np.asarray(X, dtype=float)
        output = np.zeros(X.shape[0], dtype=float)
        self._predict_node_vectorized(X, self.root, np.arange(X.shape[0]), output)
        return output


class MyExtraTreesRegressor:
    def __init__(
        self,
        n_estimators=500,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.8,
        bootstrap=False,
        max_samples=1.0,
        n_random_thresholds=2,
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

            if self.verbose and ((i + 1) % 25 == 0 or (i + 1) == self.n_estimators):
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
            raise RuntimeError("ExtraTrees chua duoc fit.")
        X = np.asarray(X, dtype=float)
        all_predictions = np.zeros((len(self.trees), X.shape[0]), dtype=float)
        for i, tree in enumerate(self.trees):
            all_predictions[i] = tree.predict(X)
        return np.mean(all_predictions, axis=0)
