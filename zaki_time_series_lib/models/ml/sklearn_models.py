from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator

from zaki_time_series_lib.models.base import BaseTimeSeriesModel
from zaki_time_series_lib.data.preprocessing.scalers import StandardScaler


class _SklearnWrapper(BaseTimeSeriesModel):
    def __init__(self, name: str, estimator_class, default_params: Optional[Dict] = None):
        super().__init__(name)
        self.estimator_class = estimator_class
        self.estimator: Optional[BaseEstimator] = None
        self.default_params = default_params or {}
        self.params.update(self.default_params)
        self._use_lagged = True
        self._n_lags = 24
        self._scaler = StandardScaler()

    def _create_lagged_features(self, y: np.ndarray, n_lags: int) -> np.ndarray:
        X = np.zeros((len(y) - n_lags, n_lags))
        for i in range(n_lags):
            X[:, i] = y[i:len(y) - n_lags + i]
        return X

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name}")
        y = self._validate_data(y).flatten()
        self._n_lags = kwargs.get('n_lags', self._n_lags)

        if X is not None:
            X_train = np.asarray(X, dtype=np.float64)
            if len(X_train) != len(y):
                min_len = min(len(X_train), len(y))
                X_train = X_train[-min_len:]
                y = y[-min_len:]
        else:
            X_train = self._create_lagged_features(y, self._n_lags)
            y = y[self._n_lags:]

        self.dlog.log_data_shape(X_train, "Feature matrix")
        self.dlog.log_data_shape(y, "Target vector")

        X_scaled = self._scaler.fit_transform(X_train)

        est_params = self.default_params.copy()
        est_params.update(kwargs.get('estimator_params', {}))
        self.estimator = self.estimator_class(**est_params)

        self.dlog.get_logger().info(f"Starting training with {len(X_scaled)} samples...")
        self.estimator.fit(X_scaled, y)
        self.is_fitted = True
        self.dlog.get_logger().info(f"{self.name} training complete")

        if hasattr(self.estimator, 'feature_importances_'):
            self.dlog.get_logger().info(f"Feature importances: {self.estimator.feature_importances_}")
        if hasattr(self.estimator, 'coef_'):
            self.dlog.get_logger().info(f"Coefficients: {self.estimator.coef_}")
        if hasattr(self.estimator, 'intercept_'):
            self.dlog.get_logger().info(f"Intercept: {self.estimator.intercept_:.4f}")

        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"{self.name} predicting horizon={horizon}")

        predictions = []
        last_window = np.zeros(self._n_lags)

        if X_future is not None:
            X_future = np.asarray(X_future, dtype=np.float64)
            if len(X_future) >= horizon:
                X_f_scaled = self._scaler.transform(X_future[:horizon])
                pred = self.estimator.predict(X_f_scaled)
                return pred

        if hasattr(self, '_last_observed'):
            last_window = self._last_observed[-self._n_lags:]

        for i in range(horizon):
            X_in = last_window.reshape(1, -1)
            X_in_scaled = self._scaler.transform(X_in)
            pred = self.estimator.predict(X_in_scaled)[0]
            predictions.append(pred)
            last_window = np.roll(last_window, -1)
            last_window[-1] = pred

        result = np.array(predictions)
        self.dlog.get_logger().info(f"{self.name} predictions: min={result.min():.4f}, max={result.max():.4f}")
        return result


class LinearModel(_SklearnWrapper):
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        super().__init__("LinearRegression", LinearRegression, {"fit_intercept": True})


class RidgeModel(_SklearnWrapper):
    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        super().__init__("Ridge", Ridge, {"alpha": alpha, "fit_intercept": True})


class LassoModel(_SklearnWrapper):
    def __init__(self, alpha: float = 0.01):
        from sklearn.linear_model import Lasso
        super().__init__("Lasso", Lasso, {"alpha": alpha, "fit_intercept": True, "max_iter": 10000})


class ElasticNetModel(_SklearnWrapper):
    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5):
        from sklearn.linear_model import ElasticNet
        super().__init__("ElasticNet", ElasticNet,
                         {"alpha": alpha, "l1_ratio": l1_ratio, "fit_intercept": True, "max_iter": 10000})


class RandomForestModel(_SklearnWrapper):
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, n_jobs: int = -1):
        from sklearn.ensemble import RandomForestRegressor
        super().__init__("RandomForest", RandomForestRegressor,
                         {"n_estimators": n_estimators, "max_depth": max_depth, "n_jobs": n_jobs,
                          "random_state": 42})


class XGBoostModel(_SklearnWrapper):
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        super().__init__("XGBoost", None, {})
        self._estimator_class_name = "XGBRegressor"
        self._xgb_params = {
            "n_estimators": n_estimators, "max_depth": max_depth,
            "learning_rate": learning_rate, "random_state": 42,
            "verbosity": 0
        }
        self.params.update(self._xgb_params)

    def fit(self, y, X=None, **kwargs):
        try:
            from xgboost import XGBRegressor
            self.estimator_class = XGBRegressor
            self.default_params = self._xgb_params
            return super().fit(y, X, **kwargs)
        except ImportError:
            self.dlog.get_logger().warning("xgboost not installed, falling back to RandomForest")
            from sklearn.ensemble import RandomForestRegressor
            self.estimator_class = RandomForestRegressor
            self.default_params = {"n_estimators": 100, "max_depth": 10, "n_jobs": -1, "random_state": 42}
            return super().fit(y, X, **kwargs)


class LightGBMModel(_SklearnWrapper):
    def __init__(self, n_estimators: int = 100, max_depth: int = -1, learning_rate: float = 0.1,
                 num_leaves: int = 31):
        super().__init__("LightGBM", None, {})
        self._lgb_params = {
            "n_estimators": n_estimators, "max_depth": max_depth,
            "learning_rate": learning_rate, "num_leaves": num_leaves,
            "random_state": 42, "verbose": -1
        }
        self.params.update(self._lgb_params)
        self._estimator_class_name = "LGBMRegressor"

    def fit(self, y, X=None, **kwargs):
        try:
            from lightgbm import LGBMRegressor
            self.estimator_class = LGBMRegressor
            self.default_params = self._lgb_params
            return super().fit(y, X, **kwargs)
        except ImportError:
            self.dlog.get_logger().warning("lightgbm not installed, falling back to RandomForest")
            from sklearn.ensemble import RandomForestRegressor
            self.estimator_class = RandomForestRegressor
            self.default_params = {"n_estimators": 100, "max_depth": 10, "n_jobs": -1, "random_state": 42}
            return super().fit(y, X, **kwargs)


class SVRModel(_SklearnWrapper):
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1):
        from sklearn.svm import SVR
        super().__init__("SVR", SVR, {"kernel": kernel, "C": C, "epsilon": epsilon})


class GaussianProcessModel(_SklearnWrapper):
    def __init__(self, kernel=None, n_restarts_optimizer: int = 5):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        if kernel is None:
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        super().__init__("GaussianProcess", GaussianProcessRegressor,
                         {"kernel": kernel, "n_restarts_optimizer": n_restarts_optimizer,
                          "random_state": 42})

    def predict(self, horizon, X_future=None, **kwargs):
        preds = super().predict(horizon, X_future, **kwargs)
        if kwargs.get('return_std', False) and self.estimator is not None:
            return preds, None
        return preds


class KNNModel(_SklearnWrapper):
    def __init__(self, n_neighbors: int = 5, weights: str = 'distance'):
        from sklearn.neighbors import KNeighborsRegressor
        super().__init__("KNN", KNeighborsRegressor,
                         {"n_neighbors": n_neighbors, "weights": weights, "n_jobs": -1})
