from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger

logger = get_logger(__name__)


class TimeSeriesScaler(ABC):
    def __init__(self, name: str = "Scaler"):
        self.name = name
        self.dlog = DetailedLogger(f"preprocessing.{name}")
        self.fitted = False

    @abstractmethod
    def fit(self, data: Union[pd.DataFrame, np.ndarray]):
        pass

    @abstractmethod
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        pass

    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        self.dlog.get_logger().info(f"{self.name} fit_transform started")
        self.fit(data)
        result = self.transform(data)
        self.dlog.log_data_stats(result, "Transformed")
        return result

    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError


class StandardScaler(TimeSeriesScaler):
    def __init__(self):
        super().__init__("StandardScaler")
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, data):
        self.dlog.get_logger().info("Fitting StandardScaler (zero mean, unit variance)")
        arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.mean_ = np.nanmean(arr, axis=0)
        self.std_ = np.nanstd(arr, axis=0)
        self.std_[self.std_ == 0] = 1.0
        self.fitted = True
        self.dlog.get_logger().info(f"StandardScaler fitted: mean={self.mean_}, std={self.std_}")

    def transform(self, data):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted yet")
        arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        transformed = (arr - self.mean_) / self.std_
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(transformed, index=data.index, columns=data.columns)
        return transformed

    def inverse_transform(self, data):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted yet")
        arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        inverted = arr * self.std_ + self.mean_
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(inverted, index=data.index, columns=data.columns)
        return inverted


class MinMaxScaler(TimeSeriesScaler):
    def __init__(self, feature_range: tuple = (0, 1)):
        super().__init__("MinMaxScaler")
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, data):
        self.dlog.get_logger().info(f"Fitting MinMaxScaler (range={self.feature_range})")
        arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.min_ = np.nanmin(arr, axis=0)
        self.max_ = np.nanmax(arr, axis=0)
        self.max_[self.max_ - self.min_ == 0] = 1.0
        self.fitted = True

    def transform(self, data):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted yet")
        arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        scale = self.feature_range[1] - self.feature_range[0]
        transformed = self.feature_range[0] + (arr - self.min_) / (self.max_ - self.min_) * scale
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(transformed, index=data.index, columns=data.columns)
        return transformed

    def inverse_transform(self, data):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted yet")
        arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        scale = self.feature_range[1] - self.feature_range[0]
        inverted = self.min_ + (arr - self.feature_range[0]) / scale * (self.max_ - self.min_)
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(inverted, index=data.index, columns=data.columns)
        return inverted


class RobustScaler(TimeSeriesScaler):
    def __init__(self, quantile_range: tuple = (25, 75)):
        super().__init__("RobustScaler")
        self.quantile_range = quantile_range
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None

    def fit(self, data):
        self.dlog.get_logger().info(f"Fitting RobustScaler (quantiles={self.quantile_range})")
        arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.median_ = np.nanmedian(arr, axis=0)
        q_low, q_high = self.quantile_range
        self.iqr_ = np.nanpercentile(arr, q_high, axis=0) - np.nanpercentile(arr, q_low, axis=0)
        self.iqr_[self.iqr_ == 0] = 1.0
        self.fitted = True

    def transform(self, data):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted yet")
        arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        transformed = (arr - self.median_) / self.iqr_
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(transformed, index=data.index, columns=data.columns)
        return transformed

    def inverse_transform(self, data):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted yet")
        arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        inverted = arr * self.iqr_ + self.median_
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(inverted, index=data.index, columns=data.columns)
        return inverted


SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}
