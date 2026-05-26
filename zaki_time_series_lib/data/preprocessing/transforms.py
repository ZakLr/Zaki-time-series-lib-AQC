from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import special, stats

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger

logger = get_logger(__name__)


class SeriesTransformer(ABC):
    def __init__(self, name: str = "Transformer"):
        self.name = name
        self.dlog = DetailedLogger(f"preprocessing.{name}")

    @abstractmethod
    def fit_transform(self, data: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        pass

    def inverse_transform(self, data: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        raise NotImplementedError


class DifferencingTransformer(SeriesTransformer):
    def __init__(self, order: int = 1, seasonal_period: Optional[int] = None):
        super().__init__("DifferencingTransformer")
        self.order = order
        self.seasonal_period = seasonal_period
        self.first_vals = None

    def fit_transform(self, data):
        self.dlog.get_logger().info(f"Applying differencing order={self.order}, seasonal={self.seasonal_period}")
        arr = np.asarray(data, dtype=np.float64).flatten()
        result = arr.copy()
        self.first_vals = []

        for d in range(self.order):
            self.first_vals.append(result[0])
            result = np.diff(result)

        if self.seasonal_period:
            self.first_vals.append(result[:self.seasonal_period].copy())
            result = result[self.seasonal_period:] - result[:-self.seasonal_period]

        self.dlog.log_data_stats(result, "Differenced")
        return result

    def inverse_transform(self, data):
        arr = np.asarray(data, dtype=np.float64).flatten()
        result = arr.copy()

        if self.seasonal_period:
            seasonal_base = self.first_vals.pop()
            full = np.concatenate([seasonal_base, np.zeros(len(result))])
            for i in range(self.seasonal_period, len(full)):
                full[i] = result[i - self.seasonal_period] + full[i - self.seasonal_period]
            result = full[self.seasonal_period:]

        for d in range(self.order - 1, -1, -1):
            first = self.first_vals.pop()
            result = np.cumsum(np.concatenate([[first], result]))

        return result


class LogTransformer(SeriesTransformer):
    def __init__(self, offset: float = 0.0):
        super().__init__("LogTransformer")
        self.offset = offset

    def fit_transform(self, data):
        self.dlog.get_logger().info("Applying log transform")
        arr = np.asarray(data, dtype=np.float64)
        min_val = arr.min()
        if min_val <= 0:
            self.offset = abs(min_val) + 1.0
            self.dlog.get_logger().warning(f"Negative values, applying offset={self.offset:.4f}")
        result = np.log(arr + self.offset)
        self.dlog.log_data_stats(result, "Log-transformed")
        return result

    def inverse_transform(self, data):
        arr = np.asarray(data, dtype=np.float64)
        return np.exp(arr) - self.offset


class BoxCoxTransformer(SeriesTransformer):
    def __init__(self, lmbda: Optional[float] = None):
        super().__init__("BoxCoxTransformer")
        self.lmbda = lmbda
        self.fitted_lmbda = None

    def fit_transform(self, data):
        self.dlog.get_logger().info("Applying Box-Cox transform")
        arr = np.asarray(data, dtype=np.float64).flatten()
        offset = 0.0
        if arr.min() <= 0:
            offset = abs(arr.min()) + 1.0
            arr = arr + offset
            self.dlog.get_logger().warning(f"Negative values, applied offset={offset:.4f}")
        self.fitted_lmbda, _ = stats.boxcox(arr) if self.lmbda is None else (None, None)
        if self.fitted_lmbda is None:
            self.fitted_lmbda = self.lmbda
        transformed = stats.boxcox(arr, lmbda=self.fitted_lmbda)
        self.dlog.get_logger().info(f"Box-Cox λ={self.fitted_lmbda:.4f}")
        return transformed

    def inverse_transform(self, data):
        arr = np.asarray(data, dtype=np.float64).flatten()
        if self.fitted_lmbda == 0:
            return np.exp(arr)
        return np.power(arr * self.fitted_lmbda + 1, 1 / self.fitted_lmbda)


class PowerTransformer(SeriesTransformer):
    def __init__(self, exponent: float = 0.5):
        super().__init__("PowerTransformer")
        self.exponent = exponent

    def fit_transform(self, data):
        self.dlog.get_logger().info(f"Applying power transform (^{self.exponent})")
        arr = np.asarray(data, dtype=np.float64)
        if arr.min() < 0:
            offset = abs(arr.min()) + 1e-6
            arr = arr + offset
        result = np.power(arr, self.exponent)
        return result

    def inverse_transform(self, data):
        arr = np.asarray(data, dtype=np.float64)
        return np.power(arr, 1 / self.exponent)


class IdentityTransformer(SeriesTransformer):
    def fit_transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


TRANSFORMER_REGISTRY = {
    "differencing": DifferencingTransformer,
    "log": LogTransformer,
    "boxcox": BoxCoxTransformer,
    "power": PowerTransformer,
    "identity": IdentityTransformer,
}
