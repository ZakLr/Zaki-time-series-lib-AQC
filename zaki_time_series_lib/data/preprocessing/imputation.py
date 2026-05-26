from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger

logger = get_logger(__name__)


class Imputer(ABC):
    def __init__(self, name: str = "Imputer"):
        self.name = name
        self.dlog = DetailedLogger(f"preprocessing.{name}")

    @abstractmethod
    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        pass


class ForwardFillImputer(Imputer):
    def __init__(self, limit: Optional[int] = None):
        super().__init__("ForwardFillImputer")
        self.limit = limit

    def fit_transform(self, data):
        self.dlog.get_logger().info(f"Applying forward fill (limit={self.limit})")
        if isinstance(data, pd.DataFrame):
            result = data.fillna(method='ffill', limit=self.limit)
            remaining = result.isna().sum().sum()
        else:
            arr = np.asarray(data, dtype=np.float64)
            result = pd.DataFrame(arr).fillna(method='ffill', limit=self.limit).values
            remaining = np.isnan(result).sum()
        if remaining > 0:
            self.dlog.get_logger().warning(f"{remaining} NaN values remain after forward fill")
        return result


class BackwardFillImputer(Imputer):
    def __init__(self, limit: Optional[int] = None):
        super().__init__("BackwardFillImputer")
        self.limit = limit

    def fit_transform(self, data):
        self.dlog.get_logger().info(f"Applying backward fill (limit={self.limit})")
        if isinstance(data, pd.DataFrame):
            result = data.fillna(method='bfill', limit=self.limit)
            remaining = result.isna().sum().sum()
        else:
            arr = np.asarray(data, dtype=np.float64)
            result = pd.DataFrame(arr).fillna(method='bfill', limit=self.limit).values
            remaining = np.isnan(result).sum()
        if remaining > 0:
            self.dlog.get_logger().warning(f"{remaining} NaN values remain after backward fill")
        return result


class LinearInterpolationImputer(Imputer):
    def __init__(self):
        super().__init__("LinearInterpolationImputer")

    def fit_transform(self, data):
        self.dlog.get_logger().info("Applying linear interpolation")
        if isinstance(data, pd.DataFrame):
            result = data.interpolate(method='linear', limit_direction='both')
            remaining = result.isna().sum().sum()
        else:
            s = pd.Series(np.asarray(data, dtype=np.float64).flatten())
            result = s.interpolate(method='linear', limit_direction='both').values.reshape(np.asarray(data).shape)
            remaining = np.isnan(result).sum()
        if remaining > 0:
            self.dlog.get_logger().warning(f"{remaining} NaN values remain after interpolation")
        return result


class MedianImputer(Imputer):
    def __init__(self):
        super().__init__("MedianImputer")

    def fit_transform(self, data):
        self.dlog.get_logger().info("Applying median imputation")
        if isinstance(data, pd.DataFrame):
            result = data.fillna(data.median())
        else:
            arr = np.asarray(data, dtype=np.float64)
            col_median = np.nanmedian(arr, axis=0)
            mask = np.isnan(arr)
            arr = np.where(mask, col_median, arr)
            result = arr
        return result


class DropNAImputer(Imputer):
    def __init__(self, axis: int = 0, how: str = 'any'):
        super().__init__("DropNAImputer")
        self.axis = axis
        self.how = how

    def fit_transform(self, data):
        self.dlog.get_logger().info(f"Dropping NAs (axis={self.axis}, how={self.how})")
        if isinstance(data, pd.DataFrame):
            n_before = len(data)
            result = data.dropna(axis=self.axis, how=self.how)
            n_after = len(result)
            self.dlog.get_logger().info(f"Dropped {n_before - n_after} rows with NaN")
            return result
        arr = np.asarray(data, dtype=np.float64)
        mask = np.isnan(arr).any(axis=1)
        return arr[~mask]


IMPUTER_REGISTRY = {
    "ffill": ForwardFillImputer,
    "bfill": BackwardFillImputer,
    "linear": LinearInterpolationImputer,
    "median": MedianImputer,
    "dropna": DropNAImputer,
}
