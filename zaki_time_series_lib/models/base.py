from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger
from zaki_time_series_lib.utils.decorators import timer

logger = get_logger(__name__)


class BaseTimeSeriesModel(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.dlog = DetailedLogger(f"model.{self.name}")
        self.is_fitted = False
        self.training_time = 0.0
        self.params: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None, **kwargs) -> "BaseTimeSeriesModel":
        pass

    @abstractmethod
    def predict(self, horizon: int, X_future: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        pass

    def fit_predict(self, y_train: np.ndarray, y_test: Optional[np.ndarray] = None,
                    X_train: Optional[np.ndarray] = None, X_test: Optional[np.ndarray] = None,
                    **kwargs) -> np.ndarray:
        self.fit(y_train, X_train, **kwargs)
        horizon = len(y_test) if y_test is not None else kwargs.get("horizon", 1)
        return self.predict(horizon, X_test, **kwargs)

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        for k, v in params.items():
            setattr(self, k, v)
        self.dlog.get_logger().info(f"Updated params: {params}")
        return self

    def summary(self) -> str:
        lines = [
            f"{'='*50}",
            f"Model: {self.name}",
            f"Fitted: {self.is_fitted}",
            f"Training Time: {self.training_time:.4f}s",
            f"Parameters:",
        ]
        for k, v in self.params.items():
            lines.append(f"  {k}: {v}")
        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def _validate_data(self, y: np.ndarray) -> np.ndarray:
        arr = np.asarray(y, dtype=np.float64)
        if len(arr) == 0:
            raise ValueError("Empty array provided")
        if np.any(np.isnan(arr)):
            logger.warning(f"{self.name}: NaN values detected in input data")
        if np.any(np.isinf(arr)):
            logger.warning(f"{self.name}: Inf values detected in input data")
        return arr
