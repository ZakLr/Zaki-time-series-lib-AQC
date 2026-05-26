from typing import Optional

import numpy as np

from zaki_time_series_lib.models.base import BaseTimeSeriesModel


class PersistenceModel(BaseTimeSeriesModel):
    def __init__(self):
        super().__init__("Persistence")
        self.last_value = None
        self.params = {"type": "Naive (persistence)"}

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name}")
        y = self._validate_data(y).flatten()
        self.last_value = y[-1]
        self.is_fitted = True
        self.dlog.get_logger().info(f"Persistence model: last value = {self.last_value:.6f}")
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"Persistence predicting horizon={horizon}: repeating {self.last_value:.6f}")
        return np.full(horizon, self.last_value)


class SeasonalNaiveModel(BaseTimeSeriesModel):
    def __init__(self, season_period: int = 24):
        super().__init__("SeasonalNaive")
        self.season_period = season_period
        self.params = {"type": "Seasonal Naive", "season_period": season_period}
        self.last_season = None

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name}")
        y = self._validate_data(y).flatten()
        if len(y) < self.season_period:
            raise ValueError(f"Series length {len(y)} < season period {self.season_period}")
        self.last_season = y[-self.season_period:]
        self.is_fitted = True
        self.dlog.get_logger().info(f"SeasonalNaive: season_period={self.season_period}, "
                                     f"last_season shape={self.last_season.shape}")
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"SeasonalNaive predicting horizon={horizon}")
        repeats = (horizon + self.season_period - 1) // self.season_period
        pred = np.tile(self.last_season, repeats)[:horizon]
        self.dlog.get_logger().info(f"SeasonalNaive: repeated last {self.season_period} values {repeats} times")
        return pred
