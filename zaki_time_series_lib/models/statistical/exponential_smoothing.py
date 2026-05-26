from typing import Optional, Tuple

import numpy as np

from zaki_time_series_lib.models.base import BaseTimeSeriesModel


class ExponentialSmoothingModel(BaseTimeSeriesModel):
    def __init__(self, smoothing_level: Optional[float] = None):
        super().__init__("ExponentialSmoothing")
        self.smoothing_level = smoothing_level
        self.params = {"smoothing_level": smoothing_level}
        self._fitted = None
        self._last_level = None

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name}")
        y = self._validate_data(y).flatten()
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        self.dlog.get_logger().info(f"Fitting SimpleExpSmoothing on {len(y)} observations...")
        model = SimpleExpSmoothing(y)
        self._fitted = model.fit(smoothing_level=self.smoothing_level, optimized=(self.smoothing_level is None))
        fitted_levels = np.asarray(self._fitted.level)
        self._last_level = fitted_levels[-1]
        self.is_fitted = True
        smoothing = self._fitted.params.get('smoothing_level', 'N/A')
        self.dlog.get_logger().info(f"ExponentialSmoothing: alpha={smoothing:.4f}")
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"ExponentialSmoothing predicting horizon={horizon}")
        pred = self._fitted.forecast(horizon)
        self.dlog.get_logger().info(f"ExponentialSmoothing: constant prediction = {float(np.asarray(pred)[0]):.4f}")
        return np.asarray(pred)


class HoltWintersModel(BaseTimeSeriesModel):
    def __init__(self, trend: Optional[str] = 'add', seasonal: Optional[str] = 'add',
                 seasonal_periods: int = 24, damped_trend: bool = False):
        super().__init__("HoltWinters")
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.params = {
            "trend": trend, "seasonal": seasonal,
            "seasonal_periods": seasonal_periods, "damped_trend": damped_trend
        }
        self._fitted = None

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name}")
        y = self._validate_data(y).flatten()
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.dlog.get_logger().info(
            f"Fitting Holt-Winters (trend={self.trend}, seasonal={self.seasonal}, "
            f"period={self.seasonal_periods}) on {len(y)} obs..."
        )
        model = ExponentialSmoothing(
            y, trend=self.trend, seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods, damped_trend=self.damped_trend
        )
        self._fitted = model.fit(**kwargs)
        self.is_fitted = True
        self.dlog.get_logger().info(f"HoltWinters fitted: AIC={self._fitted.aic:.2f}, "
                                     f"SSE={self._fitted.sse:.4f}")
        for name in ['smoothing_level', 'smoothing_trend', 'smoothing_seasonal']:
            if name in self._fitted.params:
                self.dlog.get_logger().info(f"  {name} = {self._fitted.params[name]:.4f}")
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"HoltWinters predicting horizon={horizon}")
        pred = self._fitted.forecast(horizon)
        self.dlog.get_logger().info(f"HoltWinters predictions: min={pred.min():.4f}, max={pred.max():.4f}")
        return np.asarray(pred)
