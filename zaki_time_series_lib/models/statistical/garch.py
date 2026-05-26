from typing import Optional, Tuple

import numpy as np

from zaki_time_series_lib.models.base import BaseTimeSeriesModel


class GARCHModel(BaseTimeSeriesModel):
    def __init__(self, p: int = 1, q: int = 1, mean: str = 'Zero', dist: str = 'normal'):
        super().__init__("GARCH")
        self.p = p
        self.q = q
        self.mean = mean
        self.dist = dist
        self.params = {"p": p, "q": q, "mean": mean, "dist": dist}
        self._fitted = None

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name}({self.p},{self.q})")
        y = self._validate_data(y).flatten()
        try:
            from arch import arch_model
            self.dlog.get_logger().info(f"Fitting GARCH({self.p},{self.q}) with {self.mean} mean, "
                                         f"{self.dist} distribution on {len(y)} obs...")
            model = arch_model(y * 100, vol='Garch', p=self.p, q=self.q,
                               mean=self.mean, dist=self.dist)
            self._fitted = model.fit(disp='off', **kwargs)
            self.is_fitted = True
            self.dlog.get_logger().info(f"GARCH fitted: AIC={self._fitted.aic:.2f}, BIC={self._fitted.bic:.2f}")
            omega = self._fitted.params.get('omega', 0)
            alpha = self._fitted.params.get(f'alpha[{self.p}]' if self.p > 0 else 'alpha[1]', 0)
            beta = self._fitted.params.get(f'beta[{self.q}]' if self.q > 0 else 'beta[1]', 0)
            self.dlog.get_logger().info(f"GARCH params: ω={omega:.6f}, α={alpha:.6f}, β={beta:.6f}")
        except ImportError:
            self.dlog.get_logger().warning("arch package not installed. Install with: pip install arch")
            raise
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"GARCH predicting variance for horizon={horizon}")
        forecasts = self._fitted.forecast(horizon=horizon)
        variance = forecasts.variance.values[-1] / 10000
        self.dlog.get_logger().info(f"GARCH variance predictions: min={variance.min():.8f}, "
                                     f"max={variance.max():.8f}")
        return variance

    def predict_volatility(self, horizon: int) -> np.ndarray:
        variance = self.predict(horizon)
        return np.sqrt(variance)
