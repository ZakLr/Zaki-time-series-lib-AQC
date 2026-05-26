from typing import Optional

import numpy as np

from zaki_time_series_lib.models.base import BaseTimeSeriesModel


class ThetaModel(BaseTimeSeriesModel):
    def __init__(self, period: int = 24):
        super().__init__("Theta")
        self.period = period
        self.params = {"period": period}
        self._fitted = None

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name}")
        y = self._validate_data(y).flatten()
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel as SMThetaModel
            self.dlog.get_logger().info(f"Fitting Theta model (period={self.period}) on {len(y)} obs...")
            self._fitted = SMThetaModel(y, period=self.period).fit()
            self.is_fitted = True
            self.dlog.get_logger().info(f"Theta model fitted successfully")
        except (ImportError, AttributeError):
            self.dlog.get_logger().warning("statsmodels ThetaModel not available, using manual implementation")
            self._manual_fit(y)
        return self

    def _manual_fit(self, y):
        n = len(y)
        ses_alpha = 0.5
        self._ses = np.zeros(n)
        self._ses[0] = y[0]
        for t in range(1, n):
            self._ses[t] = ses_alpha * y[t] + (1 - ses_alpha) * self._ses[t - 1]
        t = np.arange(n)
        self._drift_coef = np.polyfit(t, y, 1)[0]
        self._y = y
        self._n = n
        self.is_fitted = True

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"Theta predicting horizon={horizon}")

        if self._fitted is not None:
            pred = self._fitted.forecast(horizon)
            return pred.values

        t = np.arange(horizon) + self._n
        pred = self._ses[-1] + self._drift_coef * (t - self._n + 1) / 2
        return pred
