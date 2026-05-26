from typing import Any, Dict, Optional, Tuple

import numpy as np

from zaki_time_series_lib.models.base import BaseTimeSeriesModel


class ARIMAModel(BaseTimeSeriesModel):
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), trend: Optional[str] = None):
        super().__init__("ARIMA")
        self.order = order
        self.trend = trend
        self.params = {"order": order, "trend": trend}
        self._model = None
        self._fitted = None

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name}(order={self.order})")
        y = self._validate_data(y).flatten()
        from statsmodels.tsa.arima.model import ARIMA
        self.dlog.get_logger().info(f"Fitting ARIMA{self.order} on {len(y)} observations...")
        self._model = ARIMA(y, order=self.order, trend=self.trend, **kwargs)
        self._fitted = self._model.fit()
        self.is_fitted = True
        self.training_time = getattr(self._fitted, 'mle_retvals', {}).get('iterations', 0)
        self.dlog.get_logger().info(f"ARIMA fitted: AIC={self._fitted.aic:.2f}, BIC={self._fitted.bic:.2f}")
        self.dlog.get_logger().info(self._fitted.summary().as_text())
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"ARIMA predicting horizon={horizon}")
        pred = self._fitted.forecast(steps=horizon)
        self.dlog.get_logger().info(f"ARIMA predictions: min={pred.min():.4f}, max={pred.max():.4f}")
        return np.asarray(pred)

    def predict_in_sample(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        return np.asarray(self._fitted.fittedvalues)


class SARIMAModel(BaseTimeSeriesModel):
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
                 trend: Optional[str] = None):
        super().__init__("SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.params = {"order": order, "seasonal_order": seasonal_order, "trend": trend}
        self._fitted = None

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name}")
        y = self._validate_data(y).flatten()
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self.dlog.get_logger().info(f"Fitting SARIMA{self.order}x{self.seasonal_order} on {len(y)} obs...")
        model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order,
                        trend=self.trend, **kwargs)
        self._fitted = model.fit(disp=False)
        self.is_fitted = True
        self.dlog.get_logger().info(f"SARIMA fitted: AIC={self._fitted.aic:.2f}")
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"SARIMA predicting horizon={horizon}")
        pred = self._fitted.forecast(steps=horizon)
        return np.asarray(pred)


class AutoARIMAModel(BaseTimeSeriesModel):
    def __init__(self, seasonal: bool = True, m: int = 24, max_order: int = 5,
                 information_criterion: str = 'aic', stepwise: bool = True):
        super().__init__("AutoARIMA")
        self.seasonal = seasonal
        self.m = m
        self.max_order = max_order
        self.information_criterion = information_criterion
        self.stepwise = stepwise
        self.params = {
            "seasonal": seasonal, "m": m, "max_order": max_order,
            "ic": information_criterion, "stepwise": stepwise
        }
        self._fitted = None
        self._best_order = None

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name}")
        y = self._validate_data(y).flatten()
        try:
            from pmdarima import auto_arima
            self.dlog.get_logger().info("Running auto_arima search (this may take a while)...")
            self._fitted = auto_arima(
                y, seasonal=self.seasonal, m=self.m,
                max_order=self.max_order,
                information_criterion=self.information_criterion,
                stepwise=self.stepwise, trace=True,
                error_action='ignore', suppress_warnings=True, **kwargs
            )
            self._best_order = (self._fitted.order, self._fitted.seasonal_order)
            self.is_fitted = True
            self.dlog.get_logger().info(f"AutoARIMA selected: order={self._fitted.order}, "
                                         f"seasonal_order={self._fitted.seasonal_order}")
            self.dlog.get_logger().info(f"AutoARIMA AIC={self._fitted.aic():.2f}")
        except ImportError:
            self.dlog.get_logger().warning("pmdarima not installed, falling back to simple ARIMA(1,1,1)")
            from statsmodels.tsa.arima.model import ARIMA
            self._fitted = ARIMA(y, order=(1, 1, 1)).fit()
            self.is_fitted = True
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"AutoARIMA predicting horizon={horizon}")
        pred, conf_int = self._fitted.predict(n_periods=horizon, return_conf_int=True)
        self.dlog.get_logger().info(f"AutoARIMA predictions: min={pred.min():.4f}, max={pred.max():.4f}")
        return pred
