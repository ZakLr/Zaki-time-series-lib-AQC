from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge

from zaki_time_series_lib.models.base import BaseTimeSeriesModel


class _ESNCore:
    def __init__(self, reservoir_size: int, spectral_radius: float = 0.9,
                 input_scaling: float = 0.5, sparsity: float = 0.1,
                 leaky_rate: float = 0.3, seed: int = 42):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.leaky_rate = leaky_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.W_in = None
        self.W_res = None

    def _initialize_weights(self, input_dim: int):
        n = self.reservoir_size
        self.W_in = self.rng.uniform(-self.input_scaling, self.input_scaling, (n, input_dim))
        self.W_res = self.rng.uniform(-1, 1, (n, n))
        mask = self.rng.uniform(0, 1, (n, n)) > self.sparsity
        self.W_res[mask] = 0.0
        eigvals = np.linalg.eigvals(self.W_res)
        radius = max(abs(eigvals)) if len(eigvals) > 0 else 1.0
        if radius > 0:
            self.W_res *= self.spectral_radius / radius

    def compute_states(self, inputs: np.ndarray, n_warmup: int = 0) -> np.ndarray:
        T, d = inputs.shape
        states = np.zeros((T, self.reservoir_size))
        x = np.zeros(self.reservoir_size)
        for t in range(T):
            u = inputs[t]
            x = (1 - self.leaky_rate) * x + self.leaky_rate * np.tanh(
                self.W_in @ u + self.W_res @ x
            )
            states[t] = x
        return states


class ESNModel(BaseTimeSeriesModel):
    def __init__(self, reservoir_size: int = 500, spectral_radius: float = 0.9,
                 input_scaling: float = 0.5, sparsity: float = 0.1,
                 leaky_rate: float = 0.3, alpha_ridge: float = 1e-4,
                 n_warmup: int = 100, seed: int = 42):
        super().__init__(f"ESN_{reservoir_size}")
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.leaky_rate = leaky_rate
        self.alpha_ridge = alpha_ridge
        self.n_warmup = n_warmup
        self.seed = seed
        self.params.update({
            "reservoir_size": reservoir_size, "spectral_radius": spectral_radius,
            "input_scaling": input_scaling, "sparsity": sparsity,
            "leaky_rate": leaky_rate, "alpha_ridge": alpha_ridge,
            "n_warmup": n_warmup
        })
        self._esn = None
        self._readout = None
        self._input_dim = 1
        self._n_lags = 24

    def _build_lagged_input(self, y: np.ndarray) -> np.ndarray:
        X = np.zeros((len(y) - self._n_lags, self._n_lags))
        for i in range(self._n_lags):
            X[:, i] = y[i:len(y) - self._n_lags + i]
        return X

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name} (reservoir={self.reservoir_size})")
        y = self._validate_data(y).flatten()
        self._n_lags = kwargs.get('n_lags', self._n_lags)
        self._input_dim = self._n_lags if X is None else X.shape[1]

        if X is not None:
            inputs = np.asarray(X, dtype=np.float64)
            if len(inputs) != len(y):
                min_len = min(len(inputs), len(y))
                inputs = inputs[-min_len:]
                y = y[-min_len:]
        else:
            inputs = self._build_lagged_input(y)
            y = y[self._n_lags:]

        self.dlog.log_data_shape(inputs, "ESN inputs")
        self.dlog.log_data_shape(y, "ESN targets")

        self._esn = _ESNCore(
            self.reservoir_size, self.spectral_radius, self.input_scaling,
            self.sparsity, self.leaky_rate, self.seed
        )
        self._esn._initialize_weights(self._input_dim)
        self.dlog.get_logger().info(f"Reservoir initialized: {self.reservoir_size} units, "
                                     f"spectral_radius={self.spectral_radius}")

        states = self._esn.compute_states(inputs, self.n_warmup)
        states_train = states[self.n_warmup:]
        y_train = y[self.n_warmup:]

        self.dlog.log_data_shape(states_train, "Reservoir states (post-warmup)")

        self._readout = Ridge(alpha=self.alpha_ridge, fit_intercept=True)
        self._readout.fit(states_train, y_train)
        self.is_fitted = True

        train_pred = self._readout.predict(states_train)
        train_mse = np.mean((y_train - train_pred) ** 2)
        self.dlog.get_logger().info(f"ESN training MSE: {train_mse:.6f}")
        self.dlog.get_logger().info(f"Readout weights: min={self._readout.coef_.min():.6f}, "
                                     f"max={self._readout.coef_.max():.6f}")

        self._last_inputs = inputs[-self._n_lags:] if len(inputs) >= self._n_lags else inputs
        self._last_state = states[-1].copy()
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"ESN predicting horizon={horizon}")

        if X_future is not None and len(X_future) >= horizon:
            inputs = np.asarray(X_future[:horizon], dtype=np.float64)
            states = self._esn.compute_states(inputs)
            return self._readout.predict(states)

        predictions = []
        state = np.zeros(self.reservoir_size)

        last_window = kwargs.get('last_window', None)
        if last_window is None:
            if hasattr(self, '_last_state'):
                state = self._last_state.copy()
            if hasattr(self, '_last_inputs') and len(self._last_inputs) > 0:
                last_window = self._last_inputs[-1].flatten()
            else:
                last_window = np.zeros(self._n_lags)
        else:
            last_window = np.asarray(last_window).flatten()
            # Warmup: feed last_window repeatedly to settle reservoir state
            for _ in range(self.n_warmup):
                state = (1 - self._esn.leaky_rate) * state + self._esn.leaky_rate * np.tanh(
                    self._esn.W_in @ last_window + self._esn.W_res @ state
                )

        for i in range(horizon):
            u = last_window.reshape(1, -1)
            state = (1 - self._esn.leaky_rate) * state + self._esn.leaky_rate * np.tanh(
                self._esn.W_in @ u.flatten() + self._esn.W_res @ state
            )
            pred = self._readout.predict(state.reshape(1, -1))[0]
            predictions.append(pred)
            last_window = np.roll(last_window, -1)
            last_window[-1] = pred

        result = np.array(predictions)
        self.dlog.get_logger().info(f"ESN predictions: min={result.min():.6f}, max={result.max():.6f}")
        return result


class ESN500Model(ESNModel):
    def __init__(self, **kwargs):
        kwargs['reservoir_size'] = 500
        super().__init__(**kwargs)
        self.name = "ESN_500"


class ESN1000Model(ESNModel):
    def __init__(self, **kwargs):
        kwargs['reservoir_size'] = 1000
        super().__init__(**kwargs)
        self.name = "ESN_1000"
