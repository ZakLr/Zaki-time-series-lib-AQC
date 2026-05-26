# Zaki Time Series Library

Time series forecasting with statistical, ML, deep learning, and reservoir computing models. Unified interface, built-in datasets, comprehensive benchmarks.

## Quick Install

```bash
pip install -r zaki_time_series_lib/requirements.txt
```

## Quick Start

```python
from zaki_time_series_lib.models.statistical import ARIMAModel
from zaki_time_series_lib.models.deep_learning import ESN500Model

model = ARIMAModel(order=(2, 1, 2))
model.fit(y_train)
pred = model.predict(horizon=24)
```

## Full Documentation

See [`zaki_time_series_lib/GUIDE.md`](zaki_time_series_lib/GUIDE.md) for complete API reference, examples, and benchmarks.

## Features

- **30+ models**: statistical (ARIMA, GARCH, Holt-Winters), ML (Ridge, XGBoost, RandomForest), deep learning (LSTM, Transformer, N-BEATS), reservoir computing (ESN)
- **7 built-in datasets**: ETTh1/2, ETTm1, Weather, Electricity, Traffic, ExchangeRate
- **Full evaluation pipeline**: multi-horizon metrics, cross-validation, Diebold-Mariano tests, VPT/FSDH
- **Echo State Networks**: configurable reservoirs with ESP verification and Lyapunov analysis
- **Preprocessing**: scaling, imputation, differencing, Box-Cox transformations
