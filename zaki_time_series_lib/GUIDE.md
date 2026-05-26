# Zaki Time Series Library — Complete Usage Guide

## Table of Contents

1. [Installation](#1-installation)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Loading](#3-data-loading)
4. [Preprocessing](#4-preprocessing)
5. [Models](#5-models)
6. [Benchmarking](#6-benchmarking)
7. [Pipeline](#7-pipeline)
8. [Results & Exports](#8-results--exports)
9. [Cross-Validation](#9-cross-validation)
10. [Verification & Diagnostics](#10-verification--diagnostics)
11. [Custom Model Registration](#11-custom-model-registration)
12. [Full Reference Examples](#12-full-reference-examples)

---

## 1. Installation

```bash
# From the library root
pip install -r requirements.txt

# Required dependencies
numpy pandas scipy statsmodels scikit-learn                  # Core
matplotlib seaborn tqdm                                      # Visualization
torch                                                         # Deep learning
requests pyarrow                                              # Data loading

# Optional but recommended
pmdarima     # AutoARIMA
arch         # GARCH
xgboost      # XGBoost model
lightgbm     # LightGBM model
tabulate     # Markdown table export
```

---

## 2. Architecture Overview

```
zaki_time_series_lib/
  config/        # Global settings (paths, log levels, defaults)
  data/          # Dataset loaders + preprocessing pipeline
  models/        # All forecasting models
    statistical/ # Persistence, ARIMA, GARCH, Holt-Winters, Theta
    ml/          # scikit-learn wrappers (RF, XGBoost, SVR, etc.)
    deep_learning/ # LSTM, Transformer, N-BEATS, ESN (PyTorch)
  benchmark/     # Metrics, CV, statistical tests, runner
  pipeline/      # End-to-end orchestration
  utils/         # Logger, decorators, exporters, visualization
```

All models share a common interface via `BaseTimeSeriesModel`:

```python
model.fit(y_train, X_train)           # Train on historical data
model.predict(horizon, X_future)       # Forecast n steps ahead
model.fit_predict(y_train, y_test)     # Train + predict shortcut
model.get_params()                     # Get model hyperparameters
model.set_params(**params)             # Set hyperparameters
model.summary()                        # Print model summary
```

All datasets share a common interface via `BaseDatasetLoader`:

```python
loader.load()                          # Download + cache data
loader.get_splits()                    # train/val/test splits
loader.get_X_y(seq_len)                # Sliding window sequences
loader.get_metadata()                  # Shape, freq, columns info
```

---

## 3. Data Loading

### Built-in Datasets

| Dataset | Frequency | Samples | Features | Source |
|---------|-----------|---------|----------|--------|
| `ETTh1` | Hourly | 17,420 | 7 (oil temp + 6 weather) | Zhou et al. 2021 |
| `ETTh2` | Hourly | 17,420 | 7 | Zhou et al. 2021 |
| `ETTm1` | 15-min | 69,680 | 7 | Zhou et al. 2021 |
| `Weather` | 10-min | 52,696 | 21 | Max Planck Institute |
| `Electricity` | Hourly | 26,304 | 321 | UCI |
| `Traffic` | Hourly | 17,544 | 862 | Caltrans |
| `ExchangeRate` | Daily | 7,588 | 8 | OANDA |

### Loading Data

```python
from zaki_time_series_lib.data import ETTh1Loader, WeatherLoader, ElectricityLoader

# Basic load (auto-downloads + caches to ~/.zaki_ts_data/)
loader = ETTh1Loader()
df = loader.load()
print(df.shape)  # (17420, 7)

# With custom cache and split ratios
loader = WeatherLoader(
    cache_dir="./my_cache",
    train_split=0.8,
    val_split=0.1,
    test_split=0.1
)
data = loader.load()

# Get metadata
meta = loader.get_metadata()
print(meta["freq"], meta["n_features"], meta["date_range"])

# Train/val/test splits
train_df, val_df, test_df = loader.get_splits()
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Sliding window sequences for DL models
X, y = loader.get_X_y(data, target_col=data.columns[0], sequence_length=168)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Load all feature columns (not just target)
loader.load()  # full DataFrame with all columns
```

### Loading Custom CSV

```python
from zaki_time_series_lib.data.datasets import CSVLoader

loader = CSVLoader(
    file_path="path/to/my_data.csv",
    name="MyDataset",
    date_col="timestamp"
)
df = loader.load()
```

### Listing Available Datasets

```python
from zaki_time_series_lib.data.datasets import list_available_datasets
for d in list_available_datasets():
    print(d)
```

---

## 4. Preprocessing

### Scalers

```python
from zaki_time_series_lib.data.preprocessing.scalers import (
    StandardScaler, MinMaxScaler, RobustScaler
)

# StandardScaler: zero mean, unit variance
scaler = StandardScaler()
scaled = scaler.fit_transform(train_array)
inverted = scaler.inverse_transform(scaled)

# MinMaxScaler: range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))

# RobustScaler: median/IQR (outlier-robust)
scaler = RobustScaler(quantile_range=(25, 75))
```

### Imputation

```python
from zaki_time_series_lib.data.preprocessing.imputation import (
    ForwardFillImputer, LinearInterpolationImputer, MedianImputer
)

# Forward fill
imputer = ForwardFillImputer(limit=24)
cleaned = imputer.fit_transform(df)

# Linear interpolation
imputer = LinearInterpolationImputer()
cleaned = imputer.fit_transform(df_with_nans)

# Median imputation
imputer = MedianImputer()
```

### Transforms

```python
from zaki_time_series_lib.data.preprocessing.transforms import (
    DifferencingTransformer, LogTransformer, BoxCoxTransformer
)

# Differencing
diff = DifferencingTransformer(order=1, seasonal_period=24)
stationary = diff.fit_transform(series)
original = diff.inverse_transform(stationary)

# Log transform (auto-adds offset for non-positive values)
log_t = LogTransformer()
transformed = log_t.fit_transform(series)

# Box-Cox (auto-estimates lambda)
boxcox = BoxCoxTransformer()
transformed = boxcox.fit_transform(series)
```

### Using String Aliases (in Pipeline)

```python
# In the pipeline, you can pass string names:
results = pipeline.run(
    dataset_name='ETTh1',
    model_configs=[...],
    scaler='standard',      # or 'minmax', 'robust'
)
```

---

## 5. Models

### Statistical Models

```python
from zaki_time_series_lib.models.statistical import (
    PersistenceModel, SeasonalNaiveModel,
    ARIMAModel, SARIMAModel, AutoARIMAModel,
    GARCHModel, ExponentialSmoothingModel,
    HoltWintersModel, ThetaModel
)

# Persistence (naive: repeat last value)
model = PersistenceModel()
model.fit(y_train)
pred = model.predict(horizon=24)

# Seasonal Naive (repeat last season)
model = SeasonalNaiveModel(season_period=24)

# ARIMA
model = ARIMAModel(order=(2, 1, 2), trend='c')

# SARIMA (with seasonality)
model = SARIMAModel(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 24)
)

# AutoARIMA (automatically selects p,d,q)
model = AutoARIMAModel(seasonal=True, m=24, stepwise=True)

# GARCH (for volatility forecasting)
model = GARCHModel(p=1, q=1, mean='Zero', dist='normal')

# Simple Exponential Smoothing
model = ExponentialSmoothingModel(smoothing_level=0.3)

# Holt-Winters (trend + seasonality)
model = HoltWintersModel(
    trend='add',
    seasonal='add',
    seasonal_periods=24,
    damped_trend=False
)

# Theta model
model = ThetaModel(period=24)
```

### ML Models (scikit-learn wrappers)

```python
from zaki_time_series_lib.models.ml import (
    LinearModel, RidgeModel, LassoModel,
    RandomForestModel, XGBoostModel, LightGBMModel,
    SVRModel, GaussianProcessModel, KNNModel
)

# All ML models auto-create lagged features internally
model = RandomForestModel(n_estimators=200, max_depth=10)
model.fit(y_train)
pred = model.predict(horizon=24)

# With external features
X_train = ...  # shape (n_samples, n_features)
X_test = ...   # shape (horizon, n_features)
model.fit(y_train, X_train)
pred = model.predict(horizon=24, X_future=X_test)

# Other ML models
ridge = RidgeModel(alpha=1.0)
xgb = XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1)
lgbm = LightGBMModel(n_estimators=100, num_leaves=31)
svr = SVRModel(kernel='rbf', C=1.0, epsilon=0.1)
gp = GaussianProcessModel()
knn = KNNModel(n_neighbors=5)
```

### Deep Learning Models (PyTorch)

```python
from zaki_time_series_lib.models.deep_learning import (
    LSTMModel, BiLSTMModel, GRUModel,
    CNNModel, TCNModel,
    TransformerModel, InformerModel,
    NBeatsModel,
    ESN500Model, ESN1000Model
)

# All DL models use sliding windows internally (default seq_len=168)

# LSTM
model = LSTMModel(hidden_dim=64, num_layers=2, dropout=0.2)
model.fit(y_train, seq_len=168, max_epochs=50, patience=10)
pred = model.predict(horizon=24)

# BiLSTM
model = BiLSTMModel(hidden_dim=64, num_layers=2, dropout=0.2)

# GRU
model = GRUModel(hidden_dim=64, num_layers=2, dropout=0.2)

# CNN (temporal convolutional)
model = CNNModel(hidden_channels=64, kernel_size=3, num_layers=2)

# TCN (dilated causal convolutions)
model = TCNModel(hidden_channels=64, num_layers=3, kernel_size=3)

# Transformer (encoder-only)
model = TransformerModel(d_model=64, nhead=4, num_layers=3)

# Informer (with ProbSparse attention)
model = InformerModel(d_model=64, nhead=4, num_layers=2)

# N-BEATS (interpretable trend/seasonality decomposition)
model = NBeatsModel(num_blocks=3, hidden_dim=256)

# ESN — Echo State Network (reservoir computing, no backprop)
# Reservoir size 500
model = ESN500Model(
    spectral_radius=0.9,
    input_scaling=0.5,
    sparsity=0.1,
    leaky_rate=0.3,
    alpha_ridge=1e-4,   # ridge regression regularization
    n_warmup=100         # washout period
)
model.fit(y_train)
pred = model.predict(horizon=24)

# Reservoir size 1000
model = ESN1000Model(
    spectral_radius=0.9,
    input_scaling=0.5,
    sparsity=0.1,
    leaky_rate=0.3,
    alpha_ridge=1e-4,
)
model.fit(y_train)
pred = model.predict(horizon=24)
```

#### ESN Hyperparameter Guide

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| `spectral_radius` | 0.1–1.5 | Larger = longer memory, risk of instability |
| `input_scaling` | 0.1–1.0 | Higher = more nonlinear dynamics |
| `sparsity` | 0.01–0.3 | % of recurrent connections kept |
| `leaky_rate` | 0.1–1.0 | 1.0 = no leakage, lower = slower dynamics |
| `alpha_ridge` | 1e-8–1e-1 | Regularization for output weights |
| `n_warmup` | 50–500 | Transient steps discarded before readout training |

### Training Customization (DL models)

```python
model.fit(
    y_train,
    seq_len=168,           # lookback window
    batch_size=64,
    lr=1e-3,
    max_epochs=100,
    patience=10,           # early stopping
    val_split=0.1          # validation split for early stopping
)

# Access training history
print(model.train_history)  # {"loss": [...], "val_loss": [...]}
```

---

## 6. Benchmarking

### Single Model

```python
from zaki_time_series_lib.benchmark.runner import BenchmarkRunner

runner = BenchmarkRunner(output_dir='./benchmark_results')
result = runner.run_single(
    model=ARIMAModel(order=(2, 1, 2)),
    y_train=y_train,
    y_test=y_test
)
print(result["metrics"]["rmse"])
```

### Multiple Models Comparison

```python
from zaki_time_series_lib.models.statistical import PersistenceModel, ARIMAModel
from zaki_time_series_lib.models.deep_learning import LSTMModel, ESN500Model

models = [
    PersistenceModel(),
    ARIMAModel(order=(2, 1, 2)),
    LSTMModel(hidden_dim=64),
    ESN500Model(),
]

results = runner.run_multiple(models, y_train, y_test)

# Print comparison table
runner.print_summary()

# Access per-model results
for name, result in results.items():
    if isinstance(result, dict) and result.get("success"):
        print(f"{name}: RMSE={result['metrics']['rmse']:.4f}")
```

### Cross-Validation

```python
from zaki_time_series_lib.benchmark.runner import BenchmarkRunner
from zaki_time_series_lib.benchmark.cross_validation import TimeSeriesSplitter

runner = BenchmarkRunner()

# Option 1: via runner
cv_results = runner.run_cross_validation(
    model=ARIMAModel(order=(2, 1, 2)),
    y=y_train,
    n_splits=5,
    test_size=24
)

# Option 2: manual splitter
splitter = TimeSeriesSplitter(n_splits=5, test_size=24, expanding=True)
for train_idx, test_idx in splitter.split(y):
    y_tr, y_te = y[train_idx], y[test_idx]
    model.fit(y_tr)
    pred = model.predict(len(y_te))
```

### Full Benchmark (data → split → scale → model → metrics → export)

```python
from zaki_time_series_lib.data import ETTh1Loader
from zaki_time_series_lib.models.statistical import PersistenceModel, ARIMAModel

runner = BenchmarkRunner(output_dir='./full_benchmark')
results = runner.run_full_benchmark(
    dataset_loader=ETTh1Loader(),
    models=[
        PersistenceModel(),
        ARIMAModel(order=(2, 1, 2)),
    ],
    target_col='HUFL',         # optional, defaults to first column
    scaler='standard',          # or a TimeSeriesScaler instance
    cv=True,                    # run cross-validation too
    cv_splits=5,
    cv_test_size=24
)

print(f"Best model: {results['best_model']}")
```

---

## 7. Pipeline (End-to-End)

The `TimeSeriesPipeline` is the recommended way to run a complete experiment:

```python
from zaki_time_series_lib.pipeline import TimeSeriesPipeline

pipeline = TimeSeriesPipeline(output_dir='./pipeline_results')

results = pipeline.run(
    dataset_name='ETTh1',
    model_configs=[
        {'name': 'Persistence', 'params': {}},
        {'name': 'SeasonalNaive', 'params': {'season_period': 24}},
        {'name': 'ARIMA', 'params': {'order': (2, 1, 2)}},
        {'name': 'LSTM', 'params': {'hidden_dim': 64}},
        {'name': 'ESN500', 'params': {'spectral_radius': 0.9}},
        {'name': 'ESN1000', 'params': {'spectral_radius': 0.85}},
    ],
    target_col='HUFL',          # target column name
    scaler='standard',          # auto-resolves from registry ('standard', 'minmax', 'robust')
    imputer=None,               # optional: ForwardFillImputer(), etc.
    transformer=None,           # optional: DifferencingTransformer(), etc.
    run_cv=True,                # run cross-validation on each model
    cv_splits=5,
    cv_test_size=24,
)

# View results
pipeline.print_summary()

# Get summary DataFrame
summary_df = pipeline.get_summary_dataframe()
print(summary_df)
```

### Pipeline Step-by-Step (Manual Mode)

```python
pipeline = TimeSeriesPipeline(output_dir='./manual_pipeline')

# Step 1: Load data
loader = pipeline.load_data('Weather')

# Step 2: Preprocess
preprocessed = pipeline.preprocess(
    data=loader.data,
    target_col='WetBulbCelsius',
    scaler='standard',
    imputer=None,
    transformer=None
)

# Step 3: Split
splits = pipeline.split_data(
    data=preprocessed['data'],
    target_col=preprocessed['target_col'],
    train_split=0.7,
    val_split=0.1,
    test_split=0.2
)

# Step 4: Create models
models = pipeline.create_models([
    {'name': 'LSTM', 'params': {'hidden_dim': 64}},
    {'name': 'Transformer', 'params': {'d_model': 64}},
])

# Step 5: Run benchmark
results = pipeline.benchmark.run_multiple(
    models,
    splits['train'],
    splits['test']
)
```

---

## 8. Results & Exports

All results are automatically saved to timestamped directories under the output folder:

```
zaki_results/
  run_20260521_120818/
    metrics.csv                   # Per-model metrics
    metrics_comparison.csv        # All models side-by-side
    metrics_comparison.md         # Markdown table
    metrics_comparison.tex        # LaTeX table
    forecasts.csv                 # All predictions + actuals
    predictions_Persistence.csv   # Per-model predictions + residuals
    params_Persistence.json       # Model hyperparameters
    run_summary.json              # Full experiment config and results
    comprehensive_report.html     # HTML report
```

### Using ResultsExporter Directly

```python
from zaki_time_series_lib.utils.results_exporter import ResultsExporter

exporter = ResultsExporter(output_dir='./exports')

# Export predictions with detailed error analysis
exporter.export_predictions('my_model', y_true, y_pred)

# Export training history
exporter.export_training_history('my_model', {"loss": [0.5, 0.3, ...], "val_loss": [0.6, 0.4, ...]})

# Export comparison table across models
exporter.export_metrics_comparison(all_metrics_dict)

# Export comprehensive HTML report
exporter.export_comprehensive_report({
    "config": {...},
    "metrics_comparison": {...},
    "best_model": "LSTM"
})
```

### Using TimeSeriesVisualizer

```python
from zaki_time_series_lib.utils.visualization import TimeSeriesVisualizer

viz = TimeSeriesVisualizer(output_dir='./plots')

# Single forecast plot with residuals
viz.plot_forecast(y_true, y_pred, model_name="LSTM")

# Multi-model comparison
viz.plot_multi_forecast(y_true, {"LSTM": pred1, "Transformer": pred2})

# Metrics bar chart
viz.plot_metrics_comparison(metrics_dict, metric_name="rmse")

# Training history
viz.plot_training_history(history, model_name="LSTM")

# Residual analysis (fitted vs residuals, Q-Q, histogram, ACF)
viz.plot_residual_analysis(y_true, y_pred, model_name="LSTM")

# Seasonal decomposition
viz.plot_seasonal_decomposition(series, period=24, model='additive')

# Correlation heatmap
viz.plot_correlation_heatmap(dataframe)
```

---

## 9. Cross-Validation

```python
from zaki_time_series_lib.benchmark.cross_validation import (
    TimeSeriesSplitter, RollingWindowCV, PurgedWalkForwardCV
)

# Expanding window (standard time series CV)
splitter = TimeSeriesSplitter(
    n_splits=5,
    test_size=24,
    gap=0,
    expanding=True
)

# Rolling window (fixed training window size)
splitter = RollingWindowCV(
    window_size=1000,
    test_size=24,
    step=24
)

# Purged walk-forward (for financial data, avoids leakage)
splitter = PurgedWalkForwardCV(
    n_splits=5,
    test_size=24,
    purge_size=24,      # gap between train and test
    embargo_size=12     # additional gap after train
)

for train_idx, test_idx in splitter.split(y):
    y_train, y_test = y[train_idx], y[test_idx]
    model.fit(y_train)
    pred = model.predict(len(y_test))
```

### Statistical Tests

```python
from zaki_time_series_lib.benchmark.statistical_tests import StatisticalTestSuite

tests = StatisticalTestSuite()

# Diebold-Mariano test
dm = tests.diebold_mariano(y_true, pred1, pred2, h=1)
print(f"DM statistic: {dm['DM_statistic']:.4f}, p-value: {dm['p_value']:.4f}")

# Compare all model pairs
all_tests = tests.summarize_all(y_true, {"LSTM": pred1, "ARIMA": pred2, "ESN": pred3})

# Residual diagnostics
diag = tests.model_significance(residuals)
print(f"Normality test p-value: {diag['shapiro_p_value']:.4f}")
```

---

## 10. Verification & Diagnostics

### 10.1 Lorenz 96 System

The Lorenz 96 system is a standard chaotic benchmark for time series forecasting:

```
dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
```

With N=40 variables and F=8, the system exhibits strong chaos (Lyapunov exponent ~1.7).

```python
from zaki_time_series_lib.benchmark.verification import Lorenz96

# Generate Lorenz96 data (default: N=40, F=8, dt=0.01)
lorenz = Lorenz96(n_vars=40, F=8.0, dt=0.01)
data = lorenz.generate(n_steps=10000, transient=1000)  # shape (10000, 40)
ts = lorenz.generate_single(n_steps=10000)              # shape (10000,) - first variable

# Split into train/test
y_train, y_test = Lorenz96.split_train_test(ts, train_ratio=0.7)

# Estimate Lyapunov exponent from data
lyap = lorenz.lyapunov_estimate(data)
print(f"Estimated Lyapunov exponent: {lyap:.4f}")

# Compute prediction horizon (how many steps until error exceeds threshold)
from zaki_time_series_lib.models.statistical import PersistenceModel
model = PersistenceModel()
model.fit(y_train)
pred = model.predict(len(y_test))
ph = Lorenz96.compute_prediction_horizon(y_test, pred, threshold=0.5)
print(f"Prediction horizon: {ph} steps")
```

### 10.2 ESP Verification (Echo State Property)

For Echo State Networks, the ESP ensures the reservoir state is uniquely determined by input history:

```python
from zaki_time_series_lib.benchmark.verification import ESPVerification

# Build a reservoir
from zaki_time_series_lib.models.deep_learning.esn import _ESNCore
reservoir = _ESNCore(reservoir_size=500, spectral_radius=0.9, input_scaling=0.5)
reservoir._initialize_weights(input_dim=24)

# Run full ESP verification
esp = ESPVerification()
results = esp.verify(reservoir.W_in, reservoir.W_res)
print(f"ESP passed: {results['esp_passed']}")
print(f"Spectral radius: {results['spectral_radius']:.4f}")
print(f"State forgetting converged: {results['state_forgetting']['converged']}")
print(f"Conditional Lyapunov: {results['conditional_lyapunov']:.6f}")

# Individual checks
sr = esp.check_spectral_radius(reservoir.W_res)
forgetting = esp.check_state_forgetting(reservoir.W_in, reservoir.W_res, n_inputs=2000)
lyap = esp.estimate_conditional_lyapunov(reservoir.W_in, reservoir.W_res)
```

### 10.3 FSDH (First-order Sensitivity to Data perturbation in Hidden space)

FSDH measures how sensitive a model's hidden representations (or predictions) are to small input perturbations. Lower values indicate more robust models.

```python
from zaki_time_series_lib.benchmark.verification import FSDHCalculator
import numpy as np

# From hidden states (requires a function that maps input -> hidden)
def get_hidden_states(inputs):
    states = reservoir.compute_states(inputs)
    return states

fsdh = FSDHCalculator()
result = fsdh.compute_from_hidden(
    hidden_func=get_hidden_states,
    inputs=some_inputs[:100],
    epsilon=1e-5,
    n_perturbations=10
)
print(f"FSDH (hidden): mean={result['fsdh_mean']:.6f}")

# From model predictions
from zaki_time_series_lib.models.deep_learning import ESN500Model
model = ESN500Model()
model.fit(y_train)

result_pred = fsdh.compute_from_predictions(
    model=model,
    inputs=lagged_inputs[:100],
    epsilon=1e-5
)
print(f"FSDH (pred): mean={result_pred['fsdh_pred_mean']:.6f}")
```

### 10.4 VPT (Valid Prediction Time)

VPT is the first forecast horizon where NRMSE exceeds a threshold (default 0.4).
It measures how far ahead predictions remain accurate — the "predictability horizon."

```python
from zaki_time_series_lib.benchmark.metrics import MetricsCalculator
import numpy as np

mc = MetricsCalculator()

# Multi-horizon arrays: shape (n_samples, n_horizons)
y_true_h = np.column_stack([...])   # each column = one horizon
y_pred_h = np.column_stack([...])

vpt_val = mc.vpt(y_true_h, y_pred_h, threshold=0.4)
print(f"VPT = {vpt_val}  (first horizon where NRMSE >= 0.4)")
```

### 10.5 FSDH (Forecast Skill Decay Horizon)

FSDH is the **last** horizon where the model beats persistence (lower RMSE than persistence).
It measures how long your model provides value over a naive baseline.

```python
y_persist_h = np.column_stack([...])  # persistence forecasts
fsdh_val = mc.fsdh(y_true_h, y_pred_h, y_persist_h)
print(f"FSDH = {fsdh_val}  (last horizon where model beats persistence)")
```

### 10.6 Multi-Horizon Evaluation

The full evaluation pipeline across multiple horizons simultaneously:

```python
from zaki_time_series_lib.benchmark.metrics import MetricsCalculator

mc = MetricsCalculator()
results = mc.evaluate_horizons(
    y_true_horizons=y_true_h,       # (n_samples, n_horizons)
    y_pred_horizons=y_pred_h,       # (n_samples, n_horizons)
    y_persist_horizons=persist_h,   # optional, needed for Skill + FSDH
    horizons=[1, 3, 6, 12, 24],    # the horizon labels
    label="MyModel",
    vpt_threshold=0.4,
)

# Per-horizon metrics
for hkey, hm in results["per_horizon"].items():
    print(f"{hkey}: RMSE={hm['rmse']:.4f}, NRMSE={hm['nrmse']:.4f}, "
          f"Skill={hm.get('skill', 'N/A'):.4f}")

# Global metrics
print(f"VPT  = {results['vpt']}")   # first h where NRMSE >= 0.4
print(f"FSDH = {results['fsdh']}")  # last h where model beats persistence
```

### Complete Model Summary Table

| Model | Type | Key Parameters | Best For |
|-------|------|---------------|----------|
| **Persistence** | Statistical | — | Naive baseline |
| **SeasonalNaive** | Statistical | `season_period` | Highly seasonal data |
| **ARIMA** | Statistical | `order=(p,d,q)` | Stationary/univariate |
| **SARIMA** | Statistical | `order`, `seasonal_order` | Seasonal + trend |
| **AutoARIMA** | Statistical | `m` (seasonality) | Automatic order selection |
| **GARCH** | Statistical | `p`, `q` | Volatility forecasting |
| **ExpSmoothing** | Statistical | `smoothing_level` | No trend/seasonality |
| **HoltWinters** | Statistical | `trend`, `seasonal` | Trend + seasonality |
| **Theta** | Statistical | `period` | Decomposition-based |
| **LinearRegression** | ML | — | Simple linear baseline |
| **Ridge** | ML | `alpha` | Regularized linear |
| **Lasso** | ML | `alpha` | Feature selection |
| **ElasticNet** | ML | `alpha`, `l1_ratio` | Mixed regularization |
| **RandomForest** | ML | `n_estimators`, `max_depth` | Non-linear, robust |
| **XGBoost** | ML | `n_estimators`, `max_depth` | SOTA tree-based |
| **LightGBM** | ML | `num_leaves` | Fast gradient boosting |
| **SVR** | ML | `kernel`, `C` | High-dimensional |
| **GaussianProcess** | ML | `kernel` | Uncertainty estimates |
| **KNN** | ML | `n_neighbors` | Pattern matching |
| **LSTM** | DL | `hidden_dim`, `num_layers` | Sequential memory |
| **BiLSTM** | DL | `hidden_dim`, `num_layers` | Bidirectional memory |
| **GRU** | DL | `hidden_dim`, `num_layers` | Lighter than LSTM |
| **CNN** | DL | `hidden_channels` | Local patterns |
| **TCN** | DL | `num_layers`, `kernel_size` | Dilated convolutions |
| **Transformer** | DL | `d_model`, `nhead` | Long-range dependencies |
| **Informer** | DL | `d_model`, `top_k` | Efficient attention (ProbSparse) |
| **N-BEATS** | DL | `num_blocks`, `hidden_dim` | Interpretable decomposition |
| **ESN500** | Reservoir | `spectral_radius`, `leaky_rate` | Fast training, chaotic systems |
| **ESN1000** | Reservoir | `spectral_radius`, `leaky_rate` | Larger memory, chaotic systems |

### Dataset Summary

| Dataset | Variables | Frequency | Period | Chaos Level |
|---------|-----------|-----------|--------|-------------|
| ETTh1 | 7 | Hourly | 2 years | Low |
| ETTh2 | 7 | Hourly | 2 years | Low |
| ETTm1 | 7 | 15-min | 2 years | Medium |
| Weather | 21 | 10-min | 4 years | Medium |
| Electricity | 321 | Hourly | 3 years | Low |
| Traffic | 862 | Hourly | 2 years | Low |
| ExchangeRate | 8 | Daily | 20 years | Medium |
| Lorenz96 (generated) | 40 | dt=0.01 | Configurable | High (F=8) |

### Metrics Summary

| Metric | Range | Perfect | Description |
|--------|-------|---------|-------------|
| MA | [0, ∞) | 0 | Mean Absolute error |
| MSE | [0, ∞) | 0 | Mean Squared error |
| RMSE | [0, ∞) | 0 | Root Mean Squared error |
| MAPE | [0, ∞) | 0% | Mean Absolute Percentage error |
| SMAPE | [0, 200] | 0% | Symmetric MAPE |
| MASE | [0, ∞) | 0 | Mean Absolute Scaled Error |
| R2 | (-∞, 1] | 1 | Coefficient of determination |
| MPE | (-∞, ∞) | 0% | Mean Percentage error |
| WMAPE | [0, ∞) | 0% | Weighted MAPE |
| Max Error | [0, ∞) | 0 | Maximum absolute error |
| MDA | [0, 100] | 100% | Mean Directional Accuracy |
| Correlation | [-1, 1] | 1 | Pearson correlation |
| AIC | (-∞, ∞) | -∞ | Akaike Info Criterion |
| BIC | (-∞, ∞) | -∞ | Bayesian Info Criterion |
| VPT | [0, ∞) | 1 | Variance of Prediction Trajectory |
| FSDH | [0, ∞) | 0 | Prediction sensitivity (lower = more robust) |

### Preprocessing Summary

| Step | Options | Purpose |
|------|---------|---------|
| **Imputation** | ffill, bfill, linear, median, dropna | Handle missing values |
| **Transform** | differencing, log, boxcox, power | Stationarity, normality |
| **Scaling** | standard, minmax, robust | Normalize feature ranges |

### Lorenz 96 as Forecasting Benchmark

The Lorenz 96 system is a standard test for forecasting models because:

- **Chaotic**: Small errors grow exponentially (Lyapunov exponent ~1.7 for F=8)
- **Multi-variate**: 40 spatially-extended variables with coupling
- **Deterministic**: Single trajectory, no noise
- **Prediction Horizon**: How many steps before error exceeds a threshold

To benchmark models on Lorenz96:

```python
from zaki_time_series_lib.benchmark.verification import Lorenz96
from zaki_time_series_lib.benchmark.runner import BenchmarkRunner
from zaki_time_series_lib.models.deep_learning import ESN500Model, LSTMModel
from zaki_time_series_lib.models.statistical import PersistenceModel

lorenz = Lorenz96(n_vars=40, F=8.0, dt=0.01)
data = lorenz.generate_single(n_steps=10000)
y_train, y_test = Lorenz96.split_train_test(data, train_ratio=0.7)

runner = BenchmarkRunner(output_dir='./lorenz96_benchmark')
results = runner.run_multiple(
    models=[PersistenceModel(), ESN500Model(), LSTMModel(hidden_dim=64)],
    y_train=y_train,
    y_test=y_test
)

# Compute prediction horizon for each model
for name, r in results.items():
    if isinstance(r, dict) and r.get("success"):
        ph = Lorenz96.compute_prediction_horizon(y_test, r["predictions"])
        print(f"{name}: prediction horizon = {ph} steps")
```

## 11. Custom Model Registration

You can register your own models:

```python
from zaki_time_series_lib.models.base import BaseTimeSeriesModel
from zaki_time_series_lib.pipeline.registry import ModelRegistry

class MyCustomModel(BaseTimeSeriesModel):
    def __init__(self, param1=10):
        super().__init__("MyCustom")
        self.param1 = param1
        self.params = {"param1": param1}

    def fit(self, y, X=None, **kwargs):
        y = self._validate_data(y).flatten()
        self._mean = y.mean()
        self.is_fitted = True
        self.dlog.get_logger().info(f"MyCustom fitted, mean={self._mean:.4f}")
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        return np.full(horizon, self._mean)

# Register globally
registry = ModelRegistry()
registry.register_model("MyCustom", MyCustomModel)

# Now use in pipeline
pipeline = TimeSeriesPipeline()
results = pipeline.run(
    dataset_name='ETTh1',
    model_configs=[
        {'name': 'MyCustom', 'params': {'param1': 20}},
    ]
)
```

Similarly for custom datasets:

```python
from zaki_time_series_lib.data.base_loader import BaseDatasetLoader
from zaki_time_series_lib.pipeline.registry import DatasetRegistry

class MyDataLoader(BaseDatasetLoader):
    def __init__(self):
        super().__init__("MyData")

    def _download(self):
        import pandas as pd
        return pd.DataFrame({"value": range(1000)})

registry = DatasetRegistry()
registry.register_dataset("MyData", MyDataLoader)
```

---

## 12. Full Reference Examples

### Example A: Quick Benchmark (2 Models, 1 Dataset)

```python
from zaki_time_series_lib.pipeline import TimeSeriesPipeline

pipeline = TimeSeriesPipeline(output_dir='./quick_benchmark')
results = pipeline.run(
    dataset_name='ETTh1',
    model_configs=[
        {'name': 'Persistence', 'params': {}},
        {'name': 'ARIMA', 'params': {'order': (2, 1, 2)}},
    ],
    scaler='standard',
)
pipeline.print_summary()
```

### Example B: Comprehensive Benchmark (All Model Types)

```python
from zaki_time_series_lib.pipeline import TimeSeriesPipeline

pipeline = TimeSeriesPipeline(output_dir='./comprehensive')
results = pipeline.run(
    dataset_name='Weather',
    target_col='WetBulbCelsius',
    model_configs=[
        # Statistical
        {'name': 'Persistence', 'params': {}},
        {'name': 'SeasonalNaive', 'params': {'season_period': 24}},
        {'name': 'ARIMA', 'params': {'order': (2, 1, 2)}},
        {'name': 'HoltWinters', 'params': {'seasonal_periods': 24}},
        {'name': 'Theta', 'params': {'period': 24}},
        # ML
        {'name': 'RandomForest', 'params': {'n_estimators': 100}},
        {'name': 'XGBoost', 'params': {'n_estimators': 100}},
        {'name': 'SVR', 'params': {'kernel': 'rbf'}},
        # Deep Learning
        {'name': 'LSTM', 'params': {'hidden_dim': 64}},
        {'name': 'Transformer', 'params': {'d_model': 64}},
        # Reservoir Computing
        {'name': 'ESN500', 'params': {'spectral_radius': 0.9}},
        {'name': 'ESN1000', 'params': {'spectral_radius': 0.85}},
    ],
    scaler='standard',
    run_cv=True,
    cv_splits=3,
    cv_test_size=24,
)
pipeline.print_summary()
```

### Example C: Manual Data Loading + Preprocessing + Custom Model

```python
import numpy as np
from zaki_time_series_lib.data import ETTh1Loader
from zaki_time_series_lib.data.preprocessing.scalers import MinMaxScaler
from zaki_time_series_lib.data.preprocessing.transforms import DifferencingTransformer
from zaki_time_series_lib.models.deep_learning import LSTMModel
from zaki_time_series_lib.benchmark.runner import BenchmarkRunner

# 1. Load
loader = ETTh1Loader()
data = loader.load()

# 2. Transform
diff = DifferencingTransformer(order=1, seasonal_period=24)
stationary = diff.fit_transform(data['HUFL'].values)

# 3. Scale
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(stationary.reshape(-1, 1)).flatten()

# 4. Split
n_train = int(len(scaled) * 0.7)
y_train, y_test = scaled[:n_train], scaled[n_train:]

# 5. Train
model = LSTMModel(hidden_dim=128, num_layers=2)
model.fit(y_train, seq_len=168, max_epochs=30)

# 6. Predict
pred_scaled = model.predict(len(y_test))

# 7. Inverse transform
pred = diff.inverse_transform(scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten())

# 8. Evaluate
from zaki_time_series_lib.benchmark.metrics import MetricsCalculator
mc = MetricsCalculator()
metrics = mc.compute_all(data['HUFL'].values[n_train:], pred)
print(f"RMSE: {metrics['rmse']:.4f}")
```

### Example D: Cross-Validation + Statistical Tests

```python
from zaki_time_series_lib.data import ETTh1Loader
from zaki_time_series_lib.models.statistical import ARIMAModel
from zaki_time_series_lib.models.deep_learning import ESN500Model
from zaki_time_series_lib.benchmark.cross_validation import TimeSeriesSplitter
from zaki_time_series_lib.benchmark.metrics import MetricsCalculator
from zaki_time_series_lib.benchmark.statistical_tests import StatisticalTestSuite

loader = ETTh1Loader()
y = loader.load()['HUFL'].values
y = (y - y.mean()) / y.std()

splitter = TimeSeriesSplitter(n_splits=5, test_size=48)
mc = MetricsCalculator()
tests = StatisticalTestSuite()

all_preds = {"ARIMA": [], "ESN500": []}
all_true = []

for fold, (tr_idx, te_idx) in enumerate(splitter.split(y)):
    y_tr, y_te = y[tr_idx], y[te_idx]
    all_true.extend(y_te)

    for name, model in [("ARIMA", ARIMAModel(order=(2, 1, 2))),
                        ("ESN500", ESN500Model(spectral_radius=0.9))]:
        model.fit(y_tr)
        pred = model.predict(len(y_te))
        all_preds[name].extend(pred)

all_true = np.array(all_true)
for name in all_preds:
    all_preds[name] = np.array(all_preds[name])
    m = mc.compute_all(all_true, all_preds[name])
    print(f"{name}: RMSE={m['rmse']:.4f}, MAE={m['ma']:.4f}")

dm = tests.diebold_mariano(all_true, all_preds["ARIMA"], all_preds["ESN500"])
print(f"DM test: statistic={dm['DM_statistic']:.4f}, p={dm['p_value']:.4f}, "
      f"better={dm['better_model']}")
```

---

## Logging System

Every module logs with context-aware detail. Set via environment variables:

```bash
set ZAKI_LOG_LEVEL=DEBUG       # More detail
set ZAKI_LOG_LEVEL=WARNING     # Quiet mode
set ZAKI_LOG_FILE=run.log      # Log to file
```

The `DetailedLogger` class provides:

- `.log_section(title)` — section headers
- `.log_data_shape(data)` — shape info
- `.log_data_stats(data)` — min/max/mean/std/NaN/Inf
- `.log_model_summary(model)` — parameter count
- `.log_training_progress(...)` — epoch-by-epoch training

## Configuration

All defaults in `config/settings.py` (dataclass). Override via env vars:

```bash
set ZAKI_DATA_CACHE=./my_cache
set ZAKI_RESULTS_DIR=./my_results
set ZAKI_DL_BATCH_SIZE=32
set ZAKI_DL_MAX_EPOCHS=200
set ZAKI_DL_LEARNING_RATE=0.001
set ZAKI_DL_DEVICE=cuda         # or cpu
set ZAKI_RANDOM_SEED=123
```
