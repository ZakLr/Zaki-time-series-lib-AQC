r"""
===============================================================================
       ZAKI TIME SERIES LIBRARY — FULL EXPERIMENTATION SUITE
===============================================================================
Runs every model across every dataset with multi-horizon evaluation,
VPT, FSDH, Lorenz96 test, ESP verification, and detailed exports.

Usage:
    python -m zaki_time_series_lib.experiments.full_experiment

Output:
    ./experiment_output/run_<timestamp>/
        metrics_comparison.csv
        metrics_comparison.md
        comprehensive_report.html
        lorenz96_results.json
        esp_verification.json
        forecasts/*.png
        per_model/*.csv
"""

import os
import sys
import time
import json
import itertools
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ["ZAKI_LOG_LEVEL"] = "INFO"

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger
from zaki_time_series_lib.utils.results_exporter import ResultsExporter
from zaki_time_series_lib.utils.visualization import TimeSeriesVisualizer
from zaki_time_series_lib.utils.decorators import timer, TimerContext

from zaki_time_series_lib.data import (
    ETTh1Loader, ETTh2Loader, ETTm1Loader,
    WeatherLoader, ElectricityLoader, TrafficLoader, ExchangeRateLoader,
    DATASET_REGISTRY
)
from zaki_time_series_lib.data.preprocessing.scalers import StandardScaler, MinMaxScaler

from zaki_time_series_lib.models.statistical import (
    PersistenceModel, SeasonalNaiveModel, ARIMAModel,
    ExponentialSmoothingModel, HoltWintersModel, ThetaModel
)
from zaki_time_series_lib.models.ml import (
    LinearModel, RidgeModel, RandomForestModel, XGBoostModel, SVRModel
)
from zaki_time_series_lib.models.deep_learning import (
    LSTMModel, GRUModel, CNNModel, TransformerModel, ESN500Model, ESN1000Model
)

from zaki_time_series_lib.benchmark.metrics import MetricsCalculator
from zaki_time_series_lib.benchmark.runner import BenchmarkRunner
from zaki_time_series_lib.benchmark.verification import Lorenz96, ESPVerification
from zaki_time_series_lib.benchmark.statistical_tests import StatisticalTestSuite

from zaki_time_series_lib.pipeline import TimeSeriesPipeline

logger = get_logger("experiment")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path("./experiment_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / f"run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24]
SEQ_LEN = 168
VPT_THRESHOLD = 0.4
TRAIN_RATIO = 0.7
TEST_RATIO = 0.2

dlog = DetailedLogger("experiment")


def build_multi_horizon_data(series: np.ndarray, seq_len: int = 168,
                               horizons: List[int] = None) -> tuple:
    r"""
    Build multi-horizon sliding windows.
    X shape: (n_windows, seq_len)
    y shape: (n_windows, n_horizons)

    Returns: X, y (each horizon column = forecast target at that step ahead)
    """
    if horizons is None:
        horizons = HORIZONS
    max_h = max(horizons)
    n_windows = len(series) - seq_len - max_h + 1
    X = np.zeros((n_windows, seq_len))
    y = np.zeros((n_windows, len(horizons)))
    for i in range(n_windows):
        X[i] = series[i:i + seq_len]
        for hi, h in enumerate(horizons):
            y[i, hi] = series[i + seq_len + h - 1]
    return X, y


def make_persistence_forecast(X: np.ndarray, horizons: List[int] = None) -> np.ndarray:
    r"""
    Persistence forecast: repeat last observed value for each horizon.
    X shape: (n_windows, seq_len)
    Returns: (n_windows, n_horizons)
    """
    if horizons is None:
        horizons = HORIZONS
    last_val = X[:, -1]
    return np.column_stack([last_val] * len(horizons))


def make_seasonal_naive_forecast(X: np.ndarray, season: int = 24,
                                  horizons: List[int] = None) -> np.ndarray:
    if horizons is None:
        horizons = HORIZONS
    n_windows = len(X)
    n_h = len(horizons)
    preds = np.zeros((n_windows, n_h))
    for i in range(n_windows):
        season_vals = X[i, -season:]
        for hi, h in enumerate(horizons):
            preds[i, hi] = season_vals[(h - 1) % season]
    return preds


def train_and_predict_sk_model(model_cls, X_train, y_train, X_test,
                                 model_params: dict = None) -> np.ndarray:
    r"""
    Train a sklearn-style model and predict across all horizons.
    For each horizon, trains a separate model (direct strategy).
    """
    if model_params is None:
        model_params = {}
    n_h = y_train.shape[1]
    preds = np.zeros((len(X_test), n_h))
    for hi in range(n_h):
        m = model_cls(**model_params)
        m.fit(X_train, y_train[:, hi])
        preds[:, hi] = m.predict(X_test)
    return preds


def train_and_predict_esn(model_cls, X_train, y_train, X_test,
                           model_params: dict = None) -> np.ndarray:
    if model_params is None:
        model_params = {}
    n_h = y_train.shape[1]
    preds = np.zeros((len(X_test), n_h))
    for hi in range(n_h):
        m = model_cls(**model_params)
        m.fit(np.concatenate([X_train.flatten(), y_train[:, hi]]))
        preds[:, hi] = m.predict(len(X_test))
    return preds


def recursive_multi_horizon_predict(model, n_test_windows: int, seq_len: int,
                                     horizons: list) -> np.ndarray:
    r"""
    Generate multi-horizon predictions from a single recursive/iterative model.
    Model is already fitted. Generates one long forecast, then indexes
    for each test window and horizon.
    """
    max_h = max(horizons)
    n_forecast = n_test_windows + seq_len + max_h - 1
    forecast = model.predict(n_forecast)
    n_h = len(horizons)
    preds = np.zeros((n_test_windows, n_h))
    for i in range(n_test_windows):
        for hi, h in enumerate(horizons):
            preds[i, hi] = forecast[i + seq_len + h - 1]
    return preds


# =====================================================================
# PART 1: DATASET BENCHMARKS
# =====================================================================

@timer()
def run_dataset_benchmarks():
    dlog.log_section("PART 1: Dataset Benchmarks")
    datasets_to_run = ["ETTh1", "ETTh2", "Weather"]
    models_to_run = [
        ("Persistence", None, "stat"),
        ("SeasonalNaive", None, "stat"),
        ("ARIMA(2,1,2)", {"order": (2, 1, 2)}, "stat"),
        ("ExpSmoothing", None, "stat"),
        ("Ridge", {"alpha": 1.0}, "ml"),
        ("RandomForest", {"n_estimators": 100, "max_depth": 10}, "ml"),
        ("XGBoost", {"n_estimators": 100}, "ml"),
        ("LSTM", {"hidden_dim": 64}, "dl"),
        ("ESN500", {"spectral_radius": 0.9}, "esn"),
        ("ESN1000", {"spectral_radius": 0.85}, "esn"),
    ]

    all_results = {}
    exporter = ResultsExporter(str(RUN_DIR))
    viz = TimeSeriesVisualizer(str(RUN_DIR / "figures"))
    mc = MetricsCalculator()

    for ds_name in datasets_to_run:
        dlog.log_section(f"Dataset: {ds_name}")

        loader_cls = DATASET_REGISTRY[ds_name]
        loader = loader_cls()
        df = loader.load()
        series = df.iloc[:, 0].values.astype(np.float64)

        n_train = int(len(series) * TRAIN_RATIO)
        n_test = int(len(series) * TEST_RATIO)
        n_val = len(series) - n_train - n_test

        train_raw = series[:n_train]
        test_raw = series[n_train:n_train + n_test]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_raw.reshape(-1, 1)).flatten()

        X_train, y_train = build_multi_horizon_data(train_scaled, SEQ_LEN, HORIZONS)
        n_train_windows = len(X_train)
        split_cv = int(n_train_windows * 0.8)
        X_tr, y_tr = X_train[:split_cv], y_train[:split_cv]
        X_val, y_val = X_train[split_cv:], y_train[split_cv:]

        X_test_full = np.zeros((len(test_scaled) - SEQ_LEN - max(HORIZONS) + 1, SEQ_LEN))
        for i in range(len(X_test_full)):
            X_test_full[i] = test_scaled[i:i + SEQ_LEN]
        y_test_full = np.zeros((len(X_test_full), len(HORIZONS)))
        for i in range(len(X_test_full)):
            for hi, h in enumerate(HORIZONS):
                y_test_full[i, hi] = test_scaled[i + SEQ_LEN + h - 1]

        if len(X_test_full) == 0:
            dlog.get_logger().warning(f"Not enough test data for {ds_name}, skipping")
            continue

        n_test_final = min(300, len(X_test_full))
        X_test = X_test_full[:n_test_final]
        y_test = y_test_full[:n_test_final]

        persist_preds = make_persistence_forecast(X_test, HORIZONS)
        seasonal_preds = make_seasonal_naive_forecast(X_test, season=24, horizons=HORIZONS)

        dlog.get_logger().info(f"Train windows: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

        model_results = {}

        for model_name, params, model_type in models_to_run:
            dlog.log_subsection(f"Model: {model_name}")
            try:
                start_t = time.time()

                if model_type == "stat":
                    if model_name == "Persistence":
                        preds = make_persistence_forecast(X_test, HORIZONS)
                    elif model_name == "SeasonalNaive":
                        preds = make_seasonal_naive_forecast(X_test, season=24, horizons=HORIZONS)
                    elif "ARIMA" in model_name:
                        m = ARIMAModel(**params) if params else ARIMAModel()
                        m.fit(train_scaled)
                        preds = recursive_multi_horizon_predict(
                            m, len(X_test), SEQ_LEN, HORIZONS
                        )
                    elif "ExpSmoothing" in model_name:
                        m = ExponentialSmoothingModel()
                        m.fit(train_scaled)
                        preds = recursive_multi_horizon_predict(
                            m, len(X_test), SEQ_LEN, HORIZONS
                        )
                    else:
                        dlog.get_logger().warning(f"Unknown stat model {model_name}")
                        continue

                elif model_type == "ml":
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.linear_model import Ridge

                    if "Ridge" in model_name:
                        preds = train_and_predict_sk_model(Ridge, X_tr, y_tr, X_test, {"alpha": params["alpha"]})
                    elif "RandomForest" in model_name:
                        preds = train_and_predict_sk_model(RandomForestRegressor, X_tr, y_tr, X_test, params)
                    elif "XGBoost" in model_name:
                        try:
                            from xgboost import XGBRegressor
                            preds = train_and_predict_sk_model(XGBRegressor, X_tr, y_tr, X_test, params)
                        except ImportError:
                            dlog.get_logger().warning("XGBoost not installed, skipping")
                            continue
                    else:
                        continue

                elif model_type == "dl":
                    if len(X_tr) < 100:
                        dlog.get_logger().warning(f"Not enough data for DL model {model_name}, skipping")
                        continue
                    m = LSTMModel(**params) if "LSTM" in model_name else GRUModel(**params) if "GRU" in model_name else None
                    if m is None:
                        continue
                    m.fit(train_scaled, seq_len=SEQ_LEN, max_epochs=20, val_split=0.1)
                    preds = recursive_multi_horizon_predict(
                        m, len(X_test), SEQ_LEN, HORIZONS
                    )

                elif model_type == "esn":
                    m = ESN500Model(**params) if "ESN500" in model_name else ESN1000Model(**params)
                    m.fit(train_scaled, n_lags=SEQ_LEN)
                    preds = recursive_multi_horizon_predict(
                        m, len(X_test), SEQ_LEN, HORIZONS
                    )

                elapsed = time.time() - start_t

                preds_inv = np.zeros_like(preds)
                for hi in range(preds.shape[1]):
                    preds_inv[:, hi] = scaler.inverse_transform(preds[:, hi].reshape(-1, 1)).flatten()
                y_test_inv = np.zeros_like(y_test)
                for hi in range(y_test.shape[1]):
                    y_test_inv[:, hi] = scaler.inverse_transform(y_test[:, hi].reshape(-1, 1)).flatten()
                persist_inv = np.zeros_like(persist_preds)
                for hi in range(persist_preds.shape[1]):
                    persist_inv[:, hi] = scaler.inverse_transform(persist_preds[:, hi].reshape(-1, 1)).flatten()

                eval_results = mc.evaluate_horizons(
                    y_test_inv, preds_inv, persist_inv,
                    horizons=HORIZONS, label=model_name, vpt_threshold=VPT_THRESHOLD
                )
                eval_results["training_time"] = elapsed
                model_results[model_name] = eval_results

                dlog.get_logger().info(
                    f"  {model_name:20s} | VPT={eval_results['vpt']:2d} | "
                    f"FSDH={eval_results['fsdh']:2d} | Time={elapsed:.1f}s"
                )

            except Exception as e:
                dlog.get_logger().error(f"  {model_name} FAILED: {e}")
                continue

        all_results[ds_name] = model_results

        summary_rows = []
        for mname, mres in model_results.items():
            row = {"Model": mname, "VPT": mres["vpt"], "FSDH": mres["fsdh"]}
            for hkey, hm in mres["per_horizon"].items():
                row[f"{hkey}_RMSE"] = hm["rmse"]
                row[f"{hkey}_NRMSE"] = hm["nrmse"]
                if "skill" in hm:
                    row[f"{hkey}_Skill"] = hm["skill"]
            summary_rows.append(row)

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows).set_index("Model")
            csv_path = RUN_DIR / f"{ds_name}_summary.csv"
            df_summary.to_csv(csv_path)
            dlog.get_logger().info(f"\n{df_summary.to_string()}\n")

    return all_results


# =====================================================================
# PART 2: LORENZ96 BENCHMARK
# =====================================================================

@timer()
def run_lorenz96_benchmark():
    dlog.log_section("PART 2: Lorenz96 Chaotic Benchmark")
    mc = MetricsCalculator()

    lorenz = Lorenz96(n_vars=40, F=8.0, dt=0.02)
    data = lorenz.generate(n_steps=6000, transient=500, seed=RANDOM_SEED)
    ts = data[:, 0]
    lyap = lorenz.lyapunov_estimate(data, sample_every=5)
    dlog.get_logger().info(f"Lorenz96 estimated Lyapunov exponent: {lyap:.4f}")

    n_train = int(len(ts) * TRAIN_RATIO)
    n_test = len(ts) - n_train
    train_raw = ts[:n_train]
    test_raw = ts[n_train:]

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
    test_s = scaler.transform(test_raw.reshape(-1, 1)).flatten()

    X_tr, y_tr = build_multi_horizon_data(train_s, SEQ_LEN, HORIZONS)

    n_test_windows = len(test_s) - SEQ_LEN - max(HORIZONS) + 1
    X_te = np.zeros((n_test_windows, SEQ_LEN))
    y_te = np.zeros((n_test_windows, len(HORIZONS)))
    for i in range(n_test_windows):
        X_te[i] = test_s[i:i + SEQ_LEN]
        for hi, h in enumerate(HORIZONS):
            y_te[i, hi] = test_s[i + SEQ_LEN + h - 1]

    n_use = min(500, len(X_te), len(X_tr))
    X_te, y_te = X_te[:n_use], y_te[:n_use]
    X_tr, y_tr = X_tr[:n_use], y_tr[:n_use]

    persist_preds = make_persistence_forecast(X_te, HORIZONS)

    models_lorenz = [
        ("Persistence", lambda: ("base", make_persistence_forecast(X_te, HORIZONS))),
        ("Ridge(alpha=1)", lambda: ("ml", train_and_predict_sk_model(
            RidgeModel, X_tr, y_tr, X_te, {"alpha": 1.0}))),
    ]

    esn_params_list = [
        ("ESN500(sr=0.9)", ESN500Model, {"spectral_radius": 0.9, "leaky_rate": 0.3}),
        ("ESN500(sr=1.2)", ESN500Model, {"spectral_radius": 1.2, "leaky_rate": 0.3}),
        ("ESN1000(sr=0.9)", ESN1000Model, {"spectral_radius": 0.9, "leaky_rate": 0.3}),
    ]

    results_lorenz = {}

    for name, fn in models_lorenz:
        try:
            _, preds = fn()
            preds_inv = preds
            y_inv = y_te
            persist_inv = persist_preds
            eval_r = mc.evaluate_horizons(y_inv, preds_inv, persist_inv,
                                           horizons=HORIZONS, label=name,
                                           vpt_threshold=VPT_THRESHOLD)
            results_lorenz[name] = eval_r
            ph = Lorenz96.compute_prediction_horizon(y_inv[:, 0], preds_inv[:, 0])
            dlog.get_logger().info(f"  {name:20s} | VPT={eval_r['vpt']:2d} | FSDH={eval_r['fsdh']:2d} | PredHoriz={ph}")
        except Exception as e:
            dlog.get_logger().error(f"  {name} FAILED: {e}")

    for name, cls, params in esn_params_list:
        try:
            m = cls(**params)
            m.fit(train_s, n_lags=SEQ_LEN)
            preds = recursive_multi_horizon_predict(
                m, len(X_te), SEQ_LEN, HORIZONS
            )
            eval_r = mc.evaluate_horizons(y_te, preds, persist_preds,
                                           horizons=HORIZONS, label=name,
                                           vpt_threshold=VPT_THRESHOLD)
            results_lorenz[name] = eval_r
            ph = Lorenz96.compute_prediction_horizon(y_te[:, 0], preds[:, 0])
            dlog.get_logger().info(f"  {name:20s} | VPT={eval_r['vpt']:2d} | FSDH={eval_r['fsdh']:2d} | PredHoriz={ph}")
        except Exception as e:
            dlog.get_logger().error(f"  {name} FAILED: {e}")

    rows = []
    for name, r in results_lorenz.items():
        rows.append({"Model": name, "VPT": r["vpt"], "FSDH": r["fsdh"]})
    if rows:
        df_lorenz = pd.DataFrame(rows).set_index("Model")
        df_lorenz.to_csv(RUN_DIR / "lorenz96_results.csv")
        dlog.get_logger().info(f"\nLorenz96 Results:\n{df_lorenz.to_string()}\n")

    results_lorenz["lyapunov_exponent"] = lyap
    results_lorenz["n_vars"] = 40
    results_lorenz["forcing"] = 8.0

    with open(RUN_DIR / "lorenz96_detailed.json", "w") as f:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)): return int(obj)
                if isinstance(obj, (np.floating,)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super().default(obj)
        json.dump(results_lorenz, f, indent=2, cls=NpEncoder)

    return results_lorenz


# =====================================================================
# PART 3: ESP VERIFICATION FOR ESN MODELS
# =====================================================================

@timer()
def run_esp_verification():
    dlog.log_section("PART 3: ESP Verification for ESN Reservoirs")

    esn_configs = [
        ("ESN500_sr09", 500, 0.9, 0.3),
        ("ESN500_sr12", 500, 1.2, 0.3),
        ("ESN1000_sr09", 1000, 0.9, 0.3),
        ("ESN500_sr08_leaky05", 500, 0.8, 0.5),
    ]

    esp = ESPVerification()
    all_esp_results = {}

    for name, size, sr, leaky in esn_configs:
        dlog.log_subsection(f"ESN: {name}")
        from zaki_time_series_lib.models.deep_learning.esn import _ESNCore

        reservoir = _ESNCore(reservoir_size=size, spectral_radius=sr,
                              input_scaling=0.5, sparsity=0.1,
                              leaky_rate=leaky, seed=RANDOM_SEED)
        reservoir._initialize_weights(input_dim=SEQ_LEN)

        try:
            res = esp.verify(reservoir.W_in, reservoir.W_res,
                             n_inputs=500, n_traj=15, n_steps=80)
            all_esp_results[name] = {
                "spectral_radius": res.get("spectral_radius"),
                "spectral_radius_pass": res.get("spectral_radius_pass"),
                "conditional_lyapunov": res.get("conditional_lyapunov"),
                "conditional_lyapunov_pass": res.get("conditional_lyapunov_pass"),
                "state_forgetting_converged": res.get("state_forgetting", {}).get("converged"),
                "multi_trajectory_converged": res.get("multi_trajectory", {}).get("converged"),
                "esp_passed": res.get("esp_passed"),
            }
            dlog.get_logger().info(
                f"  {name:20s} | SR={res.get('spectral_radius',0):.4f} "
                f"| Lyap={res.get('conditional_lyapunov',0):.4f} "
                f"| ESP={'PASS' if res.get('esp_passed') else 'FAIL'}"
            )
        except Exception as e:
            dlog.get_logger().error(f"  {name} FAILED: {e}")

    df_esp = pd.DataFrame(all_esp_results).T
    df_esp.to_csv(RUN_DIR / "esp_verification.csv")
    dlog.get_logger().info(f"\nESP Results:\n{df_esp.to_string()}\n")

    return all_esp_results


# =====================================================================
# PART 4: PIPELINE QUICK DEMO
# =====================================================================

@timer()
def run_pipeline_demo():
    dlog.log_section("PART 4: Pipeline Quick Demo (ETTh1)")

    pipeline = TimeSeriesPipeline(output_dir=str(RUN_DIR / "pipeline_demo"))

    results = pipeline.run(
        dataset_name="ETTh1",
        model_configs=[
            {"name": "Persistence", "params": {}},
            {"name": "SeasonalNaive", "params": {"season_period": 24}},
            {"name": "ARIMA", "params": {"order": (2, 1, 2)}},
            {"name": "ESN500", "params": {"spectral_radius": 0.9}},
        ],
        scaler="standard",
        run_cv=True,
        cv_splits=3,
        cv_test_size=24,
    )
    pipeline.print_summary()
    return results


# =====================================================================
# GSOD-SPECIFIC HELPERS (notebook matching)
# =====================================================================

def compute_doy_normals(df_train: pd.DataFrame, target_col: str) -> pd.Series:
    r"""Day-of-year, hour-of-day normals from training data only."""
    df = df_train.copy()
    df["doy"] = df.index.dayofyear
    df["hour"] = df.index.hour
    normals = df.groupby(["doy", "hour"])[target_col].mean().rename("normal")
    df["month"] = df.index.month
    month_norm = df.groupby("month")[target_col].mean()
    for m in range(1, 13):
        m_mean = month_norm.get(m, df[target_col].mean())
        for h in range(24):
            for doy in range(1, 367):
                if (doy, h) not in normals.index or pd.isna(normals.loc[(doy, h)]):
                    normals.loc[(doy, h)] = m_mean
    return normals


def apply_doy_anomaly(df: pd.DataFrame, normals: pd.Series,
                       target_col: str) -> pd.Series:
    idx = pd.MultiIndex.from_arrays([df.index.dayofyear, df.index.hour])
    return df[target_col] - normals.reindex(idx).values


# =====================================================================
# PART 5: GSOD KORD BENCHMARK (matches qrc-complete.ipynb)
# =====================================================================

@timer()
def run_gsod_benchmark():
    dlog.log_section("PART 5: GSOD KORD Benchmark (Notebook Match)")

    GSOD_HORIZONS = [1, 6]
    GSOD_WINDOW = 24
    GSOD_TARGET = "T_db_C"

    datasets_to_run = ["GSOD_KORD"]
    models_to_run = [
        ("Persistence", None, "stat"),
        ("SeasonalNaive", None, "stat"),
        ("ARIMA(2,1,2)", {"order": (2, 1, 2)}, "stat"),
        ("Ridge", {"alpha": 1.0}, "ml"),
        ("ESN500", {"spectral_radius": 0.9}, "esn"),
    ]

    all_results = {}
    mc = MetricsCalculator()

    for ds_name in datasets_to_run:
        dlog.log_section(f"Dataset: {ds_name}")

        loader_cls = DATASET_REGISTRY[ds_name]
        loader = loader_cls(years=[2019, 2020, 2021, 2022, 2023, 2024])
        df = loader.load()

        target = GSOD_TARGET
        if target not in df.columns:
            dlog.get_logger().warning(f"Target {target} not found, using first col")
            target = df.columns[0]

        series = df[target].values.astype(np.float64)

        # Strict temporal split per notebook: 2019-2022 train, 2023 val, 2024 test
        df_train = df[df.index.year.isin([2019, 2020, 2021, 2022])].copy()
        df_val = df[df.index.year == 2023].copy()
        df_test = df[df.index.year == 2024].copy()

        dlog.get_logger().info(
            f"Split: Train {len(df_train)} | Val {len(df_val)} | Test {len(df_test)}"
        )

        # DOY climatological normals (from training only)
        normals = compute_doy_normals(df_train, target)

        train_anom = apply_doy_anomaly(df_train, normals, target)
        val_anom = apply_doy_anomaly(df_val, normals, target)
        test_anom = apply_doy_anomaly(df_test, normals, target)

        # Forward/backward fill NaN (from missing obs)
        train_anom = train_anom.ffill().bfill()
        val_anom = val_anom.ffill().bfill()
        test_anom = test_anom.ffill().bfill()

        dlog.get_logger().info(
            f"Raw std={df_train[target].std():.3f}, Anomaly std={train_anom.std():.3f}"
        )

        scaler = StandardScaler()
        train_s = scaler.fit_transform(train_anom.values.reshape(-1, 1)).flatten()
        val_s = scaler.transform(val_anom.values.reshape(-1, 1)).flatten()
        test_s = scaler.transform(test_anom.values.reshape(-1, 1)).flatten()

        # Build windows
        X_tr, y_tr = build_multi_horizon_data(train_s, GSOD_WINDOW, GSOD_HORIZONS)
        X_va, y_va = build_multi_horizon_data(val_s, GSOD_WINDOW, GSOD_HORIZONS)
        X_te_full, y_te_full = build_multi_horizon_data(test_s, GSOD_WINDOW, GSOD_HORIZONS)

        n_test = min(336, len(X_te_full))
        X_test = X_te_full[:n_test]
        y_test = y_te_full[:n_test]

        persist_preds = make_persistence_forecast(X_test, GSOD_HORIZONS)
        seasonal_preds = make_seasonal_naive_forecast(
            X_test, season=24, horizons=GSOD_HORIZONS
        )

        dlog.get_logger().info(
            f"Windows: Train {X_tr.shape} | Val {X_va.shape} | Test {X_test.shape}"
        )

        model_results = {}

        for model_name, params, model_type in models_to_run:
            dlog.log_subsection(f"Model: {model_name}")
            try:
                start_t = time.time()

                if model_type == "stat":
                    if model_name == "Persistence":
                        preds = make_persistence_forecast(X_test, GSOD_HORIZONS)
                    elif model_name == "SeasonalNaive":
                        preds = make_seasonal_naive_forecast(
                            X_test, season=24, horizons=GSOD_HORIZONS
                        )
                    elif "ARIMA" in model_name:
                        m = ARIMAModel(**params) if params else ARIMAModel()
                        m.fit(train_s)
                        preds = recursive_multi_horizon_predict(
                            m, len(X_test), GSOD_WINDOW, GSOD_HORIZONS
                        )
                    else:
                        continue

                elif model_type == "ml":
                    from sklearn.linear_model import Ridge
                    if "Ridge" in model_name:
                        preds = train_and_predict_sk_model(
                            Ridge, X_tr, y_tr, X_test,
                            {"alpha": params["alpha"]}
                        )
                    else:
                        continue

                elif model_type == "esn":
                    m = ESN500Model(**params)
                    m.fit(train_s, n_lags=GSOD_WINDOW)
                    preds = recursive_multi_horizon_predict(
                        m, len(X_test), GSOD_WINDOW, GSOD_HORIZONS
                    )

                elapsed = time.time() - start_t

                # Inverse transform from scaled anomalies
                preds_inv = np.zeros_like(preds)
                for hi in range(preds.shape[1]):
                    preds_inv[:, hi] = scaler.inverse_transform(
                        preds[:, hi].reshape(-1, 1)
                    ).flatten()
                y_test_inv = np.zeros_like(y_test)
                for hi in range(y_test.shape[1]):
                    y_test_inv[:, hi] = scaler.inverse_transform(
                        y_test[:, hi].reshape(-1, 1)
                    ).flatten()
                persist_inv = np.zeros_like(persist_preds)
                for hi in range(persist_preds.shape[1]):
                    persist_inv[:, hi] = scaler.inverse_transform(
                        persist_preds[:, hi].reshape(-1, 1)
                    ).flatten()

                eval_results = mc.evaluate_horizons(
                    y_test_inv, preds_inv, persist_inv,
                    horizons=GSOD_HORIZONS, label=model_name,
                    vpt_threshold=VPT_THRESHOLD
                )
                eval_results["training_time"] = elapsed
                model_results[model_name] = eval_results

                dlog.get_logger().info(
                    f"  {model_name:20s} | VPT={eval_results['vpt']:2d} | "
                    f"FSDH={eval_results['fsdh']:2d} | Time={elapsed:.1f}s"
                )

            except Exception as e:
                dlog.get_logger().error(f"  {model_name} FAILED: {e}")
                import traceback
                dlog.get_logger().error(traceback.format_exc())
                continue

        all_results[ds_name] = model_results

        summary_rows = []
        for mname, mres in model_results.items():
            row = {"Model": mname, "VPT": mres["vpt"], "FSDH": mres["fsdh"]}
            for hkey, hm in mres["per_horizon"].items():
                row[f"{hkey}_RMSE"] = hm["rmse"]
                row[f"{hkey}_NRMSE"] = hm["nrmse"]
                if "skill" in hm:
                    row[f"{hkey}_Skill"] = hm["skill"]
            summary_rows.append(row)

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows).set_index("Model")
            csv_path = RUN_DIR / f"GSOD_KORD_summary.csv"
            df_summary.to_csv(csv_path)
            dlog.get_logger().info(f"\n{df_summary.to_string()}\n")

    return all_results


# =====================================================================
# MAIN
# =====================================================================

def main():
    dlog.log_section("=" * 20 + " FULL EXPERIMENTATION SUITE " + "=" * 20)
    dlog.get_logger().info(f"Output: {RUN_DIR}")
    dlog.get_logger().info(f"Horizons: {HORIZONS}")
    dlog.get_logger().info(f"VPT Threshold: {VPT_THRESHOLD}")
    dlog.get_logger().info(f"Sequence Length: {SEQ_LEN}")

    all_data = {}

    with TimerContext("Part 1: Dataset Benchmarks", dlog.get_logger()):
        ds_results = run_dataset_benchmarks()
        all_data["dataset_benchmarks"] = str(RUN_DIR)
        dlog.get_logger().info(f"  -> Results in {RUN_DIR}/*_summary.csv")

    with TimerContext("Part 2: Lorenz96 Chaotic Benchmark", dlog.get_logger()):
        lorenz_results = run_lorenz96_benchmark()
        all_data["lorenz96"] = {
            "lyapunov": lorenz_results.get("lyapunov_exponent", None),
        }

    with TimerContext("Part 3: ESP Verification", dlog.get_logger()):
        esp_results = run_esp_verification()
        all_data["esp_verification"] = esp_results

    with TimerContext("Part 4: Pipeline Demo", dlog.get_logger()):
        pipeline_results = run_pipeline_demo()
        all_data["pipeline_demo"] = True

    with TimerContext("Part 5: GSOD KORD Benchmark", dlog.get_logger()):
        gsod_results = run_gsod_benchmark()
        all_data["gsod_benchmark"] = str(RUN_DIR / "GSOD_KORD_summary.csv")

    with open(RUN_DIR / "experiment_summary.json", "w") as f:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)): return int(obj)
                if isinstance(obj, (np.floating,)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super().default(obj)
        json.dump(all_data, f, indent=2, cls=NpEncoder)

    dlog.log_section("=" * 20 + " EXPERIMENT COMPLETE " + "=" * 20)
    dlog.get_logger().info(f"All results saved to: {RUN_DIR}")
    dlog.get_logger().info("Summary files:")
    dlog.get_logger().info(f"  - {RUN_DIR}/<dataset>_summary.csv (per-dataset benchmarks)")
    dlog.get_logger().info(f"  - {RUN_DIR}/lorenz96_results.csv (Lorenz96 benchmark)")
    dlog.get_logger().info(f"  - {RUN_DIR}/lorenz96_detailed.json (full Lorenz96 metrics)")
    dlog.get_logger().info(f"  - {RUN_DIR}/esp_verification.csv (ESP checks)")
    dlog.get_logger().info(f"  - {RUN_DIR}/experiment_summary.json (all metadata)")

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE — Results in: {RUN_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
