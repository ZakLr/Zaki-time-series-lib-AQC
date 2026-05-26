r"""
===============================================================================
       ZAKI TIME SERIES LIBRARY — GSOD FULL MODEL EXPERIMENT
===============================================================================
Runs ALL available models on GSOD KORD dataset (DOY anomaly, WINDOW=24,
HORIZONS=[1,6]) with full multi-horizon evaluation, VPT, FSDH, comprehensive
metrics, and detailed exports.

Usage:
    python -m zaki_time_series_lib.experiments.gsod_full_experiment

Output:
    ./experiment_output/gsod_full_<timestamp>/
        gsod_full_results.csv
        gsod_full_results_comparison.md
        comprehensive_report.html
        per_model/*.csv
        detailed_results.json
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ["ZAKI_LOG_LEVEL"] = "INFO"

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger
from zaki_time_series_lib.utils.results_exporter import ResultsExporter

from zaki_time_series_lib.data import DATASET_REGISTRY
from zaki_time_series_lib.data.preprocessing.scalers import StandardScaler

from zaki_time_series_lib.models.statistical import (
    PersistenceModel, SeasonalNaiveModel, ARIMAModel,
    SARIMAModel, AutoARIMAModel,
    ExponentialSmoothingModel, HoltWintersModel, ThetaModel
)
from zaki_time_series_lib.models.ml import (
    LinearModel, RidgeModel, LassoModel, ElasticNetModel,
    RandomForestModel, XGBoostModel, SVRModel,
    GaussianProcessModel, KNNModel
)
from zaki_time_series_lib.models.deep_learning import (
    LSTMModel, GRUModel, CNNModel, TCNModel,
    TransformerModel, InformerModel, NBeatsModel,
    ESN500Model, ESN1000Model
)

from zaki_time_series_lib.benchmark.metrics import MetricsCalculator
from zaki_time_series_lib.benchmark.verification import Lorenz63

logger = get_logger("gsod_experiment")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path("./experiment_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / f"gsod_full_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PER_MODEL_DIR = RUN_DIR / "per_model"
PER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

GSOD_HORIZONS = [1, 6]
GSOD_WINDOW = 24
GSOD_TARGET = "T_db_C"
VPT_THRESHOLD = 0.4

dlog = DetailedLogger("gsod_experiment")


# =====================================================================
# HELPERS
# =====================================================================

def compute_doy_normals(df_train: pd.DataFrame, target_col: str) -> pd.Series:
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


def build_multi_horizon_data(series: np.ndarray, seq_len: int,
                               horizons: List[int]) -> tuple:
    max_h = max(horizons)
    n_windows = len(series) - seq_len - max_h + 1
    X = np.zeros((n_windows, seq_len))
    y = np.zeros((n_windows, len(horizons)))
    for i in range(n_windows):
        X[i] = series[i:i + seq_len]
        for hi, h in enumerate(horizons):
            y[i, hi] = series[i + seq_len + h - 1]
    return X, y


def make_persistence_forecast(X: np.ndarray,
                               horizons: List[int]) -> np.ndarray:
    last_val = X[:, -1]
    return np.column_stack([last_val] * len(horizons))


def make_seasonal_naive_forecast(X: np.ndarray,
                                  horizons: List[int],
                                  season: int = 24) -> np.ndarray:
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
    if model_params is None:
        model_params = {}
    n_h = y_train.shape[1]
    preds = np.zeros((len(X_test), n_h))
    for hi in range(n_h):
        m = model_cls(**model_params)
        m.fit(X_train, y_train[:, hi])
        preds[:, hi] = m.predict(X_test)
    return preds


def recursive_multi_horizon_predict(model, n_test_windows: int, seq_len: int,
                                      horizons: list) -> np.ndarray:
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
# MODEL CONFIGURATIONS
# =====================================================================

STAT_MODELS = [
    ("Persistence", None, "stat"),
    ("SeasonalNaive", None, "stat"),
    ("ARIMA(2,1,2)", {"order": (2, 1, 2)}, "stat"),
    ("ExpSmoothing", None, "stat"),
    ("HoltWinters", None, "stat"),
    ("Theta", None, "stat"),
]

ML_MODELS = [
    ("LinearRegression", {}, "ml"),
    ("Ridge(alpha=1)", {"alpha": 1.0}, "ml"),
    ("Lasso(alpha=0.01)", {"alpha": 0.01}, "ml"),
    ("ElasticNet(alpha=0.01)", {"alpha": 0.01, "l1_ratio": 0.5}, "ml"),
    ("RandomForest", {"n_estimators": 100, "max_depth": 10}, "ml"),
    ("XGBoost", {"n_estimators": 100}, "ml"),
    ("SVR", {"C": 1.0, "gamma": "scale"}, "ml"),
    ("GaussianProcess", {}, "ml"),
    ("KNN", {"n_neighbors": 5}, "ml"),
]

DL_MODELS = [
    ("LSTM", {"hidden_dim": 64}, "dl"),
    ("GRU", {"hidden_dim": 64}, "dl"),
    ("CNN", {"hidden_channels": 64}, "dl"),
    ("TCN", {"hidden_channels": 64}, "dl"),
    ("Transformer", {"d_model": 64}, "dl"),
    ("NBeats", {"hidden_dim": 64}, "dl"),
]

ESN_MODELS = [
    ("ESN500", {"spectral_radius": 0.9, "leaky_rate": 0.3}, "esn"),
    ("ESN1000", {"spectral_radius": 0.85, "leaky_rate": 0.3}, "esn"),
]

ALL_MODELS = STAT_MODELS + ML_MODELS + DL_MODELS + ESN_MODELS


# =====================================================================
# MAIN EXPERIMENT
# =====================================================================

def run_gsod_full_experiment():
    dlog.log_section("=" * 20 + " GSOD FULL MODEL EXPERIMENT " + "=" * 20)
    dlog.get_logger().info(f"Output: {RUN_DIR}")
    dlog.get_logger().info(f"Horizons: {GSOD_HORIZONS}")
    dlog.get_logger().info(f"Window: {GSOD_WINDOW}")
    dlog.get_logger().info(f"Total models: {len(ALL_MODELS)}")
    dlog.get_logger().info("Models:")
    for name, _, mtype in ALL_MODELS:
        dlog.get_logger().info(f"  [{mtype.upper():>4}] {name}")

    # ---- Load GSOD data ----
    dlog.log_section("Loading GSOD KORD Data")
    loader_cls = DATASET_REGISTRY["GSOD_KORD"]
    loader = loader_cls(years=[2019, 2020, 2021, 2022, 2023, 2024])
    df = loader.load()

    target = GSOD_TARGET
    if target not in df.columns:
        dlog.get_logger().warning(f"Target {target} not found, using first col")
        target = df.columns[0]

    series = df[target].values.astype(np.float64)

    # ---- Strict temporal split ----
    df_train = df[df.index.year.isin([2019, 2020, 2021, 2022])].copy()
    df_val = df[df.index.year == 2023].copy()
    df_test = df[df.index.year == 2024].copy()

    dlog.get_logger().info(
        f"Split: Train {len(df_train)} | Val {len(df_val)} | Test {len(df_test)}"
    )

    # ---- DOY climatological normals ----
    normals = compute_doy_normals(df_train, target)

    train_anom = apply_doy_anomaly(df_train, normals, target)
    val_anom = apply_doy_anomaly(df_val, normals, target)
    test_anom = apply_doy_anomaly(df_test, normals, target)

    train_anom = train_anom.ffill().bfill()
    val_anom = val_anom.ffill().bfill()
    test_anom = test_anom.ffill().bfill()

    dlog.get_logger().info(
        f"Raw std={df_train[target].std():.3f}, Anomaly std={train_anom.std():.3f}"
    )

    # ---- Scale ----
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_anom.values.reshape(-1, 1)).flatten()
    val_s = scaler.transform(val_anom.values.reshape(-1, 1)).flatten()
    test_s = scaler.transform(test_anom.values.reshape(-1, 1)).flatten()

    # ---- Build windows ----
    X_tr, y_tr = build_multi_horizon_data(train_s, GSOD_WINDOW, GSOD_HORIZONS)
    X_va, y_va = build_multi_horizon_data(val_s, GSOD_WINDOW, GSOD_HORIZONS)
    X_te_full, y_te_full = build_multi_horizon_data(
        test_s, GSOD_WINDOW, GSOD_HORIZONS
    )

    n_test = min(336, len(X_te_full))
    X_test = X_te_full[:n_test]
    y_test = y_te_full[:n_test]

    dlog.get_logger().info(
        f"Windows: Train {X_tr.shape} | Val {X_va.shape} | Test {X_test.shape}"
    )

    persist_preds = make_persistence_forecast(X_test, GSOD_HORIZONS)
    seasonal_preds = make_seasonal_naive_forecast(
        X_test, season=24, horizons=GSOD_HORIZONS
    )

    mc = MetricsCalculator()
    all_model_results = {}

    # ---- Train & evaluate each model ----
    for model_name, params, model_type in ALL_MODELS:
        dlog.log_subsection(f"Model: {model_name}")
        try:
            start_t = time.time()

            if model_type == "stat":
                preds = _run_stat_model(
                    model_name, params, train_s, X_test, scaler, y_test,
                    persist_preds
                )

            elif model_type == "ml":
                preds = _run_ml_model(
                    model_name, params, X_tr, y_tr, X_test, scaler, y_test,
                    persist_preds
                )

            elif model_type == "dl":
                preds = _run_dl_model(
                    model_name, params, train_s, X_test, scaler, y_test,
                    persist_preds
                )

            elif model_type == "esn":
                preds = _run_esn_model(
                    model_name, params, train_s, X_test, scaler, y_test,
                    persist_preds
                )
            else:
                continue

            elapsed = time.time() - start_t

            # Inverse transform predictions
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

            # Store per-model CSV
            _save_per_model(eval_results, model_name)

            all_model_results[model_name] = eval_results

            dlog.get_logger().info(
                f"  {model_name:25s} | VPT={eval_results['vpt']:2d} | "
                f"FSDH={eval_results['fsdh']:2d} | Time={elapsed:.1f}s"
            )

        except Exception as e:
            dlog.get_logger().error(f"  {model_name} FAILED: {e}")
            import traceback
            dlog.get_logger().error(traceback.format_exc())
            continue

    # ---- Collect Lorenz63 benchmark (x component forecasting) ----
    dlog.log_section("Lorenz63 Chaotic Benchmark")
    lorenz63_results = _run_lorenz63_benchmark()
    for name, r in lorenz63_results.items():
        all_model_results[name] = r

    # ---- Build summary tables ----
    _build_summaries(all_model_results)

    # ---- Save detailed JSON ----
    _save_detailed_json(all_model_results)

    dlog.log_section("=" * 10 + " COMPLETE " + "=" * 10)
    dlog.get_logger().info(f"All results saved to: {RUN_DIR}")


# =====================================================================
# MODEL RUNNERS
# =====================================================================

def _run_stat_model(model_name: str, params: dict,
                     train_s: np.ndarray, X_test: np.ndarray,
                     scaler, y_test: np.ndarray,
                     persist_preds: np.ndarray) -> np.ndarray:
    if model_name == "Persistence":
        return make_persistence_forecast(X_test, GSOD_HORIZONS)
    elif model_name == "SeasonalNaive":
        return make_seasonal_naive_forecast(
            X_test, season=24, horizons=GSOD_HORIZONS
        )
    elif "ARIMA" in model_name:
        m = ARIMAModel(**params) if params else ARIMAModel()
        m.fit(train_s)
        return recursive_multi_horizon_predict(
            m, len(X_test), GSOD_WINDOW, GSOD_HORIZONS
        )
    elif model_name == "ExpSmoothing":
        m = ExponentialSmoothingModel()
        m.fit(train_s)
        return recursive_multi_horizon_predict(
            m, len(X_test), GSOD_WINDOW, GSOD_HORIZONS
        )
    elif model_name == "HoltWinters":
        m = HoltWintersModel(seasonal_periods=24)
        m.fit(train_s)
        return recursive_multi_horizon_predict(
            m, len(X_test), GSOD_WINDOW, GSOD_HORIZONS
        )
    elif model_name == "Theta":
        m = ThetaModel()
        m.fit(train_s)
        return recursive_multi_horizon_predict(
            m, len(X_test), GSOD_WINDOW, GSOD_HORIZONS
        )
    else:
        raise ValueError(f"Unknown stat model: {model_name}")


def _run_ml_model(model_name: str, params: dict,
                   X_tr: np.ndarray, y_tr: np.ndarray,
                   X_test: np.ndarray, scaler, y_test: np.ndarray,
                   persist_preds: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import (
        Ridge, Lasso, ElasticNet, LinearRegression
    )

    cls_map = {
        "LinearRegression": LinearRegression,
        "Ridge(alpha=1)": Ridge,
        "Lasso(alpha=0.01)": Lasso,
        "ElasticNet(alpha=0.01)": ElasticNet,
    }

    if model_name in cls_map:
        sk_cls = cls_map[model_name]
        return train_and_predict_sk_model(sk_cls, X_tr, y_tr, X_test, params)
    elif model_name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        return train_and_predict_sk_model(
            RandomForestRegressor, X_tr, y_tr, X_test, params
        )
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBRegressor
            return train_and_predict_sk_model(
                XGBRegressor, X_tr, y_tr, X_test, params
            )
        except ImportError:
            dlog.get_logger().warning("XGBoost not installed, skipping")
            raise
    elif model_name == "SVR":
        from sklearn.svm import SVR
        return train_and_predict_sk_model(SVR, X_tr, y_tr, X_test, params)
    elif model_name == "GaussianProcess":
        from sklearn.gaussian_process import GaussianProcessRegressor
        return train_and_predict_sk_model(
            GaussianProcessRegressor, X_tr, y_tr, X_test, params
        )
    elif model_name == "KNN":
        from sklearn.neighbors import KNeighborsRegressor
        return train_and_predict_sk_model(
            KNeighborsRegressor, X_tr, y_tr, X_test, params
        )
    else:
        raise ValueError(f"Unknown ML model: {model_name}")


def _run_dl_model(model_name: str, params: dict,
                   train_s: np.ndarray, X_test: np.ndarray,
                   scaler, y_test: np.ndarray,
                   persist_preds: np.ndarray) -> np.ndarray:
    cls_map = {
        "LSTM": LSTMModel,
        "GRU": GRUModel,
        "CNN": CNNModel,
        "TCN": TCNModel,
        "Transformer": TransformerModel,
        "NBeats": NBeatsModel,
    }
    cls = cls_map.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown DL model: {model_name}")
    m = cls(**params)
    m.fit(train_s, seq_len=GSOD_WINDOW, max_epochs=20, val_split=0.1)
    return recursive_multi_horizon_predict(
        m, len(X_test), GSOD_WINDOW, GSOD_HORIZONS
    )


def _run_esn_model(model_name: str, params: dict,
                    train_s: np.ndarray, X_test: np.ndarray,
                    scaler, y_test: np.ndarray,
                    persist_preds: np.ndarray) -> np.ndarray:
    cls_map = {
        "ESN500": ESN500Model,
        "ESN1000": ESN1000Model,
    }
    cls = cls_map.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown ESN model: {model_name}")
    m = cls(**params)
    m.fit(train_s, n_lags=GSOD_WINDOW)
    return recursive_multi_horizon_predict(
        m, len(X_test), GSOD_WINDOW, GSOD_HORIZONS
    )


# =====================================================================
# LORENZ63 BENCHMARK
# =====================================================================

def _run_lorenz63_benchmark():
    dlog.log_section("Lorenz63: generating chaotic time series")
    mc = MetricsCalculator()

    lorenz = Lorenz63(sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.02)
    data = lorenz.generate(n_steps=6000, transient=500, seed=RANDOM_SEED)
    ts = data[:, 0]
    lyap = lorenz.lyapunov_estimate(data, sample_every=5)
    dlog.get_logger().info(f"Lorenz63 estimated Lyapunov exponent: {lyap:.4f}")

    n_train = int(len(ts) * 0.7)
    train_raw = ts[:n_train]
    test_raw = ts[n_train:]

    scaler_l = StandardScaler()
    train_s = scaler_l.fit_transform(train_raw.reshape(-1, 1)).flatten()
    test_s = scaler_l.transform(test_raw.reshape(-1, 1)).flatten()

    X_tr, y_tr = build_multi_horizon_data(train_s, GSOD_WINDOW, GSOD_HORIZONS)

    n_windows = len(test_s) - GSOD_WINDOW - max(GSOD_HORIZONS) + 1
    X_te = np.zeros((n_windows, GSOD_WINDOW))
    y_te = np.zeros((n_windows, len(GSOD_HORIZONS)))
    for i in range(n_windows):
        X_te[i] = test_s[i:i + GSOD_WINDOW]
        for hi, h in enumerate(GSOD_HORIZONS):
            y_te[i, hi] = test_s[i + GSOD_WINDOW + h - 1]

    n_use = min(500, len(X_te), len(X_tr))
    X_te, y_te = X_te[:n_use], y_te[:n_use]
    X_tr, y_tr = X_tr[:n_use], y_tr[:n_use]

    persist_l = make_persistence_forecast(X_te, GSOD_HORIZONS)

    results = {}

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    models_l = [
        ("L63_Persistence", lambda: ("base", make_persistence_forecast(X_te, GSOD_HORIZONS))),
        ("L63_Ridge", lambda: ("ml", train_and_predict_sk_model(
            Ridge, X_tr, y_tr, X_te, {"alpha": 1.0}))),
        ("L63_RandomForest", lambda: ("ml", train_and_predict_sk_model(
            RandomForestRegressor, X_tr, y_tr, X_te,
            {"n_estimators": 100, "max_depth": 10}))),
        ("L63_LSTM64", lambda: ("dl", _dl_lorenz63(LSTMModel, train_s, X_te))),
        ("L63_ESN500", lambda: ("esn", _esn_lorenz63(ESN500Model, train_s, X_te))),
    ]

    for name, fn in models_l:
        try:
            _, preds = fn()
            eval_r = mc.evaluate_horizons(
                y_te, preds, persist_l,
                horizons=GSOD_HORIZONS, label=name,
                vpt_threshold=VPT_THRESHOLD
            )
            results[name] = eval_r
            ph = Lorenz63.compute_prediction_horizon(y_te[:, 0], preds[:, 0])
            dlog.get_logger().info(
                f"  {name:25s} | VPT={eval_r['vpt']:2d} | "
                f"FSDH={eval_r['fsdh']:2d} | PredHoriz={ph}"
            )
        except Exception as e:
            dlog.get_logger().error(f"  {name} FAILED: {e}")

    rows = []
    for name, r in results.items():
        rows.append({"Model": name, "VPT": r["vpt"], "FSDH": r["fsdh"]})
    if rows:
        df_l = pd.DataFrame(rows).set_index("Model")
        df_l.to_csv(RUN_DIR / "lorenz63_results.csv")
        dlog.get_logger().info(f"\nLorenz63:\n{df_l.to_string()}\n")

    results["lyapunov_exponent"] = lyap
    results["sigma"] = 10.0
    results["rho"] = 28.0
    results["beta"] = 8.0 / 3.0

    with open(RUN_DIR / "lorenz63_detailed.json", "w") as f:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)): return int(obj)
                if isinstance(obj, (np.floating,)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super().default(obj)
        json.dump(results, f, indent=2, cls=NpEncoder)

    return results


def _dl_lorenz63(cls, train_s, X_te):
    m = cls(hidden_dim=64)
    m.fit(train_s, seq_len=GSOD_WINDOW, max_epochs=20, val_split=0.1)
    return recursive_multi_horizon_predict(
        m, len(X_te), GSOD_WINDOW, GSOD_HORIZONS
    )


def _esn_lorenz63(cls, train_s, X_te):
    m = cls(spectral_radius=0.9, leaky_rate=0.3)
    m.fit(train_s, n_lags=GSOD_WINDOW)
    return recursive_multi_horizon_predict(
        m, len(X_te), GSOD_WINDOW, GSOD_HORIZONS
    )


# =====================================================================
# OUTPUT HELPERS
# =====================================================================

def _save_per_model(eval_results: dict, model_name: str):
    rows = []
    for hkey, hm in eval_results["per_horizon"].items():
        rows.append({
            "Horizon": hkey,
            "RMSE": hm["rmse"],
            "MAE": hm.get("mae", float("nan")),
            "NRMSE": hm["nrmse"],
            "Skill": hm.get("skill", float("nan")),
        })
    df = pd.DataFrame(rows)
    safe_name = model_name.replace("(", "_").replace(")", "").replace(",", "")
    df.to_csv(PER_MODEL_DIR / f"{safe_name}.csv", index=False)


def _build_summaries(all_model_results: dict):
    dlog.log_section("Building Summary Tables")

    summary_rows = []
    for mname, mres in all_model_results.items():
        if not isinstance(mres, dict) or "per_horizon" not in mres:
            continue
        row = {
            "Model": mname,
            "VPT": mres["vpt"],
            "FSDH": mres["fsdh"],
            "TrainingTime_s": round(mres.get("training_time", float("nan")), 2),
        }
        for hkey, hm in mres["per_horizon"].items():
            row[f"{hkey}_RMSE"] = round(hm["rmse"], 6)
            row[f"{hkey}_MAE"] = round(hm.get("mae", float("nan")), 6)
            row[f"{hkey}_NRMSE"] = round(hm["nrmse"], 6)
            if "skill" in hm:
                row[f"{hkey}_Skill"] = round(hm["skill"], 6)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).set_index("Model")
    csv_path = RUN_DIR / "gsod_full_results.csv"
    df_summary.to_csv(csv_path)

    dlog.get_logger().info(f"\nFULL RESULTS:\n{df_summary.to_string()}\n")
    dlog.get_logger().info(f"Saved: {csv_path}")

    # Markdown comparison
    md_lines = [
        "# GSOD Full Model Experiment Results\n",
        f"**Run ID**: {RUN_ID}  \n",
        f"**Window**: {GSOD_WINDOW}  \n",
        f"**Horizons**: {GSOD_HORIZONS}  \n",
        f"**Models**: {len(all_model_results)}  \n\n",
        "## Model Ranking by VPT\n\n",
        "| Model | VPT | FSDH | Time(s) |",
        "|-------|-----|------|---------|",
    ]
    sorted_models = sorted(
        [r for r in summary_rows if r["Model"] not in (
            "lyapunov_exponent", "sigma", "rho", "beta"
        )],
        key=lambda r: (-r["VPT"], -r["FSDH"], r["TrainingTime_s"])
    )
    for r in sorted_models:
        md_lines.append(
            f"| {r['Model']:25s} | {r['VPT']:3d} | {r['FSDH']:4d} | "
            f"{r['TrainingTime_s']:7.2f} |"
        )

    md_lines.append("\n## Per-Horizon Metrics\n")
    md_lines.append(
        "| Model | Horizon | RMSE | MAE | NRMSE | Skill |"
    )
    md_lines.append(
        "|-------|---------|------|-----|-------|-------|"
    )
    for r in summary_rows:
        if r["Model"] in ("lyapunov_exponent", "sigma", "rho", "beta"):
            continue
        for hkey in [str(h) for h in GSOD_HORIZONS]:
            rmse = r.get(f"{hkey}_RMSE", float("nan"))
            mae = r.get(f"{hkey}_MAE", float("nan"))
            nrmse = r.get(f"{hkey}_NRMSE", float("nan"))
            skill = r.get(f"{hkey}_Skill", float("nan"))
            md_lines.append(
                f"| {r['Model']:25s} | {hkey:7s} | {rmse:.4f} | "
                f"{mae:.4f} | {nrmse:.4f} | {skill:.4f} |"
            )

    md_path = RUN_DIR / "gsod_full_results_comparison.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    dlog.get_logger().info(f"Saved: {md_path}")


def _save_detailed_json(all_model_results: dict):
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(RUN_DIR / "detailed_results.json", "w") as f:
        json.dump(all_model_results, f, indent=2, cls=NpEncoder)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    run_gsod_full_experiment()
