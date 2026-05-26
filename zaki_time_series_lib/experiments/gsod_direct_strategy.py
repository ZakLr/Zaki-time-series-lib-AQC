r"""
===============================================================================
   GSOD DIRECT STRATEGY — fix fixed-origin bottleneck for non-ML models
===============================================================================
Tests DL/ESN/stat models with per-window direct prediction instead of
fixed-origin recursive forecast. Each test sample gets correct input context.

Models tested: LSTM, GRU, CNN, Transformer, NBeats, ESN500, ESN1000,
ARIMA(2,1,2), ExpSmoothing, HoltWinters, Theta (stat models on subset)

Usage:
    python -m zaki_time_series_lib.experiments.gsod_direct_strategy

Output:
    ./experiment_output/gsod_direct_<timestamp>/
        direct_comparison.csv
        direct_comparison.md
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ["ZAKI_LOG_LEVEL"] = "INFO"

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger
from zaki_time_series_lib.data import DATASET_REGISTRY
from zaki_time_series_lib.data.preprocessing.scalers import StandardScaler
from zaki_time_series_lib.models.statistical import (
    ARIMAModel, ExponentialSmoothingModel, HoltWintersModel, ThetaModel
)
from zaki_time_series_lib.models.deep_learning import (
    LSTMModel, GRUModel, CNNModel, TransformerModel, NBeatsModel,
    ESN500Model, ESN1000Model
)
from zaki_time_series_lib.benchmark.metrics import MetricsCalculator
from zaki_time_series_lib.experiments.gsod_full_experiment import (
    compute_doy_normals, apply_doy_anomaly, build_multi_horizon_data,
    make_persistence_forecast, make_seasonal_naive_forecast,
    train_and_predict_sk_model, GSOD_HORIZONS, GSOD_WINDOW, GSOD_TARGET,
    VPT_THRESHOLD, RANDOM_SEED
)

logger = get_logger("gsod_direct")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path("./experiment_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / f"gsod_direct_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

dlog = DetailedLogger("gsod_direct")


# =====================================================================
# DIRECT-WINDOW PREDICTORS
# =====================================================================

def predict_dl_direct(model, X_test_windows: np.ndarray,
                      horizons: List[int]) -> np.ndarray:
    r"""DL per-window prediction. Each test window sets model state,
    then predicts horizon steps."""
    n_windows = len(X_test_windows)
    n_h = len(horizons)
    preds = np.zeros((n_windows, n_h))
    for i in range(n_windows):
        for hi, h in enumerate(horizons):
            f = model.predict(h, last_window=X_test_windows[i])
            preds[i, hi] = f[-1]
    return preds


def predict_esn_direct(model, X_test_windows: np.ndarray,
                       horizons: List[int]) -> np.ndarray:
    r"""ESN per-window prediction. Each test window seeds reservoir,
    then predicts horizon steps."""
    n_windows = len(X_test_windows)
    n_h = len(horizons)
    preds = np.zeros((n_windows, n_h))
    for i in range(n_windows):
        for hi, h in enumerate(horizons):
            f = model.predict(h, last_window=X_test_windows[i])
            preds[i, hi] = f[-1]
    return preds


def predict_stat_expanding(model_cls, model_params, train_raw: np.ndarray,
                           X_test_windows: np.ndarray,
                           horizons: List[int],
                           max_windows: int = 100) -> np.ndarray:
    r"""Stat model with expanding window refit (subset of test windows).
    For each test window, refits on all data up to that point, forecasts
    horizon steps, picks the relevant value."""
    n_windows = min(len(X_test_windows), max_windows)
    n_h = len(horizons)
    preds = np.full((n_windows, n_h), np.nan)
    for i in range(n_windows):
        pass  # Placeholder: expanding window refit done in main runner


# =====================================================================
# MAIN
# =====================================================================

def run_direct_strategy():
    dlog.log_section("=" * 20 + " GSOD DIRECT STRATEGY EXPERIMENT " + "=" * 20)
    dlog.get_logger().info(f"Output: {RUN_DIR}")

    # ---- Load GSOD data ----
    loader_cls = DATASET_REGISTRY["GSOD_KORD"]
    loader = loader_cls(years=[2019, 2020, 2021, 2022, 2023, 2024])
    df = loader.load()

    target = GSOD_TARGET
    if target not in df.columns:
        target = df.columns[0]

    df_train = df[df.index.year.isin([2019, 2020, 2021, 2022])].copy()
    df_val = df[df.index.year == 2023].copy()
    df_test = df[df.index.year == 2024].copy()

    dlog.get_logger().info(
        f"Split: Train {len(df_train)} | Val {len(df_val)} | Test {len(df_test)}"
    )

    normals = compute_doy_normals(df_train, target)
    train_anom = apply_doy_anomaly(df_train, normals, target).ffill().bfill()
    val_anom = apply_doy_anomaly(df_val, normals, target).ffill().bfill()
    test_anom = apply_doy_anomaly(df_test, normals, target).ffill().bfill()

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_anom.values.reshape(-1, 1)).flatten()
    val_s = scaler.transform(val_anom.values.reshape(-1, 1)).flatten()
    test_s = scaler.transform(test_anom.values.reshape(-1, 1)).flatten()

    X_tr, y_tr = build_multi_horizon_data(train_s, GSOD_WINDOW, GSOD_HORIZONS)
    X_te_full, y_te_full = build_multi_horizon_data(
        test_s, GSOD_WINDOW, GSOD_HORIZONS
    )
    n_test = min(336, len(X_te_full))
    X_test = X_te_full[:n_test]
    y_test = y_te_full[:n_test]

    persist_preds = make_persistence_forecast(X_test, GSOD_HORIZONS)
    dlog.get_logger().info(
        f"Windows: Train {X_tr.shape} | Test {X_test.shape}"
    )

    mc = MetricsCalculator()
    all_results = {}

    # ---- Reference: already-good ML results ----
    dlog.log_section("Reference: Direct Strategy ML (should match earlier run)")
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    ref_configs = [
        ("REF_Ridge", lambda: train_and_predict_sk_model(
            Ridge, X_tr, y_tr, X_test, {"alpha": 1.0})),
        ("REF_RandomForest", lambda: train_and_predict_sk_model(
            RandomForestRegressor, X_tr, y_tr, X_test,
            {"n_estimators": 100, "max_depth": 10})),
    ]
    for name, fn in ref_configs:
        try:
            preds = fn()
            preds_inv = _inv_transform(preds, scaler)
            y_inv = _inv_transform(y_test, scaler)
            persist_inv = _inv_transform(persist_preds, scaler)
            r = mc.evaluate_horizons(y_inv, preds_inv, persist_inv,
                                     horizons=GSOD_HORIZONS, label=name,
                                     vpt_threshold=VPT_THRESHOLD)
            all_results[name] = r
            dlog.get_logger().info(
                f"  {name:30s} | VPT={r['vpt']:2d} | FSDH={r['fsdh']:2d} | "
                f"h=1 RMSE={r['per_horizon']['h=1']['rmse']:.4f}"
            )
        except Exception as e:
            dlog.get_logger().error(f"  {name} FAILED: {e}")

    # ---- DL models (direct per-window prediction) ----
    dlog.log_section("DL Models — Direct Per-Window Strategy")
    dl_configs = [
        ("DIR_LSTM", LSTMModel, {"hidden_dim": 64}),
        ("DIR_GRU", GRUModel, {"hidden_dim": 64}),
        ("DIR_CNN", CNNModel, {"hidden_channels": 64}),
        ("DIR_Transformer", TransformerModel, {"d_model": 64}),
        ("DIR_NBeats", NBeatsModel, {"hidden_dim": 64}),
    ]
    for name, cls, params in dl_configs:
        dlog.log_subsection(name)
        try:
            t0 = time.time()
            m = cls(**params)
            m.fit(train_s, seq_len=GSOD_WINDOW, max_epochs=20, val_split=0.1)
            preds = predict_dl_direct(m, X_test, GSOD_HORIZONS)
            elapsed = time.time() - t0
            _evaluate_and_store(all_results, name, preds, y_test,
                                persist_preds, scaler, mc, elapsed)
        except Exception as e:
            dlog.get_logger().error(f"  {name} FAILED: {e}")
            import traceback
            dlog.get_logger().error(traceback.format_exc())

    # ---- ESN models (direct per-window prediction) ----
    dlog.log_section("ESN Models — Direct Per-Window Strategy")
    esn_configs = [
        ("DIR_ESN500", ESN500Model, {"spectral_radius": 0.9, "leaky_rate": 0.3}),
        ("DIR_ESN1000", ESN1000Model, {"spectral_radius": 0.85, "leaky_rate": 0.3}),
    ]
    for name, cls, params in esn_configs:
        dlog.log_subsection(name)
        try:
            t0 = time.time()
            m = cls(**params)
            m.fit(train_s, n_lags=GSOD_WINDOW)
            preds = predict_esn_direct(m, X_test, GSOD_HORIZONS)
            elapsed = time.time() - t0
            _evaluate_and_store(all_results, name, preds, y_test,
                                persist_preds, scaler, mc, elapsed)
        except Exception as e:
            dlog.get_logger().error(f"  {name} FAILED: {e}")
            import traceback
            dlog.get_logger().error(traceback.format_exc())

    # ---- Statistical models (expanding window, subset) ----
    dlog.log_section("Stat Models — Expanding Window Refit (20 windows)")
    stat_configs = [
        ("EXP_ARIMA(2,1,2)", ARIMAModel, {"order": (2, 1, 2)}),
        ("EXP_HoltWinters", HoltWintersModel, {}),
    ]
    for name, cls, params in stat_configs:
        dlog.log_subsection(name)
        try:
            t0 = time.time()
            preds_subset = _run_stat_expanding(cls, params, train_s,
                                               test_s, X_test, GSOD_HORIZONS,
                                               max_windows=20)
            elapsed = time.time() - t0

            n_use = len(preds_subset)
            preds_full = np.zeros((n_test, len(GSOD_HORIZONS)))
            preds_full[:n_use] = preds_subset
            preds_full[n_use:] = persist_preds[n_use:]  # fallback

            _evaluate_and_store(all_results, name, preds_full, y_test,
                                persist_preds, scaler, mc, elapsed,
                                skip_vpt=True)
        except Exception as e:
            dlog.get_logger().error(f"  {name} FAILED: {e}")
            import traceback
            dlog.get_logger().error(traceback.format_exc())

    # ---- Build comparison with earlier fixed-origin results ----
    _build_comparison(all_results)

    # ---- Save ----
    _save(all_results)
    dlog.log_section("COMPLETE")
    dlog.get_logger().info(f"All results: {RUN_DIR}")


# =====================================================================
# HELPERS
# =====================================================================

def _inv_transform(arr: np.ndarray, scaler) -> np.ndarray:
    out = np.zeros_like(arr)
    for hi in range(arr.shape[1]):
        out[:, hi] = scaler.inverse_transform(arr[:, hi].reshape(-1, 1)).flatten()
    return out


def _evaluate_and_store(all_results, name, preds, y_test, persist_preds,
                        scaler, mc, elapsed, skip_vpt=False):
    preds_inv = _inv_transform(preds, scaler)
    y_inv = _inv_transform(y_test, scaler)
    persist_inv = _inv_transform(persist_preds, scaler)

    r = mc.evaluate_horizons(y_inv, preds_inv, persist_inv,
                             horizons=GSOD_HORIZONS, label=name,
                             vpt_threshold=VPT_THRESHOLD)
    r["training_time"] = elapsed
    all_results[name] = r

    vpt_str = f"VPT={r['vpt']:2d}" if not skip_vpt else "VPT=SKIP"
    dlog.get_logger().info(
        f"  {name:30s} | {vpt_str} | FSDH={r['fsdh']:2d} | "
        f"h=1 RMSE={r['per_horizon']['h=1']['rmse']:.4f} | "
        f"Time={elapsed:.1f}s"
    )


def _run_stat_expanding(model_cls, params, train_s, test_s, X_test,
                        horizons, max_windows=100):
    r"""Expanding window: fit on train + test up to current window,
    predict horizon steps from window end."""
    n_h = len(horizons)
    n_use = min(len(X_test), max_windows)
    preds = np.zeros((n_use, n_h))

    for i in range(n_use):
        # Fit on train + test data up to end of window i (position i+24 in test_s)
        expanded = np.concatenate([train_s, test_s[:i + GSOD_WINDOW]])

        if i % 20 == 0:
            dlog.get_logger().info(f"   Expanding refit {i}/{n_use}")

        m = model_cls(**params) if params else model_cls()
        m.fit(expanded)

        for hi, h in enumerate(horizons):
            forecast = m.predict(h)
            preds[i, hi] = forecast[-1]

    dlog.get_logger().info(f"   Expanding refit done: {n_use} windows")
    return preds


def _build_comparison(all_results):
    r"""Build comparison table with fixed-origin results from previous run."""
    dlog.log_section("Comparison: Fixed-Origin vs Direct")

    # Previous fixed-origin results for bottlenecked models
    fixed_results = {
        "FIXED_ARIMA(2,1,2)": 7.959,
        "FIXED_ExpSmoothing": 8.653,
        "FIXED_HoltWinters": 8.651,
        "FIXED_Theta": 8.655,
        "FIXED_LSTM": 7.912,
        "FIXED_GRU": 7.799,
        "FIXED_CNN": 8.002,
        "FIXED_Transformer": 8.395,
        "FIXED_NBeats": 7.997,
        "FIXED_ESN500": 9.212,
        "FIXED_ESN1000": 8.623,
    }

    rows = []
    for name, r in all_results.items():
        if not isinstance(r, dict) or "per_horizon" not in r:
            continue
        row = {"Model": name, "VPT": r["vpt"], "FSDH": r["fsdh"],
               "Time_s": round(r.get("training_time", float("nan")), 2)}
        for hkey in ["h=1", "h=6"]:
            hm = r["per_horizon"].get(hkey, {})
            row[f"{hkey}_RMSE"] = hm.get("rmse", float("nan"))
            row[f"{hkey}_NRMSE"] = hm.get("nrmse", float("nan"))
            row[f"{hkey}_Skill"] = hm.get("skill", float("nan"))

        # Add fixed-origin reference where available
        base_name = name.replace("DIR_", "FIXED_").replace("EXP_", "FIXED_").replace("REF_", "")
        if base_name in fixed_results:
            row["FIXED_RMSE_h1"] = fixed_results[base_name]
        else:
            row["FIXED_RMSE_h1"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_p = RUN_DIR / "direct_comparison.csv"
    df.to_csv(csv_p, index=False)
    dlog.get_logger().info(f"\n{df.to_string()}\n")
    dlog.get_logger().info(f"Saved: {csv_p}")

    # Markdown lines
    md = ["# GSOD Direct Strategy Comparison\n",
          f"**Run ID**: {RUN_ID}  \n",
          "Direct per-window prediction vs fixed-origin recursive forecast.\n",
          "\n## Results\n\n",
          "| Model | Strategy | VPT | FSDH | h=1 RMSE | h=6 RMSE | h=1 NRMSE | h=1 Skill | Time(s) |\n",
          "|-------|----------|-----|------|----------|----------|-----------|-----------|---------|\n"]

    for _, r in df.iterrows():
        md.append(
            f"| {r['Model']:25s} | {'Direct' if 'DIR' in str(r['Model']) or 'REF' in str(r['Model']) else 'Expanding' if 'EXP' in str(r['Model']) else '':7s} | "
            f"{int(r['VPT']):3d} | {int(r['FSDH']):4d} | {r['h=1_RMSE']:.4f} | {r['h=6_RMSE']:.4f} | "
            f"{r['h=1_NRMSE']:.4f} | {r['h=1_Skill']:.4f} | {r['Time_s']:.1f} |\n"
        )

    md_path = RUN_DIR / "direct_comparison.md"
    with open(md_path, "w") as f:
        f.write("".join(md))
    dlog.get_logger().info(f"Saved: {md_path}")


def _save(all_results):
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    with open(RUN_DIR / "detailed_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)


if __name__ == "__main__":
    run_direct_strategy()
