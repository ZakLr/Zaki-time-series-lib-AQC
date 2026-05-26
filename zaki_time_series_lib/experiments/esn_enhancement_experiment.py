r"""
ESN Enhancement Comparison
==========================
Compares 5 model variants on GSOD KORD temp anomaly (h=1, h=6):

  1. Persistence        — last-value baseline
  2. REF_Ridge          — sklearn Ridge direct on flat windows
  3. ESN Baseline       — best single ESN from tuning (rad=0.5, leak=1.0, inp=0.1, ridge=1.0)
  4. Ensemble ESN       — average of 10 ESNs with different seeds
  5. Two-Stage ESN      — Ridge → ESN on residuals hybrid

Output:
  ./experiment_output/esn_enhancement_<timestamp>/
      results.csv
      report.md
      detailed_results.json
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["ZAKI_LOG_LEVEL"] = "WARNING"

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger
from zaki_time_series_lib.data import DATASET_REGISTRY
from zaki_time_series_lib.data.preprocessing.scalers import StandardScaler
from zaki_time_series_lib.models.deep_learning.esn import _ESNCore, ESNModel
from zaki_time_series_lib.benchmark.metrics import MetricsCalculator

logger = get_logger("esn_enhancement")

OUTPUT_DIR = Path("./experiment_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / f"esn_enhancement_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

dlog = DetailedLogger("esn_enhancement")

N_LAGS = 24
HORIZONS = [1, 6]
VPT_THRESHOLD = 0.4
RANDOM_SEED = 42
N_WARMUP = 100
N_TEST_WINDOWS = 336
N_ENSEMBLE = 10

ESN_PARAMS = dict(
    reservoir_size=500, spectral_radius=0.5, leaky_rate=1.0,
    input_scaling=0.1, alpha_ridge=1.0, n_warmup=N_WARMUP, seed=RANDOM_SEED
)


# =====================================================================
# DATA HELPERS
# =====================================================================

def compute_doy_normals(df_train, target_col):
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

def apply_doy_anomaly(df, normals, target_col):
    idx = pd.MultiIndex.from_arrays([df.index.dayofyear, df.index.hour])
    return df[target_col] - normals.reindex(idx).values

def build_multi_horizon_data(series, seq_len, horizons):
    max_h = max(horizons)
    n_windows = len(series) - seq_len - max_h + 1
    X = np.zeros((n_windows, seq_len))
    y = np.zeros((n_windows, len(horizons)))
    for i in range(n_windows):
        X[i] = series[i:i + seq_len]
        for hi, h in enumerate(horizons):
            y[i, hi] = series[i + seq_len + h - 1]
    return X, y

def make_persistence_forecast(X, horizons):
    last_val = X[:, -1]
    return np.column_stack([last_val] * len(horizons))

def predict_dl_direct(model, X_windows, horizons):
    n_w = len(X_windows)
    n_h = len(horizons)
    preds = np.zeros((n_w, n_h))
    for i in range(n_w):
        for hi, h in enumerate(horizons):
            f = model.predict(h, last_window=X_windows[i])
            preds[i, hi] = f[-1]
    return preds

def inv_transform(arr, scaler):
    out = np.zeros_like(arr)
    for hi in range(arr.shape[1]):
        out[:, hi] = scaler.inverse_transform(arr[:, hi].reshape(-1, 1)).flatten()
    return out

def train_and_predict_ridge(X_train, y_train, X_test, alpha=1.0):
    preds = np.zeros((X_test.shape[0], y_train.shape[1]))
    for hi in range(y_train.shape[1]):
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(X_train, y_train[:, hi])
        preds[:, hi] = m.predict(X_test)
    return preds


# =====================================================================
# ENHANCED MODELS
# =====================================================================

class EnsembleESNModel(ESNModel):
    def __init__(self, n_ensemble=N_ENSEMBLE, **kwargs):
        self.n_ensemble = n_ensemble
        self._seed = kwargs.pop('seed', RANDOM_SEED)
        super().__init__(seed=self._seed, **kwargs)
        self._models = []
        self.name = f"EnsembleESN{n_ensemble}"

    def fit(self, y, X=None, **kwargs):
        self._models = []
        for i in range(self.n_ensemble):
            m = ESNModel(
                reservoir_size=self.reservoir_size,
                spectral_radius=self.spectral_radius,
                input_scaling=self.input_scaling,
                sparsity=self.sparsity,
                leaky_rate=self.leaky_rate,
                alpha_ridge=self.alpha_ridge,
                n_warmup=self.n_warmup,
                seed=self._seed + i
            )
            m.fit(y, X=X, **kwargs)
            self._models.append(m)
        self.is_fitted = True
        self._last_inputs = self._models[0]._last_inputs
        self._last_state = self._models[0]._last_state
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        preds = []
        for m in self._models:
            preds.append(m.predict(horizon, X_future=X_future, **kwargs))
        return np.mean(preds, axis=0)


class TwoStageESNModel(ESNModel):
    def __init__(self, ridge_alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.ridge_alpha = ridge_alpha
        self._ridge = None
        self.name = "TwoStageESN"

    def fit(self, y, X=None, **kwargs):
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

        self._ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self._ridge.fit(inputs, y)
        ridge_train_pred = self._ridge.predict(inputs)
        residuals = y - ridge_train_pred

        self._esn = _ESNCore(
            self.reservoir_size, self.spectral_radius, self.input_scaling,
            self.sparsity, self.leaky_rate, self.seed
        )
        self._esn._initialize_weights(self._input_dim)

        states = self._esn.compute_states(inputs, self.n_warmup)
        states_train = states[self.n_warmup:]
        residuals_train = residuals[self.n_warmup:]

        self._readout = Ridge(alpha=self.alpha_ridge, fit_intercept=True)
        self._readout.fit(states_train, residuals_train)
        self.is_fitted = True

        self._last_inputs = inputs[-self._n_lags:] if len(inputs) >= self._n_lags else inputs
        self._last_state = states[-1].copy()
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        if X_future is not None and len(X_future) >= horizon:
            inputs = np.asarray(X_future[:horizon], dtype=np.float64)
            states = self._esn.compute_states(inputs)
            esn_res = self._readout.predict(states)
            ridge_pred = self._ridge.predict(inputs)
            return esn_res + ridge_pred

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
            for _ in range(self.n_warmup):
                state = (1 - self._esn.leaky_rate) * state + self._esn.leaky_rate * np.tanh(
                    self._esn.W_in @ last_window + self._esn.W_res @ state
                )

        for i in range(horizon):
            u = last_window.reshape(1, -1)
            ridge_pred = self._ridge.predict(u)[0]
            state = (1 - self._esn.leaky_rate) * state + self._esn.leaky_rate * np.tanh(
                self._esn.W_in @ u.flatten() + self._esn.W_res @ state
            )
            esn_res = self._readout.predict(state.reshape(1, -1))[0]
            pred = ridge_pred + esn_res
            predictions.append(pred)
            last_window = np.roll(last_window, -1)
            last_window[-1] = pred

        result = np.array(predictions)
        return result


# =====================================================================
# EXPERIMENT
# =====================================================================

def run():
    dlog.log_section("Loading GSOD Data")
    loader_cls = DATASET_REGISTRY["GSOD_KORD"]
    loader = loader_cls(years=[2019, 2020, 2021, 2022, 2023, 2024])
    df = loader.load()
    target = "T_db_C"

    df_train = df[df.index.year.isin([2019, 2020, 2021, 2022])].copy()
    df_test = df[df.index.year == 2024].copy()

    normals = compute_doy_normals(df_train, target)
    train_anom = apply_doy_anomaly(df_train, normals, target).ffill().bfill()
    test_anom = apply_doy_anomaly(df_test, normals, target).ffill().bfill()

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_anom.values.reshape(-1, 1)).flatten()
    test_s = scaler.transform(test_anom.values.reshape(-1, 1)).flatten()

    X_tr, y_tr = build_multi_horizon_data(train_s, N_LAGS, HORIZONS)
    X_te_full, y_te_full = build_multi_horizon_data(test_s, N_LAGS, HORIZONS)
    n_test = min(N_TEST_WINDOWS, len(X_te_full))
    X_test = X_te_full[:n_test]
    y_test = y_te_full[:n_test]

    persist_preds = make_persistence_forecast(X_test, HORIZONS)

    dlog.get_logger().info(f"Train windows: {X_tr.shape}, Test windows: {X_test.shape}")
    dlog.get_logger().info(f"Models: Persistence, REF_Ridge, ESN, EnsembleESN{N_ENSEMBLE}, TwoStageESN")

    mc = MetricsCalculator()
    results = {}

    # --- REF_Ridge ---
    dlog.log_subsection("REF_Ridge")
    t0 = time.time()
    ridge_preds = train_and_predict_ridge(X_tr, y_tr, X_test, alpha=1.0)
    ridge_time = time.time() - t0
    ridge_preds_i = inv_transform(ridge_preds, scaler)
    y_i = inv_transform(y_test, scaler)
    p_i = inv_transform(persist_preds, scaler)
    r = mc.evaluate_horizons(y_i, ridge_preds_i, p_i, horizons=HORIZONS,
                             label="REF_Ridge", vpt_threshold=VPT_THRESHOLD)
    r["time_s"] = ridge_time
    results["REF_Ridge"] = r
    dlog.get_logger().info(f"  h=1 RMSE={r['per_horizon']['h=1']['rmse']:.4f} "
                           f"Skill={r['per_horizon']['h=1']['skill']:.4f} "
                           f"VPT={r['vpt']} [{ridge_time:.1f}s]")

    # --- ESN Baseline ---
    dlog.log_subsection("ESN Baseline")
    t0 = time.time()
    esn = ESNModel(**ESN_PARAMS)
    esn.fit(train_s, n_lags=N_LAGS)
    esn_preds = predict_dl_direct(esn, X_test, HORIZONS)
    esn_time = time.time() - t0
    esn_preds_i = inv_transform(esn_preds, scaler)
    r = mc.evaluate_horizons(y_i, esn_preds_i, p_i, horizons=HORIZONS,
                             label="ESN_Baseline", vpt_threshold=VPT_THRESHOLD)
    r["time_s"] = esn_time
    results["ESN_Baseline"] = r
    dlog.get_logger().info(f"  h=1 RMSE={r['per_horizon']['h=1']['rmse']:.4f} "
                           f"Skill={r['per_horizon']['h=1']['skill']:.4f} "
                           f"VPT={r['vpt']} [{esn_time:.1f}s]")

    # --- Ensemble ESN ---
    dlog.log_subsection(f"EnsembleESN{N_ENSEMBLE}")
    t0 = time.time()
    ensemble = EnsembleESNModel(n_ensemble=N_ENSEMBLE, **ESN_PARAMS)
    ensemble.fit(train_s, n_lags=N_LAGS)
    ens_preds = predict_dl_direct(ensemble, X_test, HORIZONS)
    ens_time = time.time() - t0
    ens_preds_i = inv_transform(ens_preds, scaler)
    r = mc.evaluate_horizons(y_i, ens_preds_i, p_i, horizons=HORIZONS,
                             label=f"EnsembleESN{N_ENSEMBLE}", vpt_threshold=VPT_THRESHOLD)
    r["time_s"] = ens_time
    results[f"EnsembleESN{N_ENSEMBLE}"] = r
    dlog.get_logger().info(f"  h=1 RMSE={r['per_horizon']['h=1']['rmse']:.4f} "
                           f"Skill={r['per_horizon']['h=1']['skill']:.4f} "
                           f"VPT={r['vpt']} [{ens_time:.1f}s]")

    # --- Two-Stage ESN ---
    dlog.log_subsection("TwoStageESN")
    t0 = time.time()
    hybrid = TwoStageESNModel(**ESN_PARAMS)
    hybrid.fit(train_s, n_lags=N_LAGS)
    hybrid_preds = predict_dl_direct(hybrid, X_test, HORIZONS)
    hybrid_time = time.time() - t0
    hybrid_preds_i = inv_transform(hybrid_preds, scaler)
    r = mc.evaluate_horizons(y_i, hybrid_preds_i, p_i, horizons=HORIZONS,
                             label="TwoStageESN", vpt_threshold=VPT_THRESHOLD)
    r["time_s"] = hybrid_time
    results["TwoStageESN"] = r
    dlog.get_logger().info(f"  h=1 RMSE={r['per_horizon']['h=1']['rmse']:.4f} "
                           f"Skill={r['per_horizon']['h=1']['skill']:.4f} "
                           f"VPT={r['vpt']} [{hybrid_time:.1f}s]")

    # Persistence metrics
    persist_inv = p_i
    r = mc.evaluate_horizons(y_i, persist_inv, persist_inv, horizons=HORIZONS,
                             label="Persistence", vpt_threshold=VPT_THRESHOLD)
    r["time_s"] = 0.0
    results["Persistence"] = r
    persist_h1_rmse = r['per_horizon']['h=1']['rmse']

    # ---- RESULTS TABLE ----
    dlog.log_section("RESULTS SUMMARY")
    rows = []
    for label, r in results.items():
        rows.append({
            "Model": label,
            "h=1 RMSE": r["per_horizon"]["h=1"]["rmse"],
            "h=1 Skill": r["per_horizon"]["h=1"]["skill"],
            "h=1 NRMSE": r["per_horizon"]["h=1"]["nrmse"],
            "h=6 RMSE": r["per_horizon"]["h=6"]["rmse"],
            "h=6 Skill": r["per_horizon"]["h=6"]["skill"],
            "h=6 NRMSE": r["per_horizon"]["h=6"]["nrmse"],
            "VPT": r["vpt"],
            "FSDH": r["fsdh"],
            "Time(s)": r["time_s"],
        })
    df_results = pd.DataFrame(rows).sort_values("h=1 RMSE")
    df_results.to_csv(RUN_DIR / "results.csv", index=False)
    dlog.get_logger().info(f"\n{df_results.to_string(index=False)}\n")

    # ---- REPORT ----
    md = [
        f"# ESN Enhancement Comparison\n",
        f"**Run ID**: {RUN_ID}  \n",
        f"**N_ENSEMBLE**: {N_ENSEMBLE}  \n\n",
        "| Model | h=1 RMSE | h=1 Skill | h=6 RMSE | h=6 Skill | VPT | FSDH | Time(s) |\n",
        "|------|---------|----------|---------|----------|-----|------|---------|\n",
    ]
    for _, row in df_results.iterrows():
        md.append(
            f"| {row['Model']} | {row['h=1 RMSE']:.4f} | {row['h=1 Skill']:.4f} | "
            f"{row['h=6 RMSE']:.4f} | {row['h=6 Skill']:.4f} | "
            f"{row['VPT']} | {row['FSDH']} | {row['Time(s)']:.1f} |\n"
        )
    md.append(f"\nPersistence h=1 RMSE: {persist_h1_rmse:.4f}\n")
    md.append(f"Best config: rad=0.5, leak=1.0, inp=0.1, ridge=1.0, warmup={N_WARMUP}\n")

    (RUN_DIR / "report.md").write_text("".join(md))

    detailed = {}
    for label, r in results.items():
        detailed[label] = {
            k: (v if not isinstance(v, dict) else {
                kk: vv for kk, vv in v.items()
            })
            for k, v in r.items()
        }
    with open(RUN_DIR / "detailed_results.json", "w") as f:
        json.dump(detailed, f, indent=2)

    dlog.log_section("COMPLETE")
    dlog.get_logger().info(f"Output: {RUN_DIR}")


if __name__ == "__main__":
    run()
