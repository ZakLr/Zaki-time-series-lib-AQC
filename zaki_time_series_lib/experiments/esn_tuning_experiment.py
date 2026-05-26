r"""
===============================================================================
   ESN TUNING EXPERIMENT — find config that beats persistence
===============================================================================
Tests ESN500 with hyperparameter grid on GSOD KORD temp anomaly + Lorenz63.
Experiments:
  1. GSOD temp anomaly — grid search over spectral_radius, leaky_rate,
     input_scaling, alpha_ridge (all with warmup fix)
  2. Lorenz63 x-component — same grid, n_lags=24

Output:
  ./experiment_output/esn_tuning_<timestamp>/
      gsod_grid.csv          — all GSOD configs ranked by h=1 RMSE
      lorenz63_grid.csv      — all L63 configs ranked by h=1 RMSE
      detailed_results.json
"""

import os
import sys
import time
import json
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ["ZAKI_LOG_LEVEL"] = "WARNING"

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger
from zaki_time_series_lib.data import DATASET_REGISTRY
from zaki_time_series_lib.data.preprocessing.scalers import StandardScaler
from zaki_time_series_lib.models.deep_learning import ESN500Model
from zaki_time_series_lib.benchmark.metrics import MetricsCalculator
from zaki_time_series_lib.benchmark.verification import Lorenz63

logger = get_logger("esn_tuning")

OUTPUT_DIR = Path("./experiment_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / f"esn_tuning_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

dlog = DetailedLogger("esn_tuning")

# constants
GSOD_WINDOW = 24
GSOD_HORIZONS = [1, 6]
VPT_THRESHOLD = 0.4
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ESN hyperparameter grid
SPECTRAL_RADII = [0.5, 0.9, 1.5]
LEAKY_RATES = [0.3, 1.0]
INPUT_SCALINGS = [0.1, 1.0]
ALPHA_RIDGES = [1e-4, 1.0]
N_WARMUP = 100
N_LAGS_GSOD = 24
N_LAGS_L63 = 24
N_LAGS_ALT = [6, 12]
N_TEST_WINDOWS = 336
L63_N_STEPS = 10000
L63_TRAIN_RATIO = 0.7


# =====================================================================
# HELPERS (borrowed from gsod_full_experiment)
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


def evaluate_models(all_results, name, preds, y_true, persist_preds, scaler, mc, elapsed):
    preds_i = inv_transform(preds, scaler)
    y_i = inv_transform(y_true, scaler)
    p_i = inv_transform(persist_preds, scaler)

    r = mc.evaluate_horizons(y_i, preds_i, p_i, horizons=GSOD_HORIZONS,
                             label=name, vpt_threshold=VPT_THRESHOLD)
    r["time_s"] = elapsed
    all_results[name] = r
    return r


# =====================================================================
# GSOD GRID
# =====================================================================

def run_gsod_grid():
    dlog.log_section("GSOD: Loading & Preprocessing")
    loader_cls = DATASET_REGISTRY["GSOD_KORD"]
    loader = loader_cls(years=[2019, 2020, 2021, 2022, 2023, 2024])
    df = loader.load()
    target = "T_db_C"

    df_train = df[df.index.year.isin([2019, 2020, 2021, 2022])].copy()
    df_val = df[df.index.year == 2023].copy()
    df_test = df[df.index.year == 2024].copy()

    normals = compute_doy_normals(df_train, target)
    train_anom = apply_doy_anomaly(df_train, normals, target).ffill().bfill()
    val_anom = apply_doy_anomaly(df_val, normals, target).ffill().bfill()
    test_anom = apply_doy_anomaly(df_test, normals, target).ffill().bfill()

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_anom.values.reshape(-1, 1)).flatten()
    test_s = scaler.transform(test_anom.values.reshape(-1, 1)).flatten()

    X_tr, y_tr = build_multi_horizon_data(train_s, GSOD_WINDOW, GSOD_HORIZONS)
    X_te_full, y_te_full = build_multi_horizon_data(test_s, GSOD_WINDOW, GSOD_HORIZONS)
    n_test = min(N_TEST_WINDOWS, len(X_te_full))
    X_test = X_te_full[:n_test]
    y_test = y_te_full[:n_test]

    persist_preds = make_persistence_forecast(X_test, GSOD_HORIZONS)
    mc = MetricsCalculator()

    dlog.get_logger().info(f"Windows: train {X_tr.shape}, test {X_test.shape}")

    # build grid
    grid = list(itertools.product(SPECTRAL_RADII, LEAKY_RATES,
                                  INPUT_SCALINGS, ALPHA_RIDGES))
    total = len(grid)
    dlog.get_logger().info(f"GSOD grid: {total} configs")

    results = {}
    for idx, (sr, lr, isc, ar) in enumerate(grid):
        name = f"ESN500_rad{sr}_leak{lr}_inp{isc}_ridge{ar}"
        dlog.log_subsection(f"[{idx+1}/{total}] {name}")
        try:
            t0 = time.time()
            m = ESN500Model(
                spectral_radius=sr, leaky_rate=lr,
                input_scaling=isc, alpha_ridge=ar,
                n_warmup=N_WARMUP, seed=RANDOM_SEED
            )
            m.fit(train_s, n_lags=N_LAGS_GSOD)
            preds = predict_dl_direct(m, X_test, GSOD_HORIZONS)
            elapsed = time.time() - t0
            r = evaluate_models(results, name, preds, y_test,
                                persist_preds, scaler, mc, elapsed)
            r["_sr"] = sr; r["_lr"] = lr; r["_isc"] = isc; r["_ar"] = ar
            dlog.get_logger().info(
                f"  h=1 RMSE={r['per_horizon']['h=1']['rmse']:.4f} "
                f"Skill={r['per_horizon']['h=1']['skill']:.4f} "
                f"VPT={r['vpt']} FSDH={r['fsdh']} [{elapsed:.1f}s]"
            )
        except Exception as e:
            dlog.get_logger().error(f"  FAILED: {e}")
            import traceback
            dlog.get_logger().error(traceback.format_exc())

    df_results = pd.DataFrame([
        {
            "name": name,
            "h1_rmse": r["per_horizon"]["h=1"]["rmse"],
            "h1_skill": r["per_horizon"]["h=1"]["skill"],
            "h1_nrmse": r["per_horizon"]["h=1"]["nrmse"],
            "h6_rmse": r["per_horizon"]["h=6"]["rmse"],
            "h6_skill": r["per_horizon"]["h=6"]["skill"],
            "vpt": r["vpt"],
            "fsdh": r["fsdh"],
            "time_s": r["time_s"],
        }
        for name, r in results.items()
    ])
    df_results = df_results.sort_values("h1_rmse")
    df_results.to_csv(RUN_DIR / "gsod_grid.csv", index=False)

    dlog.log_section("GSOD Top 5")
    dlog.get_logger().info(f"\n{df_results.head(10).to_string()}\n")

    return results


# =====================================================================
# LORENZ63 GRID
# =====================================================================

def run_l63_grid():
    dlog.log_section("Lorenz63: Generating")
    l63 = Lorenz63(sigma=10.0, rho=28.0, beta=8/3, dt=0.01)
    data = l63.generate(n_steps=L63_N_STEPS, transient=1000, seed=RANDOM_SEED)
    x = data[:, 0]

    split = int(len(x) * L63_TRAIN_RATIO)
    train_raw, test_raw = x[:split], x[split:]

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
    test_s = scaler.transform(test_raw.reshape(-1, 1)).flatten()

    X_tr, y_tr = build_multi_horizon_data(train_s, N_LAGS_L63, GSOD_HORIZONS)
    X_te_full, y_te_full = build_multi_horizon_data(test_s, N_LAGS_L63, GSOD_HORIZONS)
    n_test = min(N_TEST_WINDOWS, len(X_te_full))
    X_test = X_te_full[:n_test]
    y_test = y_te_full[:n_test]

    persist_preds = make_persistence_forecast(X_test, GSOD_HORIZONS)
    mc = MetricsCalculator()

    dlog.get_logger().info(f"Windows: train {X_tr.shape}, test {X_test.shape}")

    grid = list(itertools.product(SPECTRAL_RADII, LEAKY_RATES,
                                  INPUT_SCALINGS, ALPHA_RIDGES))
    total = len(grid)
    dlog.get_logger().info(f"L63 grid: {total} configs")

    results = {}
    for idx, (sr, lr, isc, ar) in enumerate(grid):
        name = f"ESN500_rad{sr}_leak{lr}_inp{isc}_ridge{ar}"
        dlog.log_subsection(f"[{idx+1}/{total}] {name}")
        try:
            t0 = time.time()
            m = ESN500Model(
                spectral_radius=sr, leaky_rate=lr,
                input_scaling=isc, alpha_ridge=ar,
                n_warmup=N_WARMUP, seed=RANDOM_SEED
            )
            m.fit(train_s, n_lags=N_LAGS_L63)
            preds = predict_dl_direct(m, X_test, GSOD_HORIZONS)
            elapsed = time.time() - t0
            r = evaluate_models(results, name, preds, y_test,
                                persist_preds, scaler, mc, elapsed)
            r["_sr"] = sr; r["_lr"] = lr; r["_isc"] = isc; r["_ar"] = ar
            dlog.get_logger().info(
                f"  h=1 RMSE={r['per_horizon']['h=1']['rmse']:.4f} "
                f"Skill={r['per_horizon']['h=1']['skill']:.4f} "
                f"VPT={r['vpt']} FSDH={r['fsdh']} [{elapsed:.1f}s]"
            )
        except Exception as e:
            dlog.get_logger().error(f"  FAILED: {e}")

    df_results = pd.DataFrame([
        {
            "name": name,
            "h1_rmse": r["per_horizon"]["h=1"]["rmse"],
            "h1_skill": r["per_horizon"]["h=1"]["skill"],
            "h1_nrmse": r["per_horizon"]["h=1"]["nrmse"],
            "h6_rmse": r["per_horizon"]["h=6"]["rmse"],
            "h6_skill": r["per_horizon"]["h=6"]["skill"],
            "vpt": r["vpt"],
            "fsdh": r["fsdh"],
            "time_s": r["time_s"],
        }
        for name, r in results.items()
    ])
    df_results = df_results.sort_values("h1_rmse")
    df_results.to_csv(RUN_DIR / "lorenz63_grid.csv", index=False)

    dlog.log_section("L63 Top 5")
    dlog.get_logger().info(f"\n{df_results.head(10).to_string()}\n")

    return results


# =====================================================================
# ALTERNATE N_LAGS CHECK (top 3 configs)
# =====================================================================

def run_alt_lags(gsod_results):
    dlog.log_section("Alternate n_lags: [6, 12] on GSOD")
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

    mc = MetricsCalculator()

    # Top 3 configs
    sorted_names = sorted(gsod_results.keys(),
                          key=lambda n: gsod_results[n]["per_horizon"]["h=1"]["rmse"])
    top3 = sorted_names[:3]

    for n_lags in N_LAGS_ALT:
        X_tr, y_tr = build_multi_horizon_data(train_s, n_lags, GSOD_HORIZONS)
        X_te_full, y_te_full = build_multi_horizon_data(test_s, n_lags, GSOD_HORIZONS)
        n_test = min(N_TEST_WINDOWS, len(X_te_full))
        X_test = X_te_full[:n_test]
        y_test = y_te_full[:n_test]
        persist_preds = make_persistence_forecast(X_test, GSOD_HORIZONS)

        dlog.log_subsection(f"n_lags={n_lags}")
        for name in top3:
            try:
                r = gsod_results[name]
                t0 = time.time()
                m = ESN500Model(
                    spectral_radius=r.get("_sr", 0.9),
                    leaky_rate=r.get("_lr", 0.3),
                    input_scaling=r.get("_isc", 0.5),
                    alpha_ridge=r.get("_ar", 1e-4),
                    n_warmup=N_WARMUP, seed=RANDOM_SEED
                )
                m.fit(train_s, n_lags=n_lags)
                preds = predict_dl_direct(m, X_test, GSOD_HORIZONS)
                elapsed = time.time() - t0
                r2 = evaluate_models({}, f"{name}_lag{n_lags}", preds, y_test,
                                     persist_preds, scaler, mc, elapsed)
                dlog.get_logger().info(
                    f"  {name:35s} lag={n_lags} | h=1 RMSE={r2['per_horizon']['h=1']['rmse']:.4f} "
                    f"Skill={r2['per_horizon']['h=1']['skill']:.4f} "
                    f"VPT={r2['vpt']} [{elapsed:.1f}s]"
                )
            except Exception as e:
                dlog.get_logger().error(f"  {name} FAILED with lag={n_lags}: {e}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    dlog.log_section("=" * 20 + " ESN TUNING EXPERIMENT " + "=" * 20)
    dlog.get_logger().info(f"Output: {RUN_DIR}")
    dlog.get_logger().info(f"Grid: {len(SPECTRAL_RADII)} × {len(LEAKY_RATES)} × "
                           f"{len(INPUT_SCALINGS)} × {len(ALPHA_RIDGES)} = "
                           f"{len(list(itertools.product(SPECTRAL_RADII, LEAKY_RATES, INPUT_SCALINGS, ALPHA_RIDGES)))} configs")

    # Phase A: GSOD grid
    gsod_results = run_gsod_grid()

    # Phase B: Lorenz63 grid
    l63_results = run_l63_grid()

    # Phase C: Alternate n_lags for top 3 GSOD configs
    run_alt_lags(gsod_results)

    # Save final summary
    summary = pd.DataFrame({
        "dataset": ["GSOD"] * len(gsod_results) + ["Lorenz63"] * len(l63_results),
        "name": list(gsod_results.keys()) + list(l63_results.keys()),
        "h1_rmse": [r["per_horizon"]["h=1"]["rmse"] for r in gsod_results.values()]
                   + [r["per_horizon"]["h=1"]["rmse"] for r in l63_results.values()],
        "h1_skill": [r["per_horizon"]["h=1"]["skill"] for r in gsod_results.values()]
                    + [r["per_horizon"]["h=1"]["skill"] for r in l63_results.values()],
        "vpt": [r["vpt"] for r in gsod_results.values()]
               + [r["vpt"] for r in l63_results.values()],
        "fsdh": [r["fsdh"] for r in gsod_results.values()]
                + [r["fsdh"] for r in l63_results.values()],
    })
    summary_path = RUN_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # Markdown report
    md_lines = [
        f"# ESN Tuning Experiment\n",
        f"**Run ID**: {RUN_ID}  \n\n",
        "## GSOD Results (Top 10)\n\n",
        "| Rank | Config | h=1 RMSE | Skill | VPT | FSDH | Time(s) |\n",
        "|------|--------|----------|-------|-----|------|---------|\n",
    ]
    gsod_sorted = sorted(gsod_results.items(),
                         key=lambda x: x[1]["per_horizon"]["h=1"]["rmse"])
    for rank, (name, r) in enumerate(gsod_sorted[:10], 1):
        ph = r["per_horizon"]["h=1"]
        md_lines.append(
            f"| {rank} | {name} | {ph['rmse']:.4f} | {ph['skill']:.4f} | "
            f"{r['vpt']} | {r['fsdh']} | {r['time_s']:.1f} |\n"
        )

    # Persistence reference
    gsod_persist_rmse = gsod_sorted[0][1]["per_horizon"]["h=1"].get("_persist_rmse", None)
    if gsod_persist_rmse is not None:
        md_lines.append(f"\nPersistence RMSE: {gsod_persist_rmse:.4f}\n")
    else:
        md_lines.append(f"\nUse `REF_Ridge` at ~0.593 as reference (near-persistence).\n")

    md_lines.extend([
        "\n## Lorenz63 Results (Top 10)\n\n",
        "| Rank | Config | h=1 RMSE | Skill | VPT | FSDH | Time(s) |\n",
        "|------|--------|----------|-------|-----|------|---------|\n",
    ])
    l63_sorted = sorted(l63_results.items(),
                        key=lambda x: x[1]["per_horizon"]["h=1"]["rmse"])
    for rank, (name, r) in enumerate(l63_sorted[:10], 1):
        ph = r["per_horizon"]["h=1"]
        md_lines.append(
            f"| {rank} | {name} | {ph['rmse']:.4f} | {ph['skill']:.4f} | "
            f"{r['vpt']} | {r['fsdh']} | {r['time_s']:.1f} |\n"
        )

    md_path = RUN_DIR / "report.md"
    with open(md_path, "w") as f:
        f.write("".join(md_lines))
    dlog.get_logger().info(f"Report saved: {md_path}")

    # JSON
    import json as _json
    class NpEncoder(_json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    all_data = {"gsod": gsod_results, "lorenz63": l63_results}
    with open(RUN_DIR / "detailed_results.json", "w") as f:
        _json.dump(all_data, f, indent=2, cls=NpEncoder)

    dlog.log_section("COMPLETE")
    dlog.get_logger().info(f"All outputs: {RUN_DIR}")


if __name__ == "__main__":
    main()
