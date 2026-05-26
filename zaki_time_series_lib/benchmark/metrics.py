from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats as scipy_stats

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger

logger = get_logger(__name__)


class MetricsCalculator:
    def __init__(self):
        self.dlog = DetailedLogger("benchmark.metrics")

    def ma(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(self.mse(y_true, y_pred)))

    def mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
        mask = y_true != 0
        if not mask.any():
            return float('inf')
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
        denominator = np.abs(y_true) + np.abs(y_pred)
        mask = denominator != 0
        if not mask.any():
            return float('inf')
        return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)

    def mase(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: Optional[np.ndarray] = None,
             seasonality: int = 24) -> float:
        y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
        if y_train is not None:
            y_train = np.asarray(y_train).flatten()
            if len(y_train) > seasonality:
                naive_error = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
            else:
                naive_error = np.mean(np.abs(np.diff(y_train)))
        else:
            naive_error = np.mean(np.abs(np.diff(y_true)))
        if naive_error == 0:
            return float('inf')
        return float(np.mean(np.abs(y_true - y_pred)) / naive_error)

    def r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return float('nan')
        return float(1 - ss_res / ss_tot)

    def mpe(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
        mask = y_true != 0
        if not mask.any():
            return float('inf')
        return float(np.mean((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)

    def wmape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
        if np.sum(np.abs(y_true)) == 0:
            return float('inf')
        return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)

    def max_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.max(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    def mda(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        if len(true_dir) == 0:
            return 0.0
        return float(np.mean(true_dir == pred_dir) * 100)

    def correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        return float(np.corrcoef(y_true, y_pred)[0, 1])

    def rmse_relative(self, y_true: np.ndarray, y_pred: np.ndarray, baseline_pred: np.ndarray) -> float:
        rmse_model = self.rmse(y_true, y_pred)
        rmse_baseline = self.rmse(y_true, baseline_pred)
        if rmse_baseline == 0:
            return float('inf')
        return float(rmse_model / rmse_baseline)

    def aic(self, mse: float, n_params: int, n_samples: int) -> float:
        if mse <= 0:
            return float('inf')
        return float(n_samples * np.log(mse) + 2 * n_params)

    def bic(self, mse: float, n_params: int, n_samples: int) -> float:
        if mse <= 0:
            return float('inf')
        return float(n_samples * np.log(mse) + n_params * np.log(n_samples))

    def nrmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r = self.rmse(y_true, y_pred)
        s = float(np.std(np.asarray(y_true)))
        if s == 0:
            return float('inf')
        return float(r / s)

    def skill_score(self, y_true: np.ndarray, y_pred: np.ndarray, y_baseline: np.ndarray) -> float:
        r_model = self.rmse(y_true, y_pred)
        r_base = self.rmse(y_true, y_baseline)
        if r_base == 0:
            return float('nan')
        return float(1.0 - r_model / r_base)

    def variance_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        var_true = float(np.var(y_true))
        var_pred = float(np.var(y_pred))
        if var_true == 0:
            return float('nan')
        return float(var_pred / var_true)

    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray,
                    y_train: Optional[np.ndarray] = None,
                    seasonality: int = 24,
                    n_params: int = 0,
                    baseline_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
        self.dlog.log_section("Computing All Metrics")
        y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
        mse_val = self.mse(y_true, y_pred)

        metrics = {
            "ma": self.ma(y_true, y_pred),
            "mse": mse_val,
            "rmse": self.rmse(y_true, y_pred),
            "nrmse": self.nrmse(y_true, y_pred),
            "mape": self.mape(y_true, y_pred),
            "smape": self.smape(y_true, y_pred),
            "mase": self.mase(y_true, y_pred, y_train, seasonality),
            "r2": self.r2(y_true, y_pred),
            "mpe": self.mpe(y_true, y_pred),
            "wmape": self.wmape(y_true, y_pred),
            "max_error": self.max_error(y_true, y_pred),
            "mda": self.mda(y_true, y_pred),
            "correlation": self.correlation(y_true, y_pred),
            "aic": self.aic(mse_val, n_params, len(y_true)),
            "bic": self.bic(mse_val, n_params, len(y_true)),
            "variance_ratio": self.variance_ratio(y_true, y_pred),
        }

        if baseline_pred is not None:
            metrics["rmse_relative"] = self.rmse_relative(y_true, y_pred, baseline_pred)
            metrics["skill_score"] = self.skill_score(y_true, y_pred, baseline_pred)

        for k, v in metrics.items():
            self.dlog.get_logger().info(f"  {k.upper()}: {v:.6f}")

        return metrics

    def vpt(self, y_true_seq: np.ndarray, y_pred_seq: np.ndarray, threshold: float = 0.4) -> int:
        r"""
        Valid Prediction Time (VPT).
        First horizon h where NRMSE >= threshold, computed from multi-horizon arrays.

        Parameters
        ----------
        y_true_seq : (n_samples, n_horizons)
        y_pred_seq : (n_samples, n_horizons)
        threshold : float, default=0.4
            NRMSE threshold

        Returns
        -------
        vpt_horizon : int
            First horizon where NRMSE exceeds threshold. If never exceeded, returns n_horizons.
        """
        n_h = y_true_seq.shape[1]
        for h in range(n_h):
            if self.nrmse(y_true_seq[:, h], y_pred_seq[:, h]) >= threshold:
                return h
        return n_h

    def fsdh(self, y_true_horizons: np.ndarray, y_model_horizons: np.ndarray,
             y_persist_horizons: np.ndarray) -> int:
        r"""
        Forecast Skill Decay Horizon (FSDH).
        Last horizon h where model RMSE < persistence RMSE.

        Parameters
        ----------
        y_true_horizons : (n_samples, n_horizons)
        y_model_horizons : (n_samples, n_horizons)
        y_persist_horizons : (n_samples, n_horizons)

        Returns
        -------
        fsdh_horizon : int
            Last horizon where model beats persistence. Returns 0 if never beats it.
        """
        max_h = y_true_horizons.shape[1]
        fsdh_val = 0
        for h in range(max_h):
            rmse_model = self.rmse(y_true_horizons[:, h], y_model_horizons[:, h])
            rmse_persist = self.rmse(y_true_horizons[:, h], y_persist_horizons[:, h])
            if rmse_model < rmse_persist:
                fsdh_val = h + 1
        return fsdh_val

    def evaluate_horizons(self, y_true_horizons: np.ndarray, y_pred_horizons: np.ndarray,
                           y_persist_horizons: Optional[np.ndarray] = None,
                           horizons: Optional[List[int]] = None,
                           label: str = "model",
                           vpt_threshold: float = 0.4) -> Dict[str, Any]:
        r"""
        Full multi-horizon evaluation.
        Computes per-horizon RMSE, MAE, NRMSE, Skill Score, plus VPT and FSDH.

        Parameters
        ----------
        y_true_horizons : (n_samples, n_horizons)
        y_pred_horizons : (n_samples, n_horizons)
        y_persist_horizons : (n_samples, n_horizons) or None
        horizons : list[int], optional
            The horizon values (e.g., [1, 3, 6, 12, 24])
        label : str
        vpt_threshold : float

        Returns
        -------
        results : dict with keys:
            'per_horizon' : {h: {rmse, mae, nrmse, skill}}
            'vpt' : int
            'fsdh' : int
        """
        self.dlog.log_section(f"Multi-Horizon Evaluation: {label}")
        n_h = y_true_horizons.shape[1]
        if horizons is None:
            horizons = list(range(1, n_h + 1))

        per_horizon = {}
        for hi, h in enumerate(horizons):
            yt = y_true_horizons[:, hi]
            yp = y_pred_horizons[:, hi]

            rh = {
                'rmse': self.rmse(yt, yp),
                'mae': self.ma(yt, yp),
                'nrmse': self.nrmse(yt, yp),
            }

            if y_persist_horizons is not None:
                rh['skill'] = self.skill_score(yt, yp, y_persist_horizons[:, hi])

            per_horizon[f"h={h}"] = rh
            self.dlog.get_logger().info(
                f"  h={h:2d} | RMSE={rh['rmse']:.4f} | MAE={rh['mae']:.4f} | "
                f"NRMSE={rh['nrmse']:.4f}" +
                (f" | Skill={rh['skill']:.4f}" if 'skill' in rh else "")
            )

        vpt_val = self.vpt(y_true_horizons, y_pred_horizons, vpt_threshold)
        self.dlog.get_logger().info(f"  VPT  = {vpt_val} (first h where NRMSE >= {vpt_threshold})")

        if y_persist_horizons is not None:
            fsdh_val = self.fsdh(y_true_horizons, y_pred_horizons, y_persist_horizons)
            self.dlog.get_logger().info(f"  FSDH = {fsdh_val} (last h where model beats persistence)")
        else:
            fsdh_val = 0

        return {
            'per_horizon': per_horizon,
            'vpt': vpt_val,
            'fsdh': fsdh_val,
        }


def compute_all_metrics_from_dict(predictions: Dict[str, np.ndarray],
                                   y_true: np.ndarray,
                                   y_train: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    calc = MetricsCalculator()
    results = {}
    for name, pred in predictions.items():
        results[name] = calc.compute_all(y_true, pred, y_train)
    return results
