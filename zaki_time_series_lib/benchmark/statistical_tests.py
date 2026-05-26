from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger

logger = get_logger(__name__)


class StatisticalTestSuite:
    def __init__(self):
        self.dlog = DetailedLogger("benchmark.stat_tests")

    def diebold_mariano(self, y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray,
                        h: int = 1, one_sided: bool = True) -> Dict[str, Any]:
        self.dlog.log_section("Diebold-Mariano Test")
        y_true, pred1, pred2 = [np.asarray(a).flatten() for a in (y_true, pred1, pred2)]
        e1 = y_true - pred1
        e2 = y_true - pred2
        d = e1 ** 2 - e2 ** 2

        n = len(d)
        if h > 1:
            from statsmodels.tsa.stattools import acf
            gamma = np.zeros(h)
            gamma[0] = np.var(d, ddof=1)
            for k in range(1, h):
                gamma[k] = np.cov(d[:-k], d[k:])[0, 1] if len(d[:-k]) > 1 else 0
            var_d = gamma[0] + 2 * np.sum(gamma[1:])
        else:
            var_d = np.var(d, ddof=1)

        if var_d <= 0:
            self.dlog.get_logger().warning("Variance of loss differential is zero/negative")
            return {"statistic": 0.0, "p_value": 1.0, "DM": 0.0}

        dm_stat = np.mean(d) / np.sqrt(var_d / n)
        p_value = stats.norm.cdf(dm_stat) if one_sided else 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        better = "Model 1" if np.mean(e1 ** 2) < np.mean(e2 ** 2) else "Model 2"
        result = {
            "DM_statistic": float(dm_stat),
            "p_value": float(p_value),
            "significant_5pct": bool(p_value < 0.05),
            "better_model": better,
            "test_type": "one-sided" if one_sided else "two-sided",
            "horizon": h,
        }
        self.dlog.get_logger().info(f"DM statistic: {dm_stat:.4f}, p-value: {p_value:.4f}")
        self.dlog.get_logger().info(f"Better model: {better}")
        return result

    def paired_t_test(self, y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Dict[str, Any]:
        self.dlog.log_section("Paired t-Test")
        y_true, pred1, pred2 = [np.asarray(a).flatten() for a in (y_true, pred1, pred2)]
        e1 = np.abs(y_true - pred1)
        e2 = np.abs(y_true - pred2)

        t_stat, p_value = stats.ttest_rel(e1, e2)
        result = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_5pct": bool(p_value < 0.05),
            "mean_error_1": float(np.mean(e1)),
            "mean_error_2": float(np.mean(e2)),
        }
        self.dlog.get_logger().info(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        return result

    def wilcoxon_signed_rank(self, y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Dict[str, Any]:
        self.dlog.log_section("Wilcoxon Signed-Rank Test")
        y_true, pred1, pred2 = [np.asarray(a).flatten() for a in (y_true, pred1, pred2)]
        e1 = np.abs(y_true - pred1)
        e2 = np.abs(y_true - pred2)

        stat, p_value = stats.wilcoxon(e1, e2)
        result = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant_5pct": bool(p_value < 0.05),
        }
        self.dlog.get_logger().info(f"Wilcoxon: stat={stat:.4f}, p={p_value:.4f}")
        return result

    def model_significance(self, residuals: np.ndarray) -> Dict[str, Any]:
        self.dlog.log_section("Model Residual Tests")
        residuals = np.asarray(residuals).flatten()
        n = len(residuals)

        _, shapiro_p = stats.shapiro(residuals[:min(5000, n)])
        _, jarque_p = stats.jarque_bera(residuals)
        _, lilliefors_p = stats.normaltest(residuals)

        from statsmodels.stats.diagnostic import acorr_ljungbox
        try:
            lb_result = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
            lb_pvalues = lb_result['lb_pvalue'].to_dict() if hasattr(lb_result, 'to_dict') else {}
        except Exception:
            lb_pvalues = {}

        result = {
            "shapiro_p_value": float(shapiro_p),
            "jarque_bera_p_value": float(jarque_p),
            "normality_test_p_value": float(lilliefors_p),
            "is_normal_5pct": bool(shapiro_p > 0.05),
            "ljung_box_p_values": lb_pvalues,
        }
        return result

    def compare_models(self, y_true: np.ndarray,
                       predictions: Dict[str, np.ndarray],
                       baseline_model: Optional[str] = None) -> Dict[str, Any]:
        self.dlog.log_section("Model Comparison")
        results = {}
        model_names = list(predictions.keys())

        if baseline_model is None:
            baseline_model = model_names[0]

        for name in model_names:
            if name != baseline_model:
                results[f"{name}_vs_{baseline_model}"] = self.diebold_mariano(
                    y_true, predictions[name], predictions[baseline_model]
                )

        return results

    def summarize_all(self, y_true: np.ndarray,
                       predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        all_tests = {}
        models = list(predictions.keys())

        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                key = f"{m1}_vs_{m2}"
                all_tests[key] = self.diebold_mariano(y_true, predictions[m1], predictions[m2])

        return all_tests
