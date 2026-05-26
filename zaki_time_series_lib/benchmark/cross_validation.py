from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger

logger = get_logger(__name__)


class TimeSeriesSplitter:
    def __init__(self, n_splits: int = 5, test_size: int = 24,
                 gap: int = 0, expanding: bool = True):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding
        self.dlog = DetailedLogger("benchmark.cv")
        self.dlog.get_logger().info(
            f"TimeSeriesSplitter: n_splits={n_splits}, test_size={test_size}, "
            f"gap={gap}, expanding={expanding}"
        )

    def split(self, y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        y = np.asarray(y).flatten()
        n = len(y)
        total_per_split = self.test_size + self.gap

        for i in range(self.n_splits):
            test_end = n - i * total_per_split
            test_start = test_end - self.test_size

            if test_start <= 0:
                self.dlog.get_logger().warning(f"Not enough data for split {i + 1}, stopping")
                break

            if self.expanding:
                train_indices = np.arange(0, max(1, test_start - self.gap))
            else:
                train_end = test_start - self.gap
                train_start = max(0, train_end - self.test_size * 3)
                train_indices = np.arange(train_start, train_end)

            test_indices = np.arange(test_start, test_end)
            self.dlog.get_logger().debug(
                f"Split {i + 1}: train={len(train_indices)}, test={len(test_indices)}"
            )
            yield train_indices, test_indices

    def get_n_splits(self, y: np.ndarray) -> int:
        return sum(1 for _ in self.split(y))


class RollingWindowCV:
    def __init__(self, window_size: int = 1000, test_size: int = 24, step: int = 24):
        self.window_size = window_size
        self.test_size = test_size
        self.step = step
        self.dlog = DetailedLogger("benchmark.rolling_cv")

    def split(self, y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        y = np.asarray(y).flatten()
        n = len(y)

        for start in range(0, n - self.window_size - self.test_size, self.step):
            train_end = start + self.window_size
            test_start = train_end
            test_end = test_start + self.test_size

            if test_end > n:
                break

            train_indices = np.arange(start, train_end)
            test_indices = np.arange(test_start, test_end)
            yield train_indices, test_indices

    def get_n_splits(self, y: np.ndarray) -> int:
        return sum(1 for _ in self.split(y))


class PurgedWalkForwardCV:
    def __init__(self, n_splits: int = 5, test_size: int = 24,
                 purge_size: int = 24, embargo_size: int = 12):
        self.n_splits = n_splits
        self.test_size = test_size
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        self.dlog = DetailedLogger("benchmark.purged_cv")

    def split(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        y = np.asarray(y).flatten()
        n = len(y)
        split_size = (n - self.test_size) // self.n_splits

        for i in range(self.n_splits):
            train_end = (i + 1) * split_size
            test_start = train_end + self.purge_size
            test_end = min(test_start + self.test_size, n)

            if test_start >= n:
                break

            train_indices = np.arange(0, train_end)
            embargo_end = min(train_end + self.embargo_size, n)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices


def cross_validate(y: np.ndarray, model, splitter: TimeSeriesSplitter,
                   **kwargs) -> List[Dict[str, Any]]:
    from zaki_time_series_lib.benchmark.metrics import MetricsCalculator

    logger = get_logger(__name__)
    calc = MetricsCalculator()
    results = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(y)):
        logger.info(f"{'='*40} Fold {fold + 1} {'='*40}")
        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(y_train, **kwargs)
        y_pred = model.predict(len(y_test))

        metrics = calc.compute_all(y_test, y_pred, y_train)
        metrics["fold"] = fold + 1
        results.append(metrics)

    avg_metrics = {}
    for key in results[0]:
        if key != "fold":
            vals = [r[key] for r in results]
            avg_metrics[f"avg_{key}"] = np.mean(vals)
            avg_metrics[f"std_{key}"] = np.std(vals)

    logger.info(f"{'='*20} Cross-Validation Summary {'='*20}")
    for k, v in avg_metrics.items():
        logger.info(f"  {k}: {v:.6f}")

    return {"per_fold": results, "average": avg_metrics}
