import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from zaki_time_series_lib.config.settings import settings
from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger

logger = get_logger(__name__)


class BaseDatasetLoader(ABC):
    def __init__(self, name: str, cache_dir: Optional[str] = None,
                 train_split: float = 0.7, val_split: float = 0.1, test_split: float = 0.2):
        self.name = name
        self.cache_dir = Path(cache_dir or settings.DATA_CACHE_DIR) / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.dlog = DetailedLogger(f"data.{name}")
        self._data: Optional[pd.DataFrame] = None
        self._loaded = False

        splits_sum = train_split + val_split + test_split
        if not abs(splits_sum - 1.0) < 1e-6:
            logger.warning(f"Splits sum to {splits_sum:.4f}, normalizing...")
            total = splits_sum
            self.train_split /= total
            self.val_split /= total
            self.test_split /= total

    @abstractmethod
    def _download(self) -> pd.DataFrame:
        pass

    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        cache_file = self.cache_dir / "data.parquet"
        if cache_file.exists():
            self.dlog.get_logger().info(f"Loading cached data from {cache_file}")
            return pd.read_parquet(cache_file)
        csv_file = self.cache_dir / "data.csv"
        if csv_file.exists():
            self.dlog.get_logger().info(f"Loading cached data from {csv_file}")
            return pd.read_csv(csv_file, index_col=0, parse_dates=True)
        return None

    def _save_to_cache(self, data: pd.DataFrame):
        cache_file = self.cache_dir / "data.parquet"
        data.to_parquet(cache_file)
        self.dlog.get_logger().info(f"Data cached to {cache_file}")

    def load(self, force_download: bool = False) -> pd.DataFrame:
        self.dlog.log_section(f"Loading Dataset: {self.name}")

        if self._loaded and self._data is not None:
            self.dlog.get_logger().info(f"Returning cached in-memory data for {self.name}")
            return self._data

        if not force_download:
            cached = self._load_from_cache()
            if cached is not None:
                self._data = cached
                self._loaded = True
                self.dlog.log_data_shape(self._data)
                self.dlog.log_data_stats(self._data.iloc[:, 0] if self._data.shape[1] == 1 else self._data)
                return self._data

        self.dlog.get_logger().info(f"Downloading dataset: {self.name}")
        data = self._download()
        self._data = data
        self._loaded = True

        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                self._data.index = pd.to_datetime(self._data.index)
            except Exception:
                self.dlog.get_logger().warning("Could not convert index to DatetimeIndex")

        self._save_to_cache(self._data)
        self.dlog.log_data_shape(self._data)
        self.dlog.log_data_stats(self._data.iloc[:, 0] if self._data.shape[1] == 1 else self._data)
        self.dlog.get_logger().info(f"Dataset {self.name} loaded successfully")
        return self._data

    def get_splits(self, data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if data is None:
            data = self.load()
        n = len(data)
        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)

        train = data.iloc[:train_end]
        val = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]

        self.dlog.get_logger().info(
            f"Splits: train={len(train)} ({len(train)/n*100:.1f}%), "
            f"val={len(val)} ({len(val)/n*100:.1f}%), "
            f"test={len(test)} ({len(test)/n*100:.1f}%)"
        )
        return train, val, test

    def get_X_y(self, data: pd.DataFrame, target_col: str = None,
                sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if target_col is None:
            target_col = data.columns[0]
        if sequence_length is None:
            sequence_length = settings.DEFAULT_SEQUENCE_LENGTH

        series = data[target_col].values
        X, y = [], []
        for i in range(len(series) - sequence_length):
            X.append(series[i:i + sequence_length])
            y.append(series[i + sequence_length])
        self.dlog.get_logger().info(
            f"Created sequences: X shape ({len(X)}, {sequence_length}), y shape ({len(y)},)"
        )
        return np.array(X), np.array(y)

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.load()
        return self._data

    @property
    def freq(self) -> str:
        try:
            return pd.infer_freq(self.data.index)
        except Exception:
            return "unknown"

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

    @property
    def n_timesteps(self) -> int:
        return len(self.data)

    def get_metadata(self) -> Dict[str, Any]:
        d = self.data
        return {
            "name": self.name,
            "shape": d.shape,
            "freq": self.freq,
            "date_range": f"{d.index[0]} to {d.index[-1]}",
            "n_features": self.n_features,
            "n_timesteps": self.n_timesteps,
            "columns": list(d.columns),
            "dtypes": {c: str(d[c].dtype) for c in d.columns},
        }
