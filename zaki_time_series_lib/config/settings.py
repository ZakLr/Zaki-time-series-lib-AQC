import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Settings:
    VERSION: str = "0.1.0"
    PROJECT_NAME: str = "zaki_time_series_lib"

    LOG_LEVEL: str = os.getenv("ZAKI_LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("ZAKI_LOG_FORMAT", "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s")
    LOG_DATE_FORMAT: str = os.getenv("ZAKI_LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")
    LOG_FILE: Optional[str] = os.getenv("ZAKI_LOG_FILE", None)

    DATA_CACHE_DIR: str = os.getenv("ZAKI_DATA_CACHE", os.path.join(os.path.expanduser("~"), ".zaki_ts_data"))
    RESULTS_DIR: str = os.getenv("ZAKI_RESULTS_DIR", "./zaki_results")
    RANDOM_SEED: int = int(os.getenv("ZAKI_RANDOM_SEED", "42"))
    N_JOBS: int = int(os.getenv("ZAKI_N_JOBS", "-1"))

    DEFAULT_TRAIN_SPLIT: float = 0.7
    DEFAULT_VAL_SPLIT: float = 0.1
    DEFAULT_TEST_SPLIT: float = 0.2

    DEFAULT_FORECAST_HORIZON: int = 24
    DEFAULT_SEQUENCE_LENGTH: int = 168

    DL_BATCH_SIZE: int = int(os.getenv("ZAKI_DL_BATCH_SIZE", "64"))
    DL_MAX_EPOCHS: int = int(os.getenv("ZAKI_DL_MAX_EPOCHS", "100"))
    DL_LEARNING_RATE: float = float(os.getenv("ZAKI_DL_LEARNING_RATE", "1e-3"))
    DL_EARLY_STOPPING_PATIENCE: int = int(os.getenv("ZAKI_DL_PATIENCE", "10"))
    DL_DEVICE: str = os.getenv("ZAKI_DL_DEVICE", "auto")

    DATASET_LIST: tuple = (
        "ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity", "Traffic", "ExchangeRate"
    )

    ENSEMBLE_METRICS: tuple = ("mae", "mse", "rmse", "mape", "smape", "mase", "r2")


settings = Settings()
