from typing import Any, Dict, List, Optional, Type

from zaki_time_series_lib.data import DATASET_REGISTRY
from zaki_time_series_lib.data.preprocessing.scalers import SCALER_REGISTRY
from zaki_time_series_lib.data.preprocessing.imputation import IMPUTER_REGISTRY
from zaki_time_series_lib.data.preprocessing.transforms import TRANSFORMER_REGISTRY
from zaki_time_series_lib.models.statistical import STATISTICAL_MODEL_REGISTRY
from zaki_time_series_lib.models.ml import ML_MODEL_REGISTRY
from zaki_time_series_lib.models.deep_learning import DL_MODEL_REGISTRY
from zaki_time_series_lib.models.base import BaseTimeSeriesModel
from zaki_time_series_lib.utils.logger import get_logger

logger = get_logger(__name__)


MODEL_REGISTRY = {}
MODEL_REGISTRY.update(STATISTICAL_MODEL_REGISTRY)
MODEL_REGISTRY.update(ML_MODEL_REGISTRY)
MODEL_REGISTRY.update(DL_MODEL_REGISTRY)


class ModelRegistry:
    def __init__(self):
        self._models = MODEL_REGISTRY.copy()

    def list_models(self, category: Optional[str] = None) -> List[str]:
        if category == "statistical":
            return list(STATISTICAL_MODEL_REGISTRY.keys())
        elif category == "ml":
            return list(ML_MODEL_REGISTRY.keys())
        elif category == "deep_learning":
            return list(DL_MODEL_REGISTRY.keys())
        return list(self._models.keys())

    def get_model(self, name: str, **kwargs) -> BaseTimeSeriesModel:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        logger.info(f"Creating model: {name} with params: {kwargs}")
        return self._models[name](**kwargs)

    def register_model(self, name: str, model_class: Type[BaseTimeSeriesModel]):
        if name in self._models:
            logger.warning(f"Overwriting existing model registration: {name}")
        self._models[name] = model_class
        logger.info(f"Registered custom model: {name}")


class DatasetRegistry:
    def __init__(self):
        self._datasets = DATASET_REGISTRY.copy()

    def list_datasets(self) -> List[str]:
        return list(self._datasets.keys())

    def get_dataset(self, name: str, **kwargs):
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found. Available: {list(self._datasets.keys())}")
        logger.info(f"Creating dataset loader: {name}")
        return self._datasets[name](**kwargs)

    def register_dataset(self, name: str, loader_class):
        if name in self._datasets:
            logger.warning(f"Overwriting existing dataset registration: {name}")
        self._datasets[name] = loader_class
        logger.info(f"Registered custom dataset: {name}")


def list_all_registered() -> Dict[str, List[str]]:
    return {
        "datasets": list(DATASET_REGISTRY.keys()),
        "statistical_models": list(STATISTICAL_MODEL_REGISTRY.keys()),
        "ml_models": list(ML_MODEL_REGISTRY.keys()),
        "deep_learning_models": list(DL_MODEL_REGISTRY.keys()),
        "scalers": list(SCALER_REGISTRY.keys()),
        "imputers": list(IMPUTER_REGISTRY.keys()),
        "transformers": list(TRANSFORMER_REGISTRY.keys()),
    }
