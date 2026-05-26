from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd

from zaki_time_series_lib.benchmark.runner import BenchmarkRunner
from zaki_time_series_lib.config.settings import settings
from zaki_time_series_lib.data.base_loader import BaseDatasetLoader
from zaki_time_series_lib.data.preprocessing.imputation import Imputer, ForwardFillImputer
from zaki_time_series_lib.data.preprocessing.scalers import TimeSeriesScaler, StandardScaler, SCALER_REGISTRY
from zaki_time_series_lib.data.preprocessing.transforms import SeriesTransformer
from zaki_time_series_lib.models.base import BaseTimeSeriesModel
from zaki_time_series_lib.pipeline.registry import DatasetRegistry, ModelRegistry
from zaki_time_series_lib.utils.decorators import log_entry_exit, timer
from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger
from zaki_time_series_lib.utils.results_exporter import ResultsExporter
from zaki_time_series_lib.utils.visualization import TimeSeriesVisualizer

logger = get_logger(__name__)


class TimeSeriesPipeline:
    def __init__(self, output_dir: Optional[str] = None, random_seed: int = 42):
        self.dlog = DetailedLogger("pipeline")
        self.dataset_registry = DatasetRegistry()
        self.model_registry = ModelRegistry()
        self.benchmark = BenchmarkRunner(output_dir)
        self.exporter = self.benchmark.exporter
        self.visualizer = self.benchmark.visualizer
        self.random_seed = random_seed
        self._set_seed()
        self.pipeline_config: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

        self.dlog.log_section("Pipeline Initialized")
        self.dlog.get_logger().info(f"Output directory: {output_dir or settings.RESULTS_DIR}")

    def _set_seed(self):
        np.random.seed(self.random_seed)
        try:
            import torch
            torch.manual_seed(self.random_seed)
        except ImportError:
            pass
        try:
            import random
            random.seed(self.random_seed)
        except ImportError:
            pass

    @log_entry_exit()
    def set_config(self, **config) -> "TimeSeriesPipeline":
        self.pipeline_config.update(config)
        self.dlog.log_dict(config, "Pipeline Configuration")
        return self

    @log_entry_exit()
    def load_data(self, dataset_name: str, **loader_kwargs) -> BaseDatasetLoader:
        self.dlog.log_section(f"Pipeline Step: Load Data - {dataset_name}")
        loader = self.dataset_registry.get_dataset(dataset_name, **loader_kwargs)
        loader.load()
        self.pipeline_config["dataset"] = dataset_name
        self.pipeline_config["dataset_metadata"] = loader.get_metadata()
        return loader

    def _resolve_scaler(self, scaler: Optional[Union[str, TimeSeriesScaler]]) -> Optional[TimeSeriesScaler]:
        if scaler is None or isinstance(scaler, TimeSeriesScaler):
            return scaler
        if isinstance(scaler, str):
            if scaler.lower() in SCALER_REGISTRY:
                self.dlog.get_logger().info(f"Resolved scaler '{scaler}' from registry")
                return SCALER_REGISTRY[scaler.lower()]()
            raise ValueError(f"Unknown scaler '{scaler}'. Available: {list(SCALER_REGISTRY.keys())}")
        return scaler

    @log_entry_exit()
    def preprocess(self, data: pd.DataFrame, target_col: Optional[str] = None,
                    scaler: Optional[Union[str, TimeSeriesScaler]] = None,
                    imputer: Optional[Imputer] = None,
                    transformer: Optional[SeriesTransformer] = None) -> Dict[str, Any]:
        self.dlog.log_section("Pipeline Step: Preprocessing")

        scaler = self._resolve_scaler(scaler)
        result = {"data": data.copy(), "target_col": target_col or data.columns[0]}

        if imputer:
            self.dlog.get_logger().info("Applying imputation...")
            result["data"] = imputer.fit_transform(result["data"])

        if transformer:
            self.dlog.get_logger().info("Applying series transform...")
            transformed = transformer.fit_transform(result["data"][result["target_col"]])
            result["data"][result["target_col"]] = transformed
            result["transformer"] = transformer

        if scaler:
            self.dlog.get_logger().info("Applying scaling...")
            result["scaler"] = scaler

        self.pipeline_config["preprocessing"] = {
            "scaler": scaler.__class__.__name__ if scaler else None,
            "imputer": imputer.__class__.__name__ if imputer else None,
            "transformer": transformer.__class__.__name__ if transformer else None,
        }
        return result

    @log_entry_exit()
    def split_data(self, data: pd.DataFrame, target_col: str,
                    train_split: float = 0.7, val_split: float = 0.1,
                    test_split: float = 0.2) -> Dict[str, Any]:
        self.dlog.log_section("Pipeline Step: Data Splitting")
        n = len(data)
        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)

        train = data.iloc[:train_end]
        val = data.iloc[train_end:val_end] if val_split > 0 else pd.DataFrame()
        test = data.iloc[val_end:]

        self.dlog.get_logger().info(
            f"Train: {len(train)} ({len(train)/n*100:.1f}%), "
            f"Val: {len(val)} ({len(val)/n*100:.1f}%), "
            f"Test: {len(test)} ({len(test)/n*100:.1f}%)"
        )

        splits = {
            "train": train[target_col].values.astype(np.float64),
            "val": val[target_col].values.astype(np.float64) if len(val) > 0 else np.array([]),
            "test": test[target_col].values.astype(np.float64),
            "train_df": train,
            "val_df": val,
            "test_df": test,
            "target_col": target_col,
        }

        self.pipeline_config["splits"] = {
            "train_size": len(train), "val_size": len(val), "test_size": len(test)
        }
        return splits

    @log_entry_exit()
    def create_models(self, model_configs: List[Dict[str, Any]]) -> List[BaseTimeSeriesModel]:
        self.dlog.log_section(f"Pipeline Step: Create {len(model_configs)} Models")
        models = []
        for config in model_configs:
            name = config.get("name")
            params = config.get("params", {})
            self.dlog.get_logger().info(f"Creating model: {name} with {params}")
            model = self.model_registry.get_model(name, **params)
            models.append(model)
        return models

    @log_entry_exit()
    @timer()
    def run(self, dataset_name: str,
            model_configs: List[Dict[str, Any]],
            target_col: Optional[str] = None,
            scaler: Optional[Union[str, TimeSeriesScaler]] = None,
            imputer: Optional[Imputer] = None,
            transformer: Optional[SeriesTransformer] = None,
            use_validation: bool = False,
            run_cv: bool = False,
            cv_splits: int = 5,
            cv_test_size: int = 24) -> Dict[str, Any]:
        self.dlog.log_section("=" * 20 + " FULL PIPELINE RUN " + "=" * 20)

        scaler = self._resolve_scaler(scaler)
        loader = self.load_data(dataset_name)

        data = loader.data
        preprocessed = self.preprocess(data, target_col, scaler, imputer, transformer)
        if scaler is None:
            scaler = preprocessed.get("scaler")

        splits = self.split_data(
            preprocessed["data"], preprocessed["target_col"],
            self.pipeline_config.get("train_split", settings.DEFAULT_TRAIN_SPLIT),
            self.pipeline_config.get("val_split", settings.DEFAULT_VAL_SPLIT),
            self.pipeline_config.get("test_split", settings.DEFAULT_TEST_SPLIT),
        )

        models = self.create_models(model_configs)

        y_train = splits["train"]
        y_test = splits["test"]

        if scaler is not None:
            self.dlog.get_logger().info("Scaling target variable...")
            scaler.fit(y_train.reshape(-1, 1))
            y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

            self.dlog.log_data_stats(y_train_scaled, "Scaled y_train")
            self.dlog.log_data_stats(y_test_scaled, "Scaled y_test")
        else:
            y_train_scaled = y_train
            y_test_scaled = y_test

        self.dlog.log_section("Running Benchmark")
        results = self.benchmark.run_multiple(models, y_train_scaled, y_test_scaled)

        if scaler is not None:
            self.dlog.log_section("Inverse-transforming predictions")
            for model_name, result in results.items():
                if isinstance(result, dict) and result.get("success"):
                    raw_pred = result["predictions"]
                    inv_pred = scaler.inverse_transform(raw_pred.reshape(-1, 1)).flatten()
                    result["predictions_original_scale"] = inv_pred
                    from zaki_time_series_lib.benchmark.metrics import MetricsCalculator
                    mc = MetricsCalculator()
                    result["metrics_original_scale"] = mc.compute_all(y_test, inv_pred, y_train)

        if run_cv:
            self.dlog.log_section("Running Cross-Validation")
            for model in models:
                try:
                    cv_result = self.benchmark.run_cross_validation(
                        model, y_train_scaled if scaler else y_train,
                        n_splits=cv_splits, test_size=cv_test_size
                    )
                    results[f"{model.name}_cv"] = cv_result
                except Exception as e:
                    self.dlog.get_logger().error(f"CV failed for {model.name}: {e}")

        results["pipeline_config"] = self.pipeline_config
        results["dataset_metadata"] = loader.get_metadata()

        self.dlog.log_section("Pipeline Complete")
        best = results.get("best_model", "N/A")
        self.dlog.get_logger().info(f"Best model: {best}")

        self.results = results
        return results

    def get_results(self) -> Dict[str, Any]:
        return self.results

    def get_summary_dataframe(self) -> pd.DataFrame:
        return self.benchmark.get_summary()

    def print_summary(self):
        self.benchmark.print_summary()
