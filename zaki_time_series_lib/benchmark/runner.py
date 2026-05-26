import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from zaki_time_series_lib.benchmark.metrics import MetricsCalculator
from zaki_time_series_lib.benchmark.cross_validation import TimeSeriesSplitter, cross_validate
from zaki_time_series_lib.benchmark.statistical_tests import StatisticalTestSuite
from zaki_time_series_lib.data.base_loader import BaseDatasetLoader
from zaki_time_series_lib.data.preprocessing.scalers import TimeSeriesScaler, StandardScaler
from zaki_time_series_lib.models.base import BaseTimeSeriesModel
from zaki_time_series_lib.utils.decorators import log_entry_exit, timer
from zaki_time_series_lib.utils.logger import get_logger, DetailedLogger
from zaki_time_series_lib.utils.results_exporter import ResultsExporter
from zaki_time_series_lib.utils.visualization import TimeSeriesVisualizer

logger = get_logger(__name__)


class BenchmarkRunner:
    def __init__(self, output_dir: Optional[str] = None):
        self.dlog = DetailedLogger("benchmark.runner")
        self.metrics_calc = MetricsCalculator()
        self.stat_tests = StatisticalTestSuite()
        self.exporter = ResultsExporter(output_dir)
        self.visualizer = TimeSeriesVisualizer(output_dir)
        self.results: Dict[str, Any] = {}

    @log_entry_exit()
    @timer()
    def run_single(self, model: BaseTimeSeriesModel, y_train: np.ndarray,
                   y_test: np.ndarray, X_train: Optional[np.ndarray] = None,
                   X_test: Optional[np.ndarray] = None,
                   model_params: Optional[Dict] = None,
                   **kwargs) -> Dict[str, Any]:
        self.dlog.log_section(f"Running Single Model: {model.name}")

        if model_params:
            model.set_params(**model_params)

        start_time = time.time()
        self.dlog.get_logger().info(f"Training {model.name} on {len(y_train)} samples...")

        try:
            model.fit(y_train, X_train, **kwargs)
        except Exception as e:
            self.dlog.get_logger().error(f"Training failed for {model.name}: {e}")
            return {"model_name": model.name, "error": str(e), "success": False}

        train_time = time.time() - start_time
        model.training_time = train_time

        self.dlog.get_logger().info(f"Predicting horizon={len(y_test)}...")
        try:
            if X_test is not None:
                y_pred = model.predict(len(y_test), X_test)
            else:
                y_pred = model.predict(len(y_test))
        except Exception as e:
            self.dlog.get_logger().error(f"Prediction failed for {model.name}: {e}")
            return {"model_name": model.name, "error": str(e), "success": False}

        self.dlog.get_logger().info("Computing evaluation metrics...")
        metrics = self.metrics_calc.compute_all(y_test, y_pred, y_train)

        result = {
            "model_name": model.name,
            "success": True,
            "training_time": train_time,
            "predictions": y_pred,
            "metrics": metrics,
            "model_summary": model.summary(),
        }

        self.exporter.export_predictions(model.name, y_test, y_pred)
        self.exporter.export_metrics({model.name: metrics}, f"metrics_{model.name}")
        self.exporter.export_model_params(model.name, model.get_params())
        self.visualizer.plot_forecast(
            y_test, y_pred, model.name,
            filename=f"forecast_{model.name}.png"
        )

        self.results[model.name] = result
        self.dlog.log_section(f"Completed: {model.name}")
        return result

    @log_entry_exit()
    @timer()
    def run_multiple(self, models: List[BaseTimeSeriesModel],
                     y_train: np.ndarray, y_test: np.ndarray,
                     X_train: Optional[np.ndarray] = None,
                     X_test: Optional[np.ndarray] = None,
                     **kwargs) -> Dict[str, Dict[str, Any]]:
        self.dlog.log_section(f"Running {len(models)} Models")

        all_results = {}
        all_predictions = {}
        all_metrics = {}

        for model in models:
            result = self.run_single(model, y_train, y_test, X_train, X_test, **kwargs)
            all_results[model.name] = result
            if result.get("success"):
                all_predictions[model.name] = result["predictions"]
                all_metrics[model.name] = result["metrics"]

        if len(all_metrics) > 1:
            self.dlog.log_section("Comparing All Models")
            dm_results = self.stat_tests.summarize_all(y_test, all_predictions)
            all_results["statistical_tests"] = dm_results

            self.dlog.get_logger().info(f"{'='*60}")
            self.dlog.get_logger().info(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
            self.dlog.get_logger().info(f"{'='*60}")
            for name, m in sorted(all_metrics.items(),
                                  key=lambda x: x[1].get('rmse', float('inf'))):
                self.dlog.get_logger().info(
                    f"{name:<20} {m.get('rmse', 0):<12.6f} {m.get('ma', 0):<12.6f} {m.get('mape', 0):<12.6f}"
                )

            self.exporter.export_metrics_comparison(all_metrics)
            self.exporter.export_forecasts(y_test, all_predictions)
            self.visualizer.plot_multi_forecast(y_test, all_predictions)

            sorted_models = sorted(all_metrics.items(),
                                   key=lambda x: x[1].get('rmse', float('inf')))
            all_results["best_model"] = sorted_models[0][0]
            self.dlog.get_logger().info(f"Best model (by RMSE): {sorted_models[0][0]}")

        config = {
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_models": len(models),
        }
        self.exporter.export_run_summary(config, all_metrics,
                                         list(all_results.keys()))

        report_data = {
            "config": config,
            "metrics_comparison": all_metrics,
            "best_model": all_results.get("best_model", "N/A"),
        }
        self.exporter.export_comprehensive_report(report_data)

        return all_results

    @log_entry_exit()
    @timer()
    def run_cross_validation(self, model: BaseTimeSeriesModel, y: np.ndarray,
                              n_splits: int = 5, test_size: int = 24,
                              **kwargs) -> Dict[str, Any]:
        self.dlog.log_section(f"Cross-Validation: {model.name}")
        splitter = TimeSeriesSplitter(n_splits=n_splits, test_size=test_size)
        cv_results = cross_validate(y, model, splitter, **kwargs)
        self.results[f"{model.name}_cv"] = cv_results
        return cv_results

    @log_entry_exit()
    @timer()
    def run_full_benchmark(self, dataset_loader: BaseDatasetLoader,
                            models: List[BaseTimeSeriesModel],
                            target_col: Optional[str] = None,
                            scaler: Optional[TimeSeriesScaler] = None,
                            cv: bool = False,
                            cv_splits: int = 5,
                            cv_test_size: int = 24,
                            **kwargs) -> Dict[str, Any]:
        self.dlog.log_section(f"Full Benchmark: {dataset_loader.name}")
        self.dlog.log_section("Step 1: Loading Data")
        data = dataset_loader.load()

        self.dlog.log_section("Step 2: Splitting Data")
        train_df, val_df, test_df = dataset_loader.get_splits(data)

        if target_col is None:
            target_col = data.columns[0]

        y_train = train_df[target_col].values.astype(np.float64)
        y_val = val_df[target_col].values.astype(np.float64)
        y_test = test_df[target_col].values.astype(np.float64)

        self.dlog.log_section("Step 3: Scaling")
        if scaler is None:
            scaler = StandardScaler()
        scaler.fit(y_train.reshape(-1, 1))
        y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
        y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()

        self.dlog.log_section("Step 4: Running Models")
        all_results = {}
        all_predictions = {}
        all_metrics = {}

        for model in models:
            result = self.run_single(model, y_train_scaled, y_test_scaled, **kwargs)

            if result.get("success"):
                raw_pred = result["predictions"]
                y_pred_inv = scaler.inverse_transform(raw_pred.reshape(-1, 1)).flatten()
                metrics_orig = self.metrics_calc.compute_all(y_test, y_pred_inv, y_train)
                result["metrics_original_scale"] = metrics_orig
                result["predictions_original_scale"] = y_pred_inv

                all_results[model.name] = result
                all_predictions[model.name] = y_pred_inv
                all_metrics[model.name] = metrics_orig

        if cv:
            self.dlog.log_section("Step 5: Cross-Validation")
            for model in models:
                try:
                    cv_result = self.run_cross_validation(
                        model, y_train_scaled,
                        n_splits=cv_splits, test_size=cv_test_size
                    )
                    all_results[f"{model.name}_cv"] = cv_result
                except Exception as e:
                    self.dlog.get_logger().error(f"CV failed for {model.name}: {e}")

        self.dlog.log_section("Step 6: Statistical Tests")
        if len(all_predictions) > 1:
            dm_results = self.stat_tests.summarize_all(y_test, all_predictions)
            all_results["statistical_tests"] = dm_results

        self.dlog.log_section("Step 7: Exports")
        self.exporter.export_metrics_comparison(all_metrics)
        self.exporter.export_forecasts(y_test, all_predictions,
                                        index=test_df.index if hasattr(test_df, 'index') else None)

        sorted_models = sorted(all_metrics.items(),
                               key=lambda x: x[1].get('rmse', float('inf')))
        best_name, best_metrics = sorted_models[0] if sorted_models else ("None", {})
        self.dlog.get_logger().info(f"{'='*50}")
        self.dlog.get_logger().info(f"BENCHMARK COMPLETE - Best model: {best_name}")
        self.dlog.get_logger().info(f"Best RMSE: {best_metrics.get('rmse', 'N/A'):.6f}")
        self.dlog.get_logger().info(f"{'='*50}")

        all_results["best_model"] = best_name
        all_results["dataset"] = dataset_loader.name

        report_data = {
            "config": {
                "dataset": dataset_loader.name,
                "target": target_col,
                "scaler": scaler.__class__.__name__,
                "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
                "n_models": len(models),
            },
            "metrics_comparison": all_metrics,
            "best_model": best_name,
        }
        self.exporter.export_comprehensive_report(report_data)

        self.results[f"benchmark_{dataset_loader.name}"] = all_results
        return all_results

    def get_summary(self) -> pd.DataFrame:
        summary_rows = []
        for name, result in self.results.items():
            if isinstance(result, dict) and "metrics" in result:
                row = {"Model": name}
                row.update(result["metrics"])
                summary_rows.append(row)
        return pd.DataFrame(summary_rows).set_index("Model") if summary_rows else pd.DataFrame()

    def print_summary(self):
        df = self.get_summary()
        if not df.empty:
            print("\n" + "=" * 80)
            print("BENCHMARK SUMMARY")
            print("=" * 80)
            print(df.to_string())
            print("=" * 80)
        else:
            self.dlog.get_logger().warning("No results to summarize")
