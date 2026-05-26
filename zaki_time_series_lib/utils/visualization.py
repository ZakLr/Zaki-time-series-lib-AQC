import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from zaki_time_series_lib.config.settings import settings
from zaki_time_series_lib.utils.logger import get_logger

logger = get_logger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100


class TimeSeriesVisualizer:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or settings.RESULTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_series(self, data: Union[pd.Series, pd.DataFrame],
                    title: str = "Time Series", filename: Optional[str] = None,
                    labels: Optional[List[str]] = None, show: bool = False) -> Path:
        if filename is None:
            filename = f"series_{title.lower().replace(' ', '_')}.png"
        path = self.output_dir / filename

        fig, ax = plt.subplots()
        if isinstance(data, pd.Series):
            data.plot(ax=ax, label=labels[0] if labels else None)
        else:
            data.plot(ax=ax, label=labels)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"Plot saved to {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_forecast(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str = "Model", title: Optional[str] = None,
                      index: Optional[pd.Index] = None, filename: Optional[str] = None,
                      show: bool = False) -> Path:
        if filename is None:
            filename = f"forecast_{model_name}.png"
        path = self.output_dir / filename
        if title is None:
            title = f"Forecast - {model_name}"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        x = index if index is not None else np.arange(len(y_true))

        ax1.plot(x, y_true, label='Actual', color='#2c3e50', linewidth=1.5)
        ax1.plot(x, y_pred, label=f'{model_name} Forecast', color='#e74c3c',
                 linewidth=1.5, linestyle='--')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        residuals = np.asarray(y_true).flatten() - np.asarray(y_pred).flatten()
        ax2.plot(x, residuals, color='#8e44ad', linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.fill_between(x, residuals, 0, alpha=0.3,
                         where=(residuals >= 0), color='red', label='Overpredict')
        ax2.fill_between(x, residuals, 0, alpha=0.3,
                         where=(residuals < 0), color='blue', label='Underpredict')
        ax2.set_title("Residuals", fontsize=12)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Residual")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"Forecast plot saved to {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_multi_forecast(self, y_true: np.ndarray,
                            predictions: Dict[str, np.ndarray],
                            title: str = "Multi-Model Forecast Comparison",
                            index: Optional[pd.Index] = None,
                            filename: str = "multi_forecast_comparison.png",
                            show: bool = False) -> Path:
        path = self.output_dir / filename
        x = index if index is not None else np.arange(len(y_true))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

        ax1.plot(x, y_true, label='Actual', color='black', linewidth=2)
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        for (name, pred), color in zip(predictions.items(), colors):
            ax1.plot(x, np.asarray(pred).flatten(), label=name,
                     linewidth=1.2, linestyle='--', color=color, alpha=0.8)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        errors = {}
        for name, pred in predictions.items():
            errors[name] = np.abs(np.asarray(y_true).flatten() - np.asarray(pred).flatten())
        error_df = pd.DataFrame(errors, index=x)
        error_df.plot(ax=ax2, alpha=0.8, linewidth=1)
        ax2.set_title("Absolute Errors by Model", fontsize=12)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Absolute Error")
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"Multi-model forecast plot saved to {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                                metric_name: str = "rmse",
                                title: Optional[str] = None,
                                filename: str = "metrics_comparison.png",
                                show: bool = False) -> Path:
        path = self.output_dir / filename
        if title is None:
            title = f"{metric_name.upper()} Comparison"

        models = list(metrics_dict.keys())
        values = [metrics_dict[m].get(metric_name, 0) for m in models]

        fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.2), 6))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
        bars = ax.bar(models, values, color=colors, edgecolor='gray', linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, rotation=45)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name.upper())
        ax.set_xlabel("Model")
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"Metrics comparison plot saved to {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_training_history(self, history: Dict[str, List[float]],
                               model_name: str = "Model",
                               filename: Optional[str] = None,
                               show: bool = False) -> Path:
        if filename is None:
            filename = f"training_history_{model_name}.png"
        path = self.output_dir / filename

        fig, ax = plt.subplots()
        for key, values in history.items():
            ax.plot(values, label=key, linewidth=1.5)
        ax.set_title(f"Training History - {model_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"Training history plot saved to {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "Model",
                                filename: Optional[str] = None,
                                show: bool = False) -> Path:
        if filename is None:
            filename = f"residual_analysis_{model_name}.png"
        path = self.output_dir / filename

        residuals = np.asarray(y_true).flatten() - np.asarray(y_pred).flatten()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title("Residuals vs Fitted")
        axes[0, 0].set_xlabel("Fitted values")
        axes[0, 0].set_ylabel("Residuals")

        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Q-Q Plot")

        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title("Residual Distribution")
        axes[1, 0].set_xlabel("Residual")
        axes[1, 0].set_ylabel("Frequency")

        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, ax=axes[1, 1], lags=40, alpha=0.05)
        axes[1, 1].set_title("Residual ACF")

        plt.suptitle(f"Residual Analysis - {model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"Residual analysis plot saved to {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_seasonal_decomposition(self, data: pd.Series, period: int = 24,
                                     model: str = 'additive', filename: Optional[str] = None,
                                     show: bool = False) -> Path:
        if filename is None:
            filename = f"seasonal_decomposition_p{period}.png"
        path = self.output_dir / filename

        from statsmodels.tsa.seasonal import seasonal_decompose
        result = seasonal_decompose(data.dropna(), model=model, period=period)

        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        data.plot(ax=axes[0], title="Original")
        result.trend.plot(ax=axes[1], title="Trend")
        result.seasonal.plot(ax=axes[2], title="Seasonal")
        result.resid.plot(ax=axes[3], title="Residual")
        plt.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"Seasonal decomposition saved to {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_correlation_heatmap(self, data: pd.DataFrame,
                                  title: str = "Correlation Heatmap",
                                  filename: str = "correlation_heatmap.png",
                                  show: bool = False) -> Path:
        path = self.output_dir / filename
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path
