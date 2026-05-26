import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from zaki_time_series_lib.config.settings import settings
from zaki_time_series_lib.utils.logger import get_logger

logger = get_logger(__name__)


class ResultsExporter:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or settings.RESULTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: {self.run_dir}")

    def export_metrics(self, metrics: Dict[str, Any], filename: str = "metrics") -> Path:
        path = self.run_dir / f"{filename}.csv"
        flat = self._flatten_dict(metrics)
        df = pd.DataFrame([flat])
        df.to_csv(path, index=False)
        logger.info(f"Metrics exported to {path}")
        return path

    def export_metrics_comparison(self, all_metrics: Dict[str, Dict], filename: str = "metrics_comparison") -> Path:
        path = self.run_dir / f"{filename}.csv"
        rows = []
        for model_name, metrics in all_metrics.items():
            flat = self._flatten_dict(metrics)
            flat["Model"] = model_name
            rows.append(flat)
        df = pd.DataFrame(rows)
        df = df.set_index("Model") if "Model" in df.columns else df
        df.to_csv(path)
        logger.info(f"Metrics comparison exported to {path}")

        md_path = self.run_dir / f"{filename}.md"
        self._export_markdown_table(df, md_path, "Metrics Comparison")

        latex_path = self.run_dir / f"{filename}.tex"
        self._export_latex_table(df, latex_path, "Metrics Comparison")

        return path

    def export_forecasts(self, y_true: np.ndarray, y_pred_dict: Dict[str, np.ndarray],
                         index: Optional[pd.Index] = None, filename: str = "forecasts") -> Path:
        path = self.run_dir / f"{filename}.csv"
        data = {}
        if index is not None:
            data["timestamp"] = index
        data["actual"] = y_true
        for model_name, preds in y_pred_dict.items():
            data[f"{model_name}_pred"] = np.asarray(preds).flatten()
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        logger.info(f"Forecasts exported to {path}")
        return path

    def export_predictions(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                           index: Optional[pd.Index] = None, filename: Optional[str] = None) -> Path:
        if filename is None:
            filename = f"predictions_{model_name}"
        path = self.run_dir / f"{filename}.csv"
        data = {}
        if index is not None:
            data["timestamp"] = index
        data["actual"] = np.asarray(y_true).flatten()
        data["predicted"] = np.asarray(y_pred).flatten()
        data["residual"] = data["actual"] - data["predicted"]
        data["abs_error"] = np.abs(data["residual"])
        data["squared_error"] = data["residual"] ** 2
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        logger.info(f"Predictions for {model_name} exported to {path}")
        return path

    def export_run_summary(self, config: Dict[str, Any], metrics: Dict[str, Any],
                           model_names: List[str], filename: str = "run_summary") -> Path:
        path = self.run_dir / f"{filename}.json"
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "models_evaluated": model_names,
            "metrics": metrics,
        }
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Run summary exported to {path}")
        return path

    def export_model_params(self, model_name: str, params: Dict[str, Any], filename: Optional[str] = None) -> Path:
        if filename is None:
            filename = f"params_{model_name}"
        path = self.run_dir / f"{filename}.json"
        with open(path, 'w') as f:
            json.dump({model_name: params}, f, indent=2, default=str)
        logger.info(f"Model params for {model_name} exported to {path}")
        return path

    def export_training_history(self, model_name: str, history: Dict[str, List[float]], filename: Optional[str] = None) -> Path:
        if filename is None:
            filename = f"training_history_{model_name}"
        path = self.run_dir / f"{filename}.csv"
        df = pd.DataFrame(history)
        df.to_csv(path, index=False)
        logger.info(f"Training history for {model_name} exported to {path}")
        return path

    def export_comprehensive_report(self, all_data: Dict[str, Any], filename: str = "comprehensive_report") -> Path:
        path = self.run_dir / f"{filename}.html"
        html = self._generate_html_report(all_data)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Comprehensive HTML report exported to {path}")
        return path

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _export_markdown_table(self, df: pd.DataFrame, path: Path, title: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            try:
                f.write(df.to_markdown())
            except ImportError:
                f.write(df.to_string())
        logger.info(f"Markdown table exported to {path}")

    def _export_latex_table(self, df: pd.DataFrame, path: Path, caption: str):
        try:
            latex = df.to_latex(caption=caption, label=f"tab:{caption.lower().replace(' ', '_')}")
        except Exception:
            latex = df.to_string()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(latex)
        logger.info(f"LaTeX table exported to {path}")

    def _generate_html_report(self, all_data: Dict[str, Any]) -> str:
        parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>Time Series Benchmark Report</title>",
            "<style>body{font-family:-apple-system,sans-serif;max-width:1200px;margin:auto;padding:20px}",
            "h1{color:#2c3e50}h2{color:#34495e;border-bottom:2px solid #eee;padding-bottom:5px}",
            "table{border-collapse:collapse;width:100%;margin:10px 0}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left}",
            "th{background-color:#f5f6fa;font-weight:600}",
            "tr:nth-child(even){background-color:#f9f9f9}",
            ".metric-good{color:#27ae60}.metric-bad{color:#e74c3c}",
            ".section{margin:20px 0;padding:15px;background:#fff;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1)}",
            "</style></head><body>",
            f"<h1>Time Series Benchmark Report</h1>",
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"<p>Run ID: {self.run_id}</p>",
        ]

        if "config" in all_data:
            parts.append("<div class='section'><h2>Configuration</h2><table>")
            for k, v in all_data["config"].items():
                parts.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
            parts.append("</table></div>")

        if "metrics_comparison" in all_data:
            parts.append("<div class='section'><h2>Metrics Comparison</h2>")
            df = pd.DataFrame(all_data["metrics_comparison"])
            parts.append(df.to_html())
            parts.append("</div>")

        if "best_model" in all_data:
            parts.append(f"<div class='section'><h2>Best Model</h2>")
            parts.append(f"<p>Overall best: <strong>{all_data['best_model']}</strong></p>")
            parts.append("</div>")

        parts.append("</body></html>")
        return "\n".join(parts)
