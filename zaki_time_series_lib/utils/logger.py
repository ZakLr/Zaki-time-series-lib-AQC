import logging
import sys
from pathlib import Path
from typing import Optional

from zaki_time_series_lib.config.settings import settings


class DetailedLogger:
    def __init__(self, name: str, level: Optional[str] = None, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)

        log_level = (level or settings.LOG_LEVEL).upper()
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))

        if self.logger.handlers:
            self.logger.handlers.clear()

        formatter = logging.Formatter(settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        log_path = log_file or settings.LOG_FILE
        if log_path:
            fh = logging.FileHandler(log_path, mode='a')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def get_logger(self) -> logging.Logger:
        return self.logger

    def log_data_shape(self, data, name: str = "data"):
        if hasattr(data, 'shape'):
            self.logger.info(f"{name} shape: {data.shape}")
        elif hasattr(data, '__len__'):
            self.logger.info(f"{name} length: {len(data)}")

    def log_data_stats(self, data, name: str = "data"):
        import numpy as np
        arr = np.asarray(data)
        self.logger.info(f"{name} stats -> min: {arr.min():.6f}, max: {arr.max():.6f}, "
                         f"mean: {arr.mean():.6f}, std: {arr.std():.6f}, "
                         f"NaN count: {int(np.isnan(arr).sum())}, Inf count: {int(np.isinf(arr).sum())}")

    def log_section(self, title: str, char: str = "="):
        self.logger.info(f"{char * 20} {title} {char * 20}")

    def log_subsection(self, title: str, char: str = "-"):
        self.logger.info(f"{char * 10} {title} {char * 10}")

    def log_dict(self, d: dict, title: str = "Configuration"):
        self.log_section(title)
        for k, v in d.items():
            self.logger.info(f"  {k}: {v}")
        self.log_section(f"End {title}")

    def log_model_summary(self, model, input_shape=None):
        self.log_section("Model Summary")
        self.logger.info(f"Model type: {type(model).__name__}")
        try:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Total parameters: {total:,}")
            self.logger.info(f"Trainable parameters: {trainable:,}")
        except Exception:
            pass
        if input_shape:
            self.logger.info(f"Expected input shape: {input_shape}")

    def log_training_progress(self, epoch, max_epochs, loss, val_loss=None, lr=None, **extra):
        msg = f"Epoch [{epoch:>4d}/{max_epochs}] | Loss: {loss:.6f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.6f}"
        if lr is not None:
            msg += f" | LR: {lr:.6e}"
        for k, v in extra.items():
            msg += f" | {k}: {v:.6f}"
        self.logger.info(msg)


_loggers = {}

def get_logger(name: str = __name__) -> logging.Logger:
    if name not in _loggers:
        dl = DetailedLogger(name)
        _loggers[name] = dl.get_logger()
    return _loggers[name]
