from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from zaki_time_series_lib.config.settings import settings
from zaki_time_series_lib.models.base import BaseTimeSeriesModel
from zaki_time_series_lib.utils.logger import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    if settings.DL_DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(settings.DL_DEVICE)


class BaseTorchWrapper(BaseTimeSeriesModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.device = get_device()
        self.model: Optional[nn.Module] = None
        self.criterion = nn.MSELoss()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.train_history: Dict[str, List[float]] = {"loss": [], "val_loss": []}
        self._input_dim: int = 1
        self._seq_len: int = settings.DEFAULT_SEQUENCE_LENGTH
        self._batch_size: int = settings.DL_BATCH_SIZE
        self._lr: float = settings.DL_LEARNING_RATE
        self._max_epochs: int = settings.DL_MAX_EPOCHS
        self._patience: int = settings.DL_EARLY_STOPPING_PATIENCE

        self.dlog.get_logger().info(f"Using device: {self.device}")

    @abstractmethod
    def _build_model(self) -> nn.Module:
        pass

    def _create_sequences(self, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y_out = [], []
        for i in range(len(y) - seq_len):
            X.append(y[i:i + seq_len])
            y_out.append(y[i + seq_len])
        return np.array(X, dtype=np.float32), np.array(y_out, dtype=np.float32)

    def fit(self, y, X=None, **kwargs):
        self.dlog.log_section(f"Fitting {self.name}")
        y = self._validate_data(y).flatten()

        self._seq_len = kwargs.get('seq_len', self._seq_len)
        self._batch_size = kwargs.get('batch_size', self._batch_size)
        self._lr = kwargs.get('lr', self._lr)
        self._max_epochs = kwargs.get('max_epochs', self._max_epochs)
        self._patience = kwargs.get('patience', self._patience)
        val_split = kwargs.get('val_split', 0.1)

        X_seq, y_seq = self._create_sequences(y, self._seq_len)
        self._input_dim = 1
        self.dlog.log_data_shape(X_seq, "Sequences")
        self.dlog.log_data_shape(y_seq, "Targets")

        X_seq = X_seq.reshape(X_seq.shape[0], self._seq_len, self._input_dim)

        n_val = int(len(X_seq) * val_split)
        n_train = len(X_seq) - n_val
        if n_val == 0:
            n_val = 1
            n_train = len(X_seq) - n_val

        X_train, y_train = X_seq[:n_train], y_seq[:n_train]
        X_val, y_val = X_seq[n_train:], y_seq[n_train:] if n_val > 0 else (X_seq[:1], y_seq[:1])

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self._batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=self._batch_size, shuffle=False
        )

        self.model = self._build_model().to(self.device)
        self.dlog.log_model_summary(self.model, (self._seq_len, self._input_dim))

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.dlog.get_logger().info(f"Starting training for up to {self._max_epochs} epochs...")
        t_range = trange(self._max_epochs, desc=f"{self.name} Training", leave=True)

        for epoch in t_range:
            self.model.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(Xb)
                loss = self.criterion(output.squeeze(), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item() * len(Xb)

            train_loss /= len(X_seq)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    output = self.model(Xb)
                    loss = self.criterion(output.squeeze(), yb)
                    val_loss += loss.item() * len(Xb)
            val_loss /= len(X_val)

            self.train_history["loss"].append(train_loss)
            self.train_history["val_loss"].append(val_loss)

            if self.scheduler:
                self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            t_range.set_postfix(loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.6f}")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.dlog.log_training_progress(epoch + 1, self._max_epochs, train_loss, val_loss, current_lr)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self._patience:
                    self.dlog.get_logger().info(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.dlog.get_logger().info(f"Restored best model (val_loss={best_val_loss:.6f})")

        self.is_fitted = True
        self._last_seq_cache = y[-self._seq_len:].copy()
        self.dlog.get_logger().info(f"Training completed. Best val_loss: {best_val_loss:.6f}")
        return self

    def predict(self, horizon, X_future=None, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        self.dlog.get_logger().info(f"{self.name} predicting horizon={horizon}")

        last_window = kwargs.get('last_window', None)
        if last_window is not None:
            current_seq = np.asarray(last_window).flatten()
            self.dlog.get_logger().info(f"Using provided last_window, min={current_seq.min():.4f}, max={current_seq.max():.4f}")
        else:
            if hasattr(self, '_last_seq_cache'):
                current_seq = self._last_seq_cache.copy()
            else:
                current_seq = np.zeros(self._seq_len)

        if X_future is not None:
            X_future = np.asarray(X_future, dtype=np.float32)
            if len(X_future.shape) == 1:
                X_future = X_future.reshape(-1, 1)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(horizon):
                x_input = torch.from_numpy(current_seq.astype(np.float32).reshape(1, self._seq_len, self._input_dim)).to(self.device)
                pred = self.model(x_input).cpu().numpy()[0, 0]
                predictions.append(pred)
                current_seq = np.roll(current_seq, -1)
                current_seq[-1] = pred

        result = np.array(predictions)
        self.dlog.get_logger().info(f"{self.name} predictions: min={result.min():.4f}, max={result.max():.4f}")
        return result
