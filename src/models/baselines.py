from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.baseline_config import BaselineTrainConfig


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TCNRegressor(nn.Module):
    def __init__(self, channels: int = 128, kernel_size: int = 3, num_blocks: int = 3):
        super().__init__()
        pad = kernel_size // 2
        blocks = []
        c_in = 1
        for _ in range(num_blocks):
            blocks.extend(
                [
                    nn.Conv1d(c_in, channels, kernel_size=kernel_size, padding=pad),
                    nn.ReLU(),
                ]
            )
            c_in = channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2))
        z = z[:, :, -1]
        return self.head(z)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.encoder = nn.TransformerEncoder(
                    enc_layer, num_layers=num_layers, enable_nested_tensor=False
                )
            except TypeError:
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.encoder(z)
        return self.head(z[:, -1, :])


@dataclass
class BaselineResult:
    model_name: str
    fold: int
    mae: float
    rmse: float


def _autocast_ctx(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "mps":
        try:
            return torch.autocast(device_type="mps", dtype=torch.float16)
        except Exception:
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_supervised(series: np.ndarray, context_length: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    x_data, y_data = [], []
    for i in range(context_length, len(series) - horizon + 1):
        x_data.append(series[i - context_length : i])
        y_data.append(series[i + horizon - 1])
    x = np.asarray(x_data, dtype=np.float32)
    y = np.asarray(y_data, dtype=np.float32)
    return x[:, :, None], y[:, None]


def _normalize_fold(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    x_mean = float(train_x.mean())
    x_std = float(train_x.std()) + 1e-6
    y_mean = float(train_y.mean())
    y_std = float(train_y.std()) + 1e-6
    train_x_n = (train_x - x_mean) / x_std
    test_x_n = (test_x - x_mean) / x_std
    train_y_n = (train_y - y_mean) / y_std
    return train_x_n, train_y_n, test_x_n, y_mean, y_std, x_mean, x_std


def _fit_predict(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    model_name: str,
    fold: int,
    device: torch.device,
    train_cfg: BaselineTrainConfig,
    verbose: bool = True,
) -> np.ndarray:
    train_x_n, train_y_n, test_x_n, y_mean, y_std, _, _ = _normalize_fold(train_x, train_y.reshape(-1, 1), test_x)

    model = model.to(device)
    if train_cfg.compile_torch and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
        except Exception:
            pass
    model.train()

    batch_size = min(train_cfg.batch_size, len(train_x_n))
    if batch_size < 1:
        batch_size = 1

    pin = train_cfg.pin_memory and device.type == "cuda"
    nw = max(0, train_cfg.num_workers)
    loader_kw: Dict[str, object] = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=nw,
        pin_memory=pin,
    )
    if nw > 0:
        loader_kw["persistent_workers"] = True
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x_n), torch.from_numpy(train_y_n)),
        **loader_kw,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    loss_fn = nn.MSELoss()

    scheduler = None
    if train_cfg.use_scheduler and len(loader) > 0:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_cfg.lr,
            epochs=train_cfg.epochs,
            steps_per_epoch=max(1, len(loader)),
            pct_start=train_cfg.scheduler_pct_start,
        )

    use_amp = train_cfg.use_amp and device.type in ("cuda", "mps")
    if verbose:
        print(
            f"[train] fold={fold} model={model_name} device={device} "
            f"train={len(train_x)} test={len(test_x)} epochs={train_cfg.epochs} "
            f"lr={train_cfg.lr} batch={batch_size} amp={use_amp} workers={nw}"
        )

    log_every = max(1, train_cfg.log_every)
    for epoch in range(1, train_cfg.epochs + 1):
        epoch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=pin), yb.to(device, non_blocking=pin)
            optimizer.zero_grad()
            with _autocast_ctx(device, use_amp):
                pred = model(xb)
                loss = loss_fn(pred.float(), yb.float())
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_losses.append(float(loss.item()))

        if verbose and (epoch == 1 or epoch == train_cfg.epochs or epoch % log_every == 0):
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            print(
                f"[train] fold={fold} model={model_name} "
                f"epoch={epoch}/{train_cfg.epochs} loss={avg_loss:.6f}"
            )

    model.eval()
    preds_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(test_x_n), batch_size):
            end = start + batch_size
            xb = torch.from_numpy(test_x_n[start:end]).to(device, non_blocking=pin)
            with _autocast_ctx(device, use_amp):
                part = model(xb).float().cpu().numpy().reshape(-1)
            preds_parts.append(part)
    pred_norm = np.concatenate(preds_parts, axis=0)
    return pred_norm * y_std + y_mean


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


def _make_builders(train_cfg: BaselineTrainConfig) -> Dict[str, Callable[[], nn.Module]]:
    return {
        "transformer": lambda: TransformerRegressor(
            d_model=train_cfg.transformer_d_model,
            nhead=train_cfg.transformer_nhead,
            num_layers=train_cfg.transformer_layers,
            dropout=train_cfg.transformer_dropout,
        ),
        "lstm": lambda: LSTMRegressor(
            hidden_size=train_cfg.lstm_hidden,
            num_layers=train_cfg.lstm_layers,
        ),
        "tcn": lambda: TCNRegressor(
            channels=train_cfg.tcn_channels,
            kernel_size=train_cfg.tcn_kernel,
            num_blocks=train_cfg.tcn_blocks,
        ),
    }


def run_baselines_walk_forward(
    series: np.ndarray,
    context_length: int,
    horizon: int,
    n_splits: int = 3,
    train_cfg: BaselineTrainConfig | None = None,
    max_samples: int | None = 300_000,
    verbose: bool = True,
) -> Tuple[list[BaselineResult], Dict[str, np.ndarray]]:
    if train_cfg is None:
        train_cfg = BaselineTrainConfig()

    x, y = build_supervised(series, context_length=context_length, horizon=horizon)
    if max_samples is not None and len(x) > max_samples:
        if verbose:
            print(f"[baseline] downsample windows: {len(x)} -> {max_samples}")
        keep_idx = np.linspace(0, len(x) - 1, num=max_samples, dtype=int)
        x = x[keep_idx]
        y = y[keep_idx]
    splitter = TimeSeriesSplit(n_splits=n_splits)
    device = resolve_torch_device()

    if verbose:
        print(
            f"[baseline] device={device} samples={len(x)} "
            f"context_length={context_length} horizon={horizon} splits={n_splits} "
            f"epochs={train_cfg.epochs} hidden~{train_cfg.lstm_hidden}/{train_cfg.transformer_d_model}"
        )

    results: list[BaselineResult] = []
    latent_bank: Dict[str, list[np.ndarray]] = {"transformer": [], "lstm": [], "tcn": []}
    builders = _make_builders(train_cfg)

    for fold, (tr_idx, te_idx) in enumerate(splitter.split(x), start=1):
        tr_x, te_x = x[tr_idx], x[te_idx]
        tr_y, te_y = y[tr_idx].reshape(-1), y[te_idx].reshape(-1)
        if verbose:
            print(f"[baseline] starting fold={fold}/{n_splits}")

        for name, builder in builders.items():
            model = builder()
            pred = _fit_predict(
                model,
                tr_x,
                tr_y,
                te_x,
                model_name=name,
                fold=fold,
                device=device,
                train_cfg=train_cfg,
                verbose=verbose,
            )
            mae, rmse = _metrics(te_y, pred)
            results.append(BaselineResult(model_name=name, fold=fold, mae=mae, rmse=rmse))
            latent_bank[name].append(te_x.reshape(te_x.shape[0], -1))
            if verbose:
                print(f"[baseline] fold={fold} model={name} mae={mae:.6f} rmse={rmse:.6f}")

    merged_latents = {k: np.concatenate(v, axis=0) if v else np.zeros((0, context_length)) for k, v in latent_bank.items()}
    return results, merged_latents
