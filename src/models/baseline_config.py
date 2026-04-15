from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineTrainConfig:
    """Hyperparameters for PyTorch baseline training (walk-forward)."""

    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lstm_hidden: int = 128
    lstm_layers: int = 2
    tcn_channels: int = 128
    tcn_kernel: int = 3
    tcn_blocks: int = 3
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_layers: int = 3
    transformer_dropout: float = 0.1
    use_scheduler: bool = True
    scheduler_pct_start: float = 0.1
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True
    compile_torch: bool = False
    log_every: int = 5
