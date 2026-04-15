from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    dataset: str = "electricity"
    horizon: int = 96
    context_length: int = 96
    batch_size: int = 256
    random_seed: int = 42
    timesfm_model_id: str = "google/timesfm-2.0-500m-pytorch"
    output_dir: Path = Path("outputs")
