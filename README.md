# TSFM Latent Geometry Benchmark

Repository skeleton for your paper pipeline with 4 required blocks:

1. TSFM/backbone (`TimesFM`) + baselines (`Transformer`, `LSTM/TCN`).
2. Benchmark datasets (`ETT`, `Electricity/ECL`, `Traffic`, `Weather`, `METR-LA`).
3. Latent extraction + visualization (layer-wise embeddings + UMAP).
4. Protocol perturbation + evaluation (`LCS`, `TSI`, `GSP`).

## Project layout

- `external/`: cloned upstream repos for model/data references.
- `src/models/`: TimesFM + baseline wrappers.
- `src/data/`: dataset loading registry.
- `src/latents/`: latent extraction and UMAP projection.
- `src/eval/`: perturbation protocols and metrics.
- `scripts/run_pipeline.py`: end-to-end skeleton runner.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py --dataset electricity --horizon 96
```

## Cloned external repos

All repos requested are cloned under `external/`:
- `timesfm`
- `neuralforecast`
- `ETDataset`
- `DCRNN`
- `chronos-forecasting`
- `moment`
- `TSLib`
- `BasicTS`

Use these for checkpoints, data scripts, and baseline parity.
