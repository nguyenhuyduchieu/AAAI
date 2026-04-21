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

## Control experiments

Run the two key controls for latent-geometry claims:

```bash
python scripts/run_timesfm_demo.py \
  --dataset electricity \
  --context-length 96 \
  --max-windows 512 \
  --weight-mode both \
  --label-mode both \
  --stl-period 24 \
  --stl-clusters 8
```

- `--label-mode stl` replaces legacy `%8` regimes with STL-derived regime labels (via clustering STL features).
- `--weight-mode random-reset` runs the random-weight TimesFM control with identical architecture.
- Metrics are exported to `outputs/metrics/timesfm_control_metrics_<dataset>.csv`.

## Geometry metrics (phase 1 + phase 2)

The demo script now computes:
- Phase 1: `FSS`, `PDS`, `LTC`
- Phase 2: `LLS`, `MCS`, `CSS`, `TCS`

Example run:

```bash
PYTHONPATH=. python scripts/run_timesfm_demo.py \
  --dataset electricity \
  --weight-mode both \
  --label-mode both \
  --horizons 96,192,336,720 \
  --phase1-knn 10,20,50 \
  --bootstrap-resamples 1000 \
  --css-permutations 100
```

Outputs:
- Detailed metrics: `outputs/metrics/timesfm_geometry_metrics_<dataset>.csv`
- Summary (best/median/mean/worst): `outputs/metrics/timesfm_geometry_summary_<dataset>.csv`
- Compact paper table:

```bash
PYTHONPATH=. python scripts/export_geometry_table.py --dataset electricity
```

This writes `outputs/metrics/timesfm_geometry_table_<dataset>.csv`.

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
