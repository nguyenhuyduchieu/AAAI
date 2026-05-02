# Shared UMAP Control Report (weather)

## LTC
- `ltc_pretrained`: 9.876444
- `ltc_random_reset`: 9.003154
- `delta_pre_minus_random`: 0.873290
- `expectation_pre_gt_random`: PASS

## Fairness Checklist
- Same architecture: PASS (TimesFM 2.0 model class for both branches)
- Same dataset: PASS
- Same windowing: PASS
- Same preprocessing (z-normalization per window): PASS
- Same seed/projection: PASS (single shared UMAP reducer)
- Only weights differ: PASS (pretrained vs random-reset)

## Methodology Guardrails
- No separate `fit_transform` for random branch: PASS
- Random branch uses `reducer.transform(...)`: PASS
- Same data order and temporal coloring between branches: PASS

## Artifacts
- Figure: `outputs/figures/figure_timesfm_umap_shared_weather_pre_vs_random.png`
- Metrics CSV: `outputs/metrics/timesfm_shared_umap_metrics_weather.csv`
- Coordinates CSV: `outputs/metrics/timesfm_shared_umap_coords_weather.csv`