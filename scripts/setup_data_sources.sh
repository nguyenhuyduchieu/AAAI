#!/usr/bin/env bash
set -euo pipefail

# This script documents where each benchmark source should be fetched/prepared.
# For reproducibility, keep raw downloads in data/raw and processed artifacts in data/processed.

echo "[INFO] ETDataset source: external/ETDataset"
echo "[INFO] METR-LA source: external/DCRNN (place metr-la.h5 under external/DCRNN/data/)"
echo "[INFO] ECL/Traffic/Weather: use datasetsforecast LongHorizon loaders"
echo "[INFO] UCI Electricity original source referenced in SOURCES.md"
