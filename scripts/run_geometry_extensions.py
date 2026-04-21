from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import TimesFmModelForPrediction

from src.data.datasets import load_dataset
from src.eval.geometry_metrics import (
    compute_css,
    compute_fss,
    compute_mcs,
    compute_pds,
    primitive_labels_from_stl_features,
)
from src.eval.regime_labels import stl_regime_labels
from src.latents.visualize import project_umap, save_scatter
from src.models.baselines import LSTMRegressor, TCNRegressor, TransformerRegressor
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Complete missing geometry validations.")
    p.add_argument("--dataset", type=str, default="electricity")
    p.add_argument("--model-id", type=str, default="google/timesfm-2.0-500m-pytorch")
    p.add_argument("--context-length", type=int, default=64)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--max-windows", type=int, default=512)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-resamples", type=int, default=200)
    p.add_argument("--fss-max-pairs", type=int, default=20000)
    p.add_argument("--css-permutations", type=int, default=60)
    return p.parse_args()


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_windows(series: np.ndarray, context_length: int, horizon: int, max_windows: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(series)
    starts = np.linspace(0, n - context_length - horizon, num=max_windows, dtype=int)
    x = np.stack([series[s : s + context_length] for s in starts], axis=0).astype(np.float32)
    y = np.stack([series[s + context_length : s + context_length + horizon] for s in starts], axis=0).astype(np.float32)
    return x, y


def tsfm_embeddings_and_forecast(
    model: TimesFmModelForPrediction,
    x_norm: np.ndarray,
    horizon: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    pv = torch.from_numpy(x_norm).to(device)
    freq = torch.zeros((pv.shape[0],), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(past_values=pv, freq=freq, output_hidden_states=True)
    layer_states = [h.detach().float().cpu().numpy() for h in out.hidden_states]
    seq_emb = layer_states[-1].mean(axis=1)

    chunks = []
    cur = pv
    while sum(c.shape[1] for c in chunks) < horizon:
        with torch.no_grad():
            f = model(past_values=cur, freq=freq, output_hidden_states=False)
        p = f.mean_predictions.detach().float().cpu().numpy()
        chunks.append(p)
        cur_np = np.concatenate([cur.detach().cpu().numpy(), p], axis=1).astype(np.float32)
        cur = torch.from_numpy(cur_np).to(device)
    pred = np.concatenate(chunks, axis=1)[:, :horizon]
    return seq_emb, pred, [ls.mean(axis=1) for ls in layer_states]


def _train_simple(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    epochs: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    xb = torch.from_numpy(x_train).to(device)
    yb = torch.from_numpy(y_train[:, None]).to(device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(xb).float()
        loss = loss_fn(pred, yb.float())
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        xte = torch.from_numpy(x_test).to(device)
        out = model(xte).float().cpu().numpy().reshape(-1)
        if isinstance(model, LSTMRegressor):
            hid, _ = model.lstm(xte)
            emb = hid[:, -1, :].detach().cpu().numpy()
        elif isinstance(model, TCNRegressor):
            z = model.net(xte.transpose(1, 2))
            emb = z[:, :, -1].detach().cpu().numpy()
        else:
            z = model.encoder(model.proj(xte))
            emb = z[:, -1, :].detach().cpu().numpy()
    return out, emb


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()

    ds = load_dataset(args.dataset)
    series = ds.frame[ds.target_col].astype(float).to_numpy()
    x, y = build_windows(series, args.context_length, args.horizon, args.max_windows)
    mu = x.mean(axis=1, keepdims=True)
    sigma = x.std(axis=1, keepdims=True) + 1e-6
    x_norm = (x - mu) / sigma

    # TSFM
    tsfm = TimesFmModelForPrediction.from_pretrained(args.model_id).to(device).eval()
    tsfm_emb, tsfm_pred_norm, layer_embs = tsfm_embeddings_and_forecast(tsfm, x_norm, args.horizon, device)
    tsfm_pred = tsfm_pred_norm * sigma + mu
    tsfm_err = np.abs(tsfm_pred[:, 0] - y[:, 0])

    metrics_rows = []
    out_fig = Path("outputs/figures")
    out_fig.mkdir(parents=True, exist_ok=True)
    out_metrics = Path("outputs/metrics")
    out_metrics.mkdir(parents=True, exist_ok=True)

    # UMAP density/error visualization for PDS validation
    coords = project_umap(tsfm_emb, random_state=args.seed)
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=21, metric="euclidean").fit(tsfm_emb)
    d, _ = nbrs.kneighbors(tsfm_emb)
    density = d[:, 1:].mean(axis=1)
    save_scatter(coords, density, out_fig / f"figure_pds_density_umap_{args.dataset}.png", title=f"PDS density map ({args.dataset})")
    save_scatter(coords, tsfm_err, out_fig / f"figure_pds_error_umap_{args.dataset}.png", title=f"PDS error map ({args.dataset})")

    # OOD protocol for MCS/PDS: shifted windows via additive trend + stronger noise
    split = int(0.8 * len(tsfm_emb))
    train_emb, test_emb = tsfm_emb[:split], tsfm_emb[split:]
    in_mcs = compute_mcs(train_emb, test_emb, k=20, quantile=0.95)
    ood_x = x_norm[split:] + 0.3 + np.linspace(0.0, 0.4, num=x_norm.shape[1], dtype=np.float32)[None, :]
    ood_emb, ood_pred_norm, _ = tsfm_embeddings_and_forecast(tsfm, ood_x.astype(np.float32), args.horizon, device)
    ood_mcs = compute_mcs(train_emb, ood_emb, k=20, quantile=0.95)
    ood_pred = ood_pred_norm * sigma[split:] + mu[split:]
    ood_err = np.abs(ood_pred[:, 0] - y[split:, 0])
    pds_in, _, _, p_in, _ = compute_pds(test_emb, tsfm_err[split:], k=20, corr_type="spearman", n_resamples=args.bootstrap_resamples, seed=args.seed)
    pds_ood, _, _, p_ood, _ = compute_pds(ood_emb, ood_err, k=20, corr_type="spearman", n_resamples=args.bootstrap_resamples, seed=args.seed + 1)
    metrics_rows.extend(
        [
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "MCS_in", "value": in_mcs},
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "MCS_ood", "value": ood_mcs},
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "PDS_in_spearman", "value": pds_in},
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "PDS_ood_spearman", "value": pds_ood},
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "PDS_in_pvalue", "value": p_in},
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "PDS_ood_pvalue", "value": p_ood},
        ]
    )

    # Layer-wise CSS profile
    _, stl_feats = stl_regime_labels(x, period=24, n_clusters=8, random_state=args.seed)
    primitive_labels = primitive_labels_from_stl_features(stl_feats)
    idxs = [0, len(layer_embs) // 4, len(layer_embs) // 2, len(layer_embs) - 1]
    idxs = sorted(set(idxs))
    for li in idxs:
        e = layer_embs[li]
        for pname, py in primitive_labels.items():
            bal_acc, margin, pval = compute_css(e, py, seed=args.seed + li, n_permutations=args.css_permutations)
            metrics_rows.extend(
                [
                    {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": f"CSS_layer{li}_{pname}_bal_acc", "value": bal_acc},
                    {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": f"CSS_layer{li}_{pname}_margin", "value": margin},
                    {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": f"CSS_layer{li}_{pname}_pvalue", "value": pval},
                ]
            )

    # Direct TSFM vs baselines on FSS/PDS/LTC using holdout split
    x_in = x_norm[:, :, None]
    y1 = y[:, 0]
    tr = int(0.8 * len(x_in))
    xtr, xte = x_in[:tr], x_in[tr:]
    ytr, yte = y1[:tr], y1[tr:]
    baseline_defs = {
        "lstm": LSTMRegressor(),
        "tcn": TCNRegressor(),
        "transformer": TransformerRegressor(),
    }

    # TSFM as reference on same holdout
    tsfm_fss, _, _, tsfm_fss_p, _ = compute_fss(tsfm_emb[tr:], tsfm_pred[tr:, 0], "spearman", args.fss_max_pairs, args.bootstrap_resamples, args.seed)
    tsfm_pds, _, _, tsfm_pds_p, _ = compute_pds(tsfm_emb[tr:], tsfm_err[tr:], 20, "spearman", args.bootstrap_resamples, args.seed)
    # approximate LTC from holdout sequence order
    from src.eval.geometry_metrics import compute_ltc

    tsfm_ltc, tsfm_ltc_dir = compute_ltc(tsfm_emb[tr:], np.arange(len(tsfm_emb[tr:])))
    metrics_rows.extend(
        [
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "FSS_baseline_cmp", "value": tsfm_fss},
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "PDS_baseline_cmp", "value": tsfm_pds},
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "LTC_step_baseline_cmp", "value": tsfm_ltc},
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "LTC_dir_baseline_cmp", "value": tsfm_ltc_dir},
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "FSS_baseline_cmp_pvalue", "value": tsfm_fss_p},
            {"dataset": args.dataset, "model": "tsfm_pretrained", "metric": "PDS_baseline_cmp_pvalue", "value": tsfm_pds_p},
        ]
    )

    for name, model in baseline_defs.items():
        pred, emb = _train_simple(model, xtr.astype(np.float32), ytr.astype(np.float32), xte.astype(np.float32), args.epochs, device)
        err = np.abs(pred - yte)
        fss, _, _, fss_p, _ = compute_fss(emb, pred, "spearman", args.fss_max_pairs, args.bootstrap_resamples, args.seed)
        pds, _, _, pds_p, _ = compute_pds(emb, err, 20, "spearman", args.bootstrap_resamples, args.seed)
        ltc, ltc_dir = compute_ltc(emb, np.arange(len(emb)))
        metrics_rows.extend(
            [
                {"dataset": args.dataset, "model": name, "metric": "FSS_baseline_cmp", "value": fss},
                {"dataset": args.dataset, "model": name, "metric": "PDS_baseline_cmp", "value": pds},
                {"dataset": args.dataset, "model": name, "metric": "LTC_step_baseline_cmp", "value": ltc},
                {"dataset": args.dataset, "model": name, "metric": "LTC_dir_baseline_cmp", "value": ltc_dir},
                {"dataset": args.dataset, "model": name, "metric": "FSS_baseline_cmp_pvalue", "value": fss_p},
                {"dataset": args.dataset, "model": name, "metric": "PDS_baseline_cmp_pvalue", "value": pds_p},
            ]
        )

    out = pd.DataFrame(metrics_rows)
    out_file = out_metrics / f"timesfm_geometry_missing_validations_{args.dataset}.csv"
    out.to_csv(out_file, index=False)
    print(f"[geometry-extensions] saved: {out_file}")


if __name__ == "__main__":
    main()

