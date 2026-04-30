from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from chronos import ChronosPipeline

from src.eval.geometry_metrics import compute_fss, compute_ltc, compute_pds
from src.eval.metrics import compute_gsp, compute_lcs, compute_tsi
from src.eval.perturbation import apply_masking, apply_noise, apply_time_warp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External baseline geometry metrics runner")
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--datasets", type=str, required=True)
    p.add_argument("--context-length", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--max-windows", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs/metrics")
    return p.parse_args()


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _build_windows(series: np.ndarray, context_length: int, horizon: int, max_windows: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(series)
    starts = np.linspace(0, n - context_length - horizon, num=max_windows, dtype=int)
    x = np.stack([series[s : s + context_length] for s in starts], axis=0).astype(np.float32)
    y = np.stack([series[s + context_length : s + context_length + horizon] for s in starts], axis=0).astype(np.float32)
    return x, y


def _model_id(model: str) -> str | None:
    m = model.lower()
    if m == "sundial":
        return "thuml/sundial-base-128m"
    if m == "time-moe":
        return "Maple728/TimeMoE-50M"
    return None


def _load_series(dataset: str) -> np.ndarray:
    ds = dataset.lower()
    if ds == "weather":
        p = Path("data/raw/long_horizon/longhorizon/datasets/weather/M/df_y.csv")
        df = pd.read_csv(p)
        return df.groupby("ds", as_index=False)["y"].mean()["y"].astype(float).to_numpy()
    if ds == "electricity":
        p = Path("data/raw/long_horizon/longhorizon/datasets/ECL/M/df_y.csv")
        df = pd.read_csv(p)
        return df.groupby("ds", as_index=False)["y"].mean()["y"].astype(float).to_numpy()
    if ds == "traffic":
        p = Path("data/raw/long_horizon/longhorizon/datasets/traffic/M/df_y.csv")
        df = pd.read_csv(p)
        return df.groupby("ds", as_index=False)["y"].mean()["y"].astype(float).to_numpy()
    if ds == "ett":
        p = Path("external/ETDataset/ETT-small/ETTh1.csv")
        df = pd.read_csv(p)
        return pd.to_numeric(df["OT"], errors="coerce").dropna().astype(float).to_numpy()
    if ds in {"csi300", "csi800"}:
        p = Path(f"data/raw/{ds}.csv")
        if not p.exists():
            p = Path("data/raw/master_csi300_market.csv")
        df = pd.read_csv(p)
        col = "y" if "y" in df.columns else df.columns[-1]
        return pd.to_numeric(df[col], errors="coerce").dropna().astype(float).to_numpy()
    raise ValueError(f"Unsupported dataset: {dataset}")


def _extract_embeddings(model_name: str, windows: np.ndarray) -> tuple[np.ndarray, str]:
    mid = _model_id(model_name)
    m = model_name.lower()
    if m == "chronos-forecasting":
        pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu")
        contexts = [torch.tensor(row.astype(np.float32)) for row in windows]
        emb, _ = pipe.embed(contexts)
        emb_np = emb.mean(dim=1).cpu().numpy().astype(np.float32)
        return emb_np, "chronos_embed"
    if m == "visionts":
        sys.path.insert(0, "external/baselines/VisionTS")
        from visionts import VisionTS  # type: ignore

        model = VisionTS("mae_base", ckpt_dir="external/baselines/VisionTS/ckpt", load_ckpt=False).eval()
        x = torch.from_numpy(windows[:, :, None])
        model.update_config(context_len=windows.shape[1], pred_len=96, periodicity=1)
        with torch.no_grad():
            pred = model(x).squeeze(-1).cpu().numpy()
        return pred.astype(np.float32), "visionts_forecast"
    if m == "uni2ts":
        sys.path.insert(0, ".")
        sys.path.insert(0, "external/baselines/uni2ts/src")
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module  # type: ignore

        module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
        model = Moirai2Forecast(
            module=module,
            prediction_length=96,
            context_length=windows.shape[1],
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to("cpu")
        series_list = [row.astype(np.float32) for row in windows]
        with torch.no_grad():
            pred = model.predict(series_list)  # [B, Q, H]
        emb = pred[:, 4, :].astype(np.float32)
        return emb, "moirai2_forecast"
    if m == "moment":
        from momentfm import MOMENTPipeline

        model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small").to("cpu")
        x = torch.from_numpy(windows[:, :, None]).permute(0, 2, 1).float()
        mask = torch.ones((x.shape[0], x.shape[-1]), dtype=torch.float32)
        with torch.no_grad():
            out = model.embed(x_enc=x, input_mask=mask)
        emb = out.embeddings.cpu().numpy().astype(np.float32)
        return emb, "moment_embed"
    if m == "tirex":
        from tirex import load_model

        model = load_model("NX-AI/TiRex", device="cpu")
        context = [row.astype(np.float32).tolist() for row in windows]
        _, mean = model.forecast(context=context, prediction_length=96)
        emb = np.asarray(mean, dtype=np.float32)
        return emb, "tirex_forecast"
    if m == "neurips2023-one-fits-all":
        from types import SimpleNamespace

        root = Path("external/baselines/NeurIPS2023-One-Fits-All/Long-term_Forecasting")
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("onefitsall_gpt4ts", root / "models" / "GPT4TS.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("failed_to_load_onefitsall_module")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        GPT4TS = mod.GPT4TS

        cfg = SimpleNamespace(
            is_gpt=True,
            patch_size=16,
            pretrain=True,
            stride=8,
            seq_len=windows.shape[1],
            gpt_layers=3,
            d_model=768,
            pred_len=96,
            freeze=True,
        )
        net = GPT4TS(cfg, device="cpu").eval()
        x = torch.from_numpy(windows[:, :, None]).float()
        with torch.no_grad():
            y = net(x, itr=0).squeeze(-1).cpu().numpy()
        return y.astype(np.float32), "gpt4ts_forecast"
    if m == "unitime":
        from types import SimpleNamespace

        root = Path("external/baselines/UniTime")
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("unitime_model", root / "models" / "unitime.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("failed_to_load_unitime_module")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        UniTime = mod.UniTime

        args = SimpleNamespace(
            mask_rate=0.0,
            patch_len=16,
            max_token_num=16,
            max_backcast_len=96,
            max_forecast_len=96,
            logger=None,
            model_path="gpt2",
            lm_layer_num=2,
            lm_ft_type="freeze",
            ts_embed_dropout=0.0,
            dec_trans_layer_num=1,
            dec_head_dropout=0.0,
        )
        net = UniTime(args).eval()
        x = torch.from_numpy(windows[:, :, None]).float()
        mask = torch.ones_like(x)
        info = ("custom", windows.shape[1], 16, "")
        with torch.no_grad():
            y = net(info, x, mask).squeeze(-1).cpu().numpy()
        return y.astype(np.float32), "unitime_forecast"
    if m == "lightgts":
        sys.path.insert(0, "external/baselines/LightGTS")
        for name in list(sys.modules.keys()):
            if name == "src" or name.startswith("src."):
                del sys.modules[name]
        importlib.invalidate_caches()
        mod = importlib.import_module("src.models.LightGTS")
        LightGTS = mod.LightGTS

        patch_len = 16
        stride = 8
        num_patch = (windows.shape[1] - patch_len) // stride + 1
        model = LightGTS(
            c_in=1,
            target_dim=96,
            patch_len=patch_len,
            stride=stride,
            num_patch=num_patch,
            head_type="prediction",
            d_model=128,
            n_heads=8,
            e_layers=2,
            d_layers=1,
            d_ff=256,
            dropout=0.0,
            attn_dropout=0.0,
        ).eval()
        x = torch.from_numpy(windows[:, :, None]).float().permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=patch_len, step=stride).permute(0, 2, 1, 3).contiguous()
        with torch.no_grad():
            y = model(x).squeeze(-1).cpu().numpy()
        return y.astype(np.float32), "lightgts_forecast"
    if mid is None:
        # fallback deterministic embedding for non-HF adapters
        emb = np.fft.rfft(windows, axis=1).real[:, :64]
        return emb.astype(np.float32), "fallback_fft"

    device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True).to(device).eval()
    x = torch.from_numpy(windows).to(device)
    with torch.no_grad():
        out = model(input_ids=x, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[-1].detach().cpu().numpy()
    emb = hidden.mean(axis=1).astype(np.float32)
    return emb, "hf_hidden_state"


def _compute_rows(dataset: str, model: str, emb: np.ndarray, y: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    labels = (np.arange(len(emb)) % 8).astype(int)
    noisy = apply_noise(emb[:, None, :]).squeeze(1)
    masked = apply_masking(emb[:, None, :]).squeeze(1)
    warped = apply_time_warp(emb[:, None, :]).squeeze(1)

    lcs_clean = compute_lcs(emb, labels)
    lcs_noise = compute_lcs(noisy, labels)
    lcs_mask = compute_lcs(masked, labels)
    lcs_warp = compute_lcs(warped, labels)
    rows.extend(
        [
            {"dataset": dataset, "model": model, "metric": "LCS_clean", "value": float(lcs_clean)},
            {"dataset": dataset, "model": model, "metric": "LCS_noise", "value": float(lcs_noise)},
            {"dataset": dataset, "model": model, "metric": "LCS_mask", "value": float(lcs_mask)},
            {"dataset": dataset, "model": model, "metric": "LCS_warp", "value": float(lcs_warp)},
            {"dataset": dataset, "model": model, "metric": "TSI_noise", "value": float(compute_tsi(emb, noisy))},
            {"dataset": dataset, "model": model, "metric": "TSI_mask", "value": float(compute_tsi(emb, masked))},
            {"dataset": dataset, "model": model, "metric": "TSI_warp", "value": float(compute_tsi(emb, warped))},
            {"dataset": dataset, "model": model, "metric": "GSP_noise", "value": float(compute_gsp(lcs_clean, lcs_noise))},
            {"dataset": dataset, "model": model, "metric": "GSP_mask", "value": float(compute_gsp(lcs_clean, lcs_mask))},
            {"dataset": dataset, "model": model, "metric": "GSP_warp", "value": float(compute_gsp(lcs_clean, lcs_warp))},
        ]
    )

    pred_proxy = emb[:, 0]
    err_proxy = np.abs(pred_proxy - y[:, 0])
    fss, _, _, _, _ = compute_fss(embeddings=emb, preds=pred_proxy, corr_type="spearman", max_pairs=20000, n_resamples=200, seed=42)
    pds, _, _, _, _ = compute_pds(embeddings=emb, errors=err_proxy, k=20, corr_type="spearman", n_resamples=200, seed=42)
    ltc_step, ltc_dir = compute_ltc(embeddings=emb, starts=np.arange(len(emb)))
    rows.extend(
        [
            {"dataset": dataset, "model": model, "metric": "FSS_spearman_h96_score", "value": float(fss)},
            {"dataset": dataset, "model": model, "metric": "PDS_spearman_mae_h96_score", "value": float(pds)},
            {"dataset": dataset, "model": model, "metric": "LTC_step", "value": float(ltc_step)},
            {"dataset": dataset, "model": model, "metric": "LTC_dir", "value": float(ltc_dir)},
        ]
    )
    return rows


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    models = _split_csv(args.models)
    datasets = _split_csv(args.datasets)
    all_rows: list[dict] = []
    status_rows: list[dict] = []

    for ds_name in datasets:
        series = _load_series(ds_name)
        x, y = _build_windows(series, args.context_length, args.horizon, args.max_windows)
        for model in models:
            try:
                emb, adapter = _extract_embeddings(model, x)
                all_rows.extend(_compute_rows(ds_name, model, emb, y))
                status_rows.append({"dataset": ds_name, "model": model, "status": "ok", "adapter": adapter})
            except Exception as e:
                status_rows.append({"dataset": ds_name, "model": model, "status": "failed", "adapter": "none", "error": str(e)[:400]})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(all_rows)
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=["dataset", "model", "metric", "value"])
    summary_df = (
        metrics_df.groupby(["dataset", "model", "metric"], as_index=False)["value"]
        .agg(best="max", median="median", mean="mean", worst="min")
        if not metrics_df.empty
        else pd.DataFrame(columns=["dataset", "model", "metric", "best", "median", "mean", "worst"])
    )
    status_df = pd.DataFrame(status_rows)

    metrics_df.to_csv(out_dir / "external_geometry_metrics_all.csv", index=False)
    summary_df.to_csv(out_dir / "external_geometry_summary_all.csv", index=False)
    status_df.to_csv(out_dir / "external_geometry_status_all.csv", index=False)

    for ds in datasets:
        metrics_df[metrics_df["dataset"] == ds].to_csv(out_dir / f"external_geometry_metrics_{ds}.csv", index=False)
        summary_df[summary_df["dataset"] == ds].to_csv(out_dir / f"external_geometry_summary_{ds}.csv", index=False)
        status_df[status_df["dataset"] == ds].to_csv(out_dir / f"external_geometry_status_{ds}.csv", index=False)


if __name__ == "__main__":
    main()
