#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input npy path, shape [B, L].")
    p.add_argument("--output", required=True, help="Output npy path, shape [B, H].")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--model-id", default="thuml/sundial-base-128m")
    p.add_argument("--num-samples", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x = np.load(args.input).astype(np.float32)
    seqs = torch.tensor(x, dtype=torch.float32)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval()
    with torch.no_grad():
        out = model.generate(seqs, max_new_tokens=args.horizon, num_samples=args.num_samples)
        # Sundial output shape [B, S, H]
        if out.ndim == 3:
            pred = out.mean(dim=1)
        else:
            pred = out[:, -args.horizon :]
    np.save(args.output, pred.detach().cpu().numpy().astype(np.float32))


if __name__ == "__main__":
    main()
