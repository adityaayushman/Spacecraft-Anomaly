"""Train one model (one seed) on every channel and save its anomaly scores.

python -m sad.run_models --model lstm --seed 0 [--limit N] [--threads 2]
Resumable: channels whose score file already exists are skipped.
"""
import argparse
import time

import numpy as np
import torch

from .data import ROOT, load_channel, load_labels
from .models import MODELS, run_channel

SCORES = ROOT / "results" / "scores"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th channel starting at i")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = SCORES / a.model / f"s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    chans = load_labels().chan_id.tolist()[: a.limit]
    i, n = map(int, a.shard.split("/"))
    chans = chans[i::n]
    t0 = time.time()
    for i, c in enumerate(chans):
        f = out / f"{c}.npz"
        if f.exists():
            continue
        train, test = load_channel(c)
        te, cal = run_channel(a.model, train, test, a.seed)
        np.savez_compressed(f, test=te, cal=cal)
        print(f"[{a.model} s{a.seed}] {i + 1}/{len(chans)} {c} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
