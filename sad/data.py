"""Data loading for NASA SMAP / MSL telemetry (Hundman et al., KDD 2018)."""
from pathlib import Path

import ast
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_labels() -> pd.DataFrame:
    """One row per channel: chan_id, spacecraft, seqs [[start,end],...], classes [...].

    `P-2` appears twice in NASA's csv with different sequences; they are merged.
    """
    df = pd.read_csv(DATA / "labeled_anomalies.csv")
    df["seqs"] = df.anomaly_sequences.apply(ast.literal_eval)
    df["classes"] = df["class"].apply(lambda s: [c.strip() for c in s.strip("[]").split(",")])
    out = df.groupby("chan_id", sort=False).agg(
        spacecraft=("spacecraft", "first"),
        seqs=("seqs", lambda s: sum(s, [])),
        classes=("classes", lambda s: sum(s, [])),
        n=("num_values", "first"),
    ).reset_index()
    assert all(len(a) == len(b) for a, b in zip(out.seqs, out.classes))
    return out


def load_channel(cid):
    return np.load(DATA / "train" / f"{cid}.npy"), np.load(DATA / "test" / f"{cid}.npy")


def fold_assignment(labels: pd.DataFrame, seed=0) -> dict:
    """Split channels into two folds (stratified by spacecraft) for cross-fitted tuning."""
    rng = np.random.default_rng(seed)
    fold = {}
    for _, g in labels.groupby("spacecraft"):
        ids = g.chan_id.to_numpy().copy()
        rng.shuffle(ids)
        for i, c in enumerate(ids):
            fold[c] = i % 2
    return fold


def windows(x, w, stride=1):
    """x: (T,F) -> inputs (N,w,F) and next-step targets x[w:,0], aligned to index w..T-1."""
    starts = np.arange(0, len(x) - w, stride)
    idx = starts[:, None] + np.arange(w)[None, :]
    return x[idx].astype(np.float32), x[starts + w, 0].astype(np.float32)
