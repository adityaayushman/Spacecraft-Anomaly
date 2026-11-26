"""Turn anomaly scores into anomaly sequences: smoothing, thresholds, grouping, pruning."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

PAD, MERGE_GAP = 5, 50


@dataclass(frozen=True)
class Config:
    span: int          # EWMA span
    method: str        # static_k | static_max | dynamic
    param: float = 0.0  # k (static_k), margin (static_max); unused for dynamic
    prune: float = 0.0  # pruning ratio p (dynamic only); 0 = off

    def __str__(self):
        return f"{self.method}(span={self.span},param={self.param},prune={self.prune})"


def grid():
    out = []
    for span in (10, 30, 60):
        out += [Config(span, "static_k", k) for k in (3, 5, 8, 12, 20)]
        out += [Config(span, "static_max", m) for m in (1.0, 1.2, 1.5, 2.0)]
        out += [Config(span, "dynamic", 0.0, p) for p in (0.0, 0.13)]
    return out


def ewma(s, span):
    return pd.Series(s).ewm(span=span, adjust=False).mean().to_numpy()


def dynamic_threshold(e, zs=np.arange(2.5, 12.0, 0.5)):
    """Simplified Hundman et al. nonparametric dynamic threshold (global, no windows).

    Choose eps = mu + z*sigma maximising the relative drop in mean and std once
    points above eps are removed, penalised by the number of flagged points and
    sequences. Uses the test scores themselves (as in the original method).
    """
    mu, sd = e.mean(), e.std()
    best, best_eps = -np.inf, np.inf
    if sd == 0:
        return best_eps
    for z in zs:
        eps = mu + z * sd
        above = e > eps
        below = e[~above]
        if above.sum() == 0 or len(below) == 0:
            continue
        n_seq = int(np.sum(np.diff(np.r_[0, above.astype(int)]) == 1))
        score = ((mu - below.mean()) / mu + (sd - below.std()) / sd) / (above.sum() + n_seq ** 2)
        if score > best:
            best, best_eps = score, eps
    return best_eps


def to_sequences(mask, pad=PAD, merge_gap=MERGE_GAP):
    """Boolean mask -> merged [start, end] sequences (inclusive)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    seqs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > merge_gap:
            seqs.append((s, p))
            s = i
        p = i
    seqs.append((s, p))
    n = len(mask)
    return [[max(0, a - pad), min(n - 1, b + pad)] for a, b in seqs]


def prune(seqs, e, eps, p):
    """Hundman et al. pruning: sort sequence peaks (plus the highest non-flagged
    value); keep sequences down to the last one whose peak is >= p (relative)
    above the next peak."""
    if not seqs or p <= 0:
        return seqs
    peaks = sorted(((e[a:b + 1].max(), (a, b)) for a, b in seqs), reverse=True)
    normal = e[e <= eps]
    vals = [v for v, _ in peaks] + [normal.max() if len(normal) else 0.0]
    last = -1
    for i in range(len(peaks)):
        if (vals[i] - vals[i + 1]) / max(vals[i], 1e-12) >= p:
            last = i
    return sorted([list(peaks[i][1]) for i in range(last + 1)])


def detect(test, cal, cfg: Config, smoothed=None):
    """Return predicted sequences [[start,end],...] in test-index space.

    `smoothed` may carry the precomputed (test_smoothed, cal_smoothed) pair for cfg.span.
    """
    s, c = smoothed if smoothed is not None else (ewma(test, cfg.span), ewma(cal, cfg.span))
    if cfg.method == "static_k":
        eps = c.mean() + cfg.param * c.std()
    elif cfg.method == "static_max":
        eps = cfg.param * c.max()
    elif cfg.method == "dynamic":
        eps = dynamic_threshold(s)
    else:
        raise ValueError(cfg.method)
    seqs = to_sequences(s > eps)
    if cfg.method == "dynamic":
        seqs = prune(seqs, s, eps, cfg.prune)
    return seqs
