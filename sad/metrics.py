"""Scoring. All sequences are inclusive [start, end] index pairs in test space."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def to_mask(seqs, n):
    m = np.zeros(n, dtype=bool)
    for a, b in seqs:
        m[a:b + 1] = True
    return m


def channel_counts(pred, true, n):
    """Counts for one channel under three protocols.

    strict (sequence-level, Hundman et al.): a labeled anomaly is detected if any
      predicted sequence overlaps it; a predicted sequence overlapping no label is a FP.
    pa (point-adjusted): predicted points; any hit inside a true segment credits the
      whole segment. Known to inflate scores; reported for comparison.
    pt (point-wise): raw point overlap, no adjustment.
    """
    tp = sum(any(overlap(t, p) for p in pred) for t in true)
    fp = sum(not any(overlap(p, t) for t in true) for p in pred)
    pm, tm = to_mask(pred, n), to_mask(true, n)
    pa = pm.copy()
    for a, b in true:
        if pm[a:b + 1].any():
            pa[a:b + 1] = True
    delays = []
    for t in true:
        hit = [p for p in pred if overlap(p, t)]
        if hit:
            first = max(min(p[0] for p in hit), t[0])
            delays.append((first - t[0]) / (t[1] - t[0] + 1))
    return dict(
        tp=tp, fp=fp, fn=len(true) - tp,
        pa_tp=int((pa & tm).sum()), pa_fp=int((pa & ~tm).sum()), pa_fn=int((~pa & tm).sum()),
        pt_tp=int((pm & tm).sum()), pt_fp=int((pm & ~tm).sum()), pt_fn=int((~pm & tm).sum()),
        normal_steps=int((~tm).sum()), delays=delays)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def aggregate(rows):
    """Pool per-channel counts into the headline metrics."""
    s = lambda k: sum(r[k] for r in rows)
    out = {}
    for tag, pre in (("", ""), ("pa_", "pa_"), ("pt_", "pt_")):
        p, r, f = prf(s(pre + "tp"), s(pre + "fp"), s(pre + "fn"))
        out |= {f"{tag}precision": p, f"{tag}recall": r, f"{tag}f1": f}
    out["fa_per_10k"] = 1e4 * s("fp") / max(1, s("normal_steps"))   # strict FP sequences / 10k normal steps
    d = [x for r in rows for x in r["delays"]]
    out["median_delay"] = float(np.median(d)) if d else float("nan")
    out |= dict(tp=s("tp"), fp=s("fp"), fn=s("fn"))
    return out


def threshold_free(score, true, n):
    """AUROC / AUPRC of the point-wise score vs. the label mask (per channel)."""
    y = to_mask(true, n)
    if y.all() or not y.any():
        return float("nan"), float("nan")
    return roc_auc_score(y, score), average_precision_score(y, score)
