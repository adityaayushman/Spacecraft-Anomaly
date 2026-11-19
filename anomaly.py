"""LSTM anomaly detection on NASA SMAP / MSL (Curiosity) telemetry.

One LSTM per channel learns to forecast the next telemetry value from the
previous `WINDOW` steps (value + command one-hot features). Large forecast
errors on the test split are smoothed, thresholded dynamically and grouped
into anomaly sequences, which are scored against the expert labels from
NASA JPL (labeled_anomalies.csv, Hundman et al. 2018).
"""
import argparse, ast, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA = Path(__file__).parent / "data"
WINDOW = 100
torch.set_num_threads(4)


# ---------------------------------------------------------------- data
def load_channel(cid):
    return np.load(DATA / "train" / f"{cid}.npy"), np.load(DATA / "test" / f"{cid}.npy")


def windows(x, w=WINDOW):
    """x: (T, F) -> inputs (T-w, w, F), targets (T-w,) = next value of feature 0."""
    idx = np.arange(w)[None, :] + np.arange(len(x) - w)[:, None]
    return x[idx].astype(np.float32), x[w:, 0].astype(np.float32)


# ---------------------------------------------------------------- model
class Forecaster(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def fit(train, epochs=12, seed=0):
    torch.manual_seed(seed)
    X, y = map(torch.from_numpy, windows(train))
    n_val = max(1, int(0.1 * len(X)))                       # last 10% for early stopping
    Xt, yt, Xv, yv = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
    model = Forecaster(X.shape[2])
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    best, best_state, bad = 1e9, None, 0
    for _ in range(epochs):
        model.train()
        for i in torch.randperm(len(Xt)).split(128):
            opt.zero_grad()
            nn.functional.mse_loss(model(Xt[i]), yt[i]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(Xv), yv).item()
        if v < best:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def predict(model, x):
    X, y = windows(x)
    pred = torch.cat([model(torch.from_numpy(b)) for b in np.array_split(X, max(1, len(X) // 1024))])
    return pred.numpy(), y


# ---------------------------------------------------------------- detection
def ewma(e, span=30):
    return pd.Series(e).ewm(span=span, adjust=False).mean().to_numpy()


def dynamic_threshold(e, zs=np.arange(2.5, 12.0, 0.5), min_len=1):
    """Hundman et al. nonparametric dynamic thresholding (simplified, global).

    Pick eps = mu + z*sigma maximising the relative drop in mean/std of the
    errors once points above eps are removed, penalised by how many points
    and sequences get flagged.
    """
    mu, sd = e.mean(), e.std()
    best, best_eps = -np.inf, None
    for z in zs:
        eps = mu + z * sd
        below = e[e <= eps]
        above = e > eps
        if above.sum() == 0 or len(below) == 0:
            continue
        n_seq = int(np.sum(np.diff(np.r_[0, above.astype(int)]) == 1))
        score = ((mu - below.mean()) / mu + (sd - below.std()) / sd) / (above.sum() + n_seq ** 2)
        if score > best:
            best, best_eps = score, eps
    return best_eps if best_eps is not None else np.inf


def to_sequences(mask, pad=0, merge_gap=50):
    """Boolean mask -> merged [start, end] sequences."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    seqs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > merge_gap:
            seqs.append([s, p]); s = i
        p = i
    seqs.append([s, p])
    return [[max(0, a - pad), b + pad] for a, b in seqs]


def prune(seqs, e, eps, p=0.13):
    """Drop sequences whose peak is not at least `p` (relative) above the next
    highest peak / the threshold, as in Hundman et al."""
    if not seqs:
        return seqs
    peaks = sorted([(e[a:b + 1].max(), (a, b)) for a, b in seqs], reverse=True)
    vals = [v for v, _ in peaks] + [e[e <= eps].max() if (e <= eps).any() else 0]
    keep, last_keep = [], -1
    for i in range(len(peaks)):
        if (vals[i] - vals[i + 1]) / max(vals[i], 1e-9) >= p:
            last_keep = i
    keep = [list(peaks[i][1]) for i in range(last_keep + 1)]
    return sorted(keep)


# ---------------------------------------------------------------- scoring
def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def score_sequences(pred, true):
    """Sequence-level counts (Hundman et al.): a true anomaly is detected if any
    predicted sequence overlaps it; predicted sequences with no overlap are FPs."""
    tp = sum(any(overlap(t, p) for p in pred) for t in true)
    fn = len(true) - tp
    fp = sum(not any(overlap(p, t) for t in true) for p in pred)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ---------------------------------------------------------------- pipeline
def run(limit=None, epochs=12, out=Path("results")):
    out.mkdir(exist_ok=True)
    labels = pd.read_csv(DATA / "labeled_anomalies.csv")
    labels["seqs"] = labels.anomaly_sequences.apply(ast.literal_eval)
    # P-2 appears twice in the NASA csv with different labels: merge them.
    labels = labels.groupby("chan_id", sort=False).agg(
        spacecraft=("spacecraft", "first"), seqs=("seqs", lambda s: sum(s, []))).reset_index()
    if limit:
        labels = labels.head(limit)
    rows, t0 = [], time.time()
    for k, r in labels.iterrows():
        train, test = load_channel(r.chan_id)
        model = fit(train, epochs=epochs)
        pred, y = predict(model, test)
        err = ewma(np.abs(y - pred))
        eps = dynamic_threshold(err)
        # shift back into test-index space (first WINDOW points have no forecast)
        seqs = prune(to_sequences(err > eps, pad=5), err, eps)
        seqs = [[a + WINDOW, b + WINDOW] for a, b in seqs]
        tp, fp, fn = score_sequences(seqs, r.seqs)
        rows.append(dict(chan_id=r.chan_id, spacecraft=r.spacecraft, n_true=len(r.seqs),
                         n_pred=len(seqs), tp=tp, fp=fp, fn=fn))
        np.savez_compressed(out / f"{r.chan_id}.npz", err=err, eps=eps, pred=pred, y=y,
                            true=np.array(r.seqs), found=np.array(seqs).reshape(-1, 2))
        print(f"[{k + 1}/{len(labels)}] {r.chan_id:5s} {r.spacecraft:4s} true={len(r.seqs)} "
              f"pred={len(seqs)} tp={tp} fp={fp} fn={fn}  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_channel.csv", index=False)
    summary = {}
    for name, d in [("ALL", df)] + [(s, g) for s, g in df.groupby("spacecraft")]:
        p, rc, f1 = prf(d.tp.sum(), d.fp.sum(), d.fn.sum())
        summary[name] = dict(channels=len(d), true_anomalies=int(d.n_true.sum()), tp=int(d.tp.sum()),
                             fp=int(d.fp.sum()), fn=int(d.fn.sum()),
                             precision=round(p, 3), recall=round(rc, 3), f1=round(f1, 3))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    run(a.limit, a.epochs)
