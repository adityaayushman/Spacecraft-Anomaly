"""Build the data slice for the 3D viewer from the study's real outputs.

python web/build_data.py   (run from the repo root, after scores exist in results/scores)
Writes web/data.json, web/artifact.html (fragment) and web/index.html (standalone page).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sad.data import fold_assignment, load_channel, load_labels  # noqa: E402
from sad.detect import Config, detect, dynamic_threshold, ewma  # noqa: E402
from sad.evaluate import load_scores  # noqa: E402
from sad.metrics import overlap  # noqa: E402

CHANNELS = ["S-1", "G-7", "T-3", "P-1", "E-13", "D-16", "T-9", "F-7"]   # mix of catches, misses, false alarms
N = 360            # bins per channel
SPAN = 30
MODEL, SEED = "gru", 0


def threshold(test, cal, cfg):
    s, c = ewma(test, cfg.span), ewma(cal, cfg.span)
    if cfg.method == "static_k":
        return s, c.mean() + cfg.param * c.std()
    if cfg.method == "static_max":
        return s, cfg.param * c.max()
    return s, dynamic_threshold(s)


def binned(x, fn):
    edges = np.linspace(0, len(x), N + 1).astype(int)
    return np.array([fn(x[a:max(b, a + 1)]) for a, b in zip(edges[:-1], edges[1:])])


def main():
    labels = load_labels().set_index("chan_id")
    folds = fold_assignment(load_labels())
    chosen = json.loads((ROOT / "results" / "chosen_configs.json").read_text())[f"{MODEL}/s{SEED}"]
    scores = load_scores(MODEL, SEED)
    out = []
    for c in CHANNELS:
        r = labels.loc[c]
        _, raw = load_channel(c)
        test, cal = scores[c]
        cfg = Config(**chosen[str(1 - folds[c])])          # tuned on the other fold
        pred = detect(test, cal, cfg)
        s, eps = threshold(test, cal, cfg)
        ratio = s / eps if np.isfinite(eps) and eps > 0 else s / s.max()
        tele = raw[:, 0]
        med = np.median(tele)
        rep = binned(tele, lambda w: w[np.argmax(np.abs(w - med))])
        rep = (rep - rep.min()) / max(np.ptp(rep), 1e-9)
        n = len(test)
        b = lambda i: int(min(N - 1, i * N / n))
        out.append(dict(
            id=c, sc=r.spacecraft, n=n, cfg=str(cfg),
            tele=[round(float(v), 3) for v in rep],
            score=[round(float(v), 3) for v in binned(ratio, np.max)],
            labels=[dict(a=b(a), b=b(e), a_s=int(a), b_s=int(e), cls=cls,
                         hit=any(overlap(p, (a, e)) for p in pred)) for (a, e), cls in zip(r.seqs, r.classes)],
            pred=[dict(a=b(a), b=b(e), a_s=int(a), b_s=int(e), fp=not any(overlap((a, e), t) for t in r.seqs))
                  for a, e in pred]))
    data = dict(model=MODEL.upper(), seed=SEED, bins=N, channels=out)
    (ROOT / "web" / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    tpl = (ROOT / "web" / "template.html").read_text()
    frag = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (ROOT / "web" / "artifact.html").write_text(frag)
    (ROOT / "web" / "index.html").write_text("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        "</head>\n<body>\n" + frag + "\n</body>\n</html>\n")
    for ch in out:
        print(ch["id"], ch["sc"], "labels", [(l["cls"][0], l["hit"]) for l in ch["labels"]],
              "pred", [(p["fp"]) for p in ch["pred"]], "eps-normalised max", max(ch["score"]))


if __name__ == "__main__":
    main()
