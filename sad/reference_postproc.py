"""Score every model's saved anomaly scores with the Hundman et al. default post-processing
(EWMA span 105, windowed dynamic threshold, 100-step error buffer, pruning p=0.13).
Parameters are fixed to the authors' defaults, so nothing is tuned on any labels.

python -m sad.reference_postproc
"""
import numpy as np
import pandas as pd

from .data import load_labels
from .detect import Config, detect, ewma
from .evaluate import MAIN_MODELS, OUT, load_scores, seeds_for
from .metrics import aggregate, channel_counts, overlap

CFG = Config(span=105, method="telemanom", param=0.0, prune=0.13)


def main():
    labels = load_labels()
    rows, lines = [], []
    for model in MAIN_MODELS:
        for seed in seeds_for(model):
            scores = load_scores(model, seed)
            counts, det = [], {"point": [], "contextual": []}
            for _, r in labels.iterrows():
                test, cal = scores[r.chan_id]
                pred = detect(test, cal, CFG)
                counts.append(channel_counts(pred, r.seqs, len(test)))
                for t, cls in zip(r.seqs, r.classes):
                    det[cls].append(any(overlap(p, t) for p in pred))
            a = aggregate(counts)
            rows.append(dict(model=model, seed=seed, precision=a["precision"], recall=a["recall"], f1=a["f1"],
                             fp=a["fp"], point_recall=np.mean(det["point"]), contextual_recall=np.mean(det["contextual"])))
            print(f"{model} s{seed}: P={a['precision']:.3f} R={a['recall']:.3f} F1={a['f1']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "reference_postproc.csv", index=False)
    g = df.groupby("model", sort=False)
    lines = ["| model | seeds | precision | recall | strict F1 | point recall | contextual recall |", "|---|---|---|---|---|---|---|"]
    for m, d in g:
        f = lambda c: f"{d[c].mean():.3f} ± {d[c].std(ddof=0):.3f}" if len(d) > 1 else f"{d[c].iloc[0]:.3f}"
        lines.append(f"| {m} | {len(d)} | {f('precision')} | {f('recall')} | {f('f1')} | {f('point_recall')} | {f('contextual_recall')} |")
    (OUT / "reference_postproc.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
