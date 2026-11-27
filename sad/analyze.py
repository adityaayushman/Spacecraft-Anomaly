"""Error analysis + figures.

For every labeled anomaly and every model (seed 0) record whether it was detected
under the cross-fitted threshold config, and how separable it is from normal data:
  sep_max : peak score inside the anomaly > max score outside ALL labeled anomalies
            (a zero-false-positive threshold exists for it -> the *model* can see it)
  sep_p99 : peak inside > 99th percentile of scores outside labeled anomalies
Recall below the separability ceiling is loss due to thresholding, not the model.

python -m sad.analyze
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import fold_assignment, load_labels
from .detect import Config, detect, ewma
from .evaluate import MAIN_MODELS, OUT, VARIANTS, count_table, crossfit, load_scores
from .detect import grid

MODEL_ORDER = MAIN_MODELS + VARIANTS
from .metrics import overlap, to_mask

SPAN = 30
COL = {"strict": "#2b6cb0", "pa": "#dd8452", "pt": "#8c8c8c", "ok": "#2a9d8f", "bad": "#e63946"}


def anomaly_detail():
    labels, folds = load_labels(), fold_assignment(load_labels())
    choices = json.loads((OUT / "chosen_configs.json").read_text())
    rows = []
    for key, ch in choices.items():
        model, seed = key.split("/")
        if seed != "s0":
            continue
        scores = load_scores(model, 0)
        for _, r in labels.iterrows():
            test, cal = scores[r.chan_id]
            cfg = Config(**ch[str(1 - folds[r.chan_id])])      # config tuned on the *other* fold
            pred = detect(test, cal, cfg)
            s = ewma(test, SPAN)
            out = s[~to_mask(r.seqs, len(s))]
            for i, ((a, b), cls) in enumerate(zip(r.seqs, r.classes)):
                pk = s[a:b + 1].max()
                rows.append(dict(model=model, chan_id=r.chan_id, idx=i, spacecraft=r.spacecraft, cls=cls,
                                 length=b - a + 1, detected=any(overlap(p, (a, b)) for p in pred),
                                 peak_in=pk, max_out=out.max(), p99_out=np.percentile(out, 99),
                                 sep_max=pk > out.max(), sep_p99=pk > np.percentile(out, 99)))
    df = pd.DataFrame(rows)
    df["length_bucket"] = pd.qcut(df.length, 3, labels=["short", "medium", "long"])
    df.to_csv(OUT / "anomaly_detail.csv", index=False)
    return df


def breakdown(df):
    lines = []
    for title, col in [("Overall", None), ("By anomaly class", "cls"), ("By length tertile", "length_bucket"),
                       ("By spacecraft", "spacecraft")]:
        lines += [f"\n### {title}\n", "| model | " + (f"{col} | " if col else "") +
                  "n | recall (detected) | ceiling sep_max | ceiling sep_p99 |",
                  "|---|" + ("---|" if col else "") + "---|---|---|---|"]
        for m in [m for m in MODEL_ORDER if m in set(df.model)]:
            d = df[df.model == m]
            for k, g in (d.groupby(col, observed=True) if col else [(None, d)]):
                lines.append(f"| {m} | " + (f"{k} | " if col else "") +
                             f"{len(g)} | {g.detected.mean():.2f} | {g.sep_max.mean():.2f} | {g.sep_p99.mean():.2f} |")
    (OUT / "error_analysis.md").write_text("\n".join(lines) + "\n")


def fig_models():
    main = pd.read_csv(OUT / "main_results.csv")
    order = [m for m in MAIN_MODELS if m in set(main.model)]
    g = main.groupby("model")[["f1", "pa_f1", "pt_f1"]].agg(["mean", "std"]).loc[order].fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4))
    x, w = np.arange(len(order)), 0.27
    for j, (c, lab, col) in enumerate([("f1", "strict sequence F1", COL["strict"]),
                                       ("pa_f1", "point-adjusted F1", COL["pa"]), ("pt_f1", "point-wise F1", COL["pt"])]):
        ax.bar(x + (j - 1) * w, g[c]["mean"], w, yerr=g[c]["std"], label=lab, color=col, capsize=2)
    ax.set_xticks(x, order, rotation=20)
    ax.set_ylabel("F1 (cross-fitted)")
    ax.set_title("Same detections, three scoring protocols", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_models.png", dpi=140)


def fig_ceiling(df):
    order = [m for m in MAIN_MODELS if m in set(df.model)]
    g = df.groupby("model")[["detected", "sep_max", "sep_p99"]].mean().loc[order]
    fig, ax = plt.subplots(figsize=(9, 4))
    x, w = np.arange(len(order)), 0.27
    for j, (c, lab, col) in enumerate([("detected", "recall achieved", COL["strict"]),
                                       ("sep_p99", "separable from 99th pct. of normal", "#7fb3d5"),
                                       ("sep_max", "separable from max of normal (zero-FP ceiling)", "#c9dff0")]):
        ax.bar(x + (j - 1) * w, g[c], w, label=lab, color=col)
    ax.set_xticks(x, order, rotation=20)
    ax.set_ylabel("fraction of the 105 anomalies")
    ax.set_title("Model ceiling vs. recall achieved (seed 0)", loc="left", fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ceiling.png", dpi=140)


def fig_examples(df):
    main = pd.read_csv(OUT / "main_results.csv")
    best = main[main.model.isin(MAIN_MODELS)].groupby("model").f1.mean().idxmax()
    d = df[df.model == best]
    pick = pd.concat([d[d.detected].sample(2, random_state=1), d[~d.detected].sample(2, random_state=1)])
    labels = load_labels().set_index("chan_id")
    scores = load_scores(best, 0)
    fig, axes = plt.subplots(4, 1, figsize=(9, 8))
    for ax, (_, r) in zip(axes, pick.iterrows()):
        s = ewma(scores[r.chan_id][0], SPAN)
        ax.plot(s, lw=0.7, color="#33658a")
        for a, b in labels.loc[r.chan_id].seqs:
            ax.axvspan(a, b, color=COL["ok"], alpha=0.25)
        ax.set_title(f"{best} · {r.chan_id} · {'detected' if r.detected else 'missed'} ({r.cls}, len {r.length})",
                     loc="left", fontsize=9, color=COL["ok"] if r.detected else COL["bad"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("test timestep (green = expert label)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_examples.png", dpi=140)


def span_ablation(models=("persistence", "moving_avg", "gru", "lstm", "cnn")):
    """Does slower smoothing (spans up to 500) help with contextual anomalies?"""
    labels = load_labels()
    folds = fold_assignment(labels)
    rows = []
    for model in models:
        scores = load_scores(model, 0)
        for tag, spans in (("spans 10-60 (main)", (10, 30, 60)), ("spans 10-500", (10, 30, 60, 200, 500))):
            cfgs = grid(spans)
            table = count_table(scores, labels, cfgs)
            _, chosen = crossfit(table, cfgs, folds)
            det = {"point": [], "contextual": []}
            tp = fp = fn = 0
            for _, r in labels.iterrows():
                test, cal = scores[r.chan_id]
                pred = detect(test, cal, chosen[1 - folds[r.chan_id]])
                fp += sum(not any(overlap(p, t) for t in r.seqs) for p in pred)
                for t, cls in zip(r.seqs, r.classes):
                    hit = any(overlap(p, t) for p in pred)
                    det[cls].append(hit)
                    tp, fn = tp + hit, fn + (not hit)
            prec, rec = tp / max(tp + fp, 1), tp / (tp + fn)
            rows.append(f"| {model} | {tag} | {2 * prec * rec / max(prec + rec, 1e-9):.3f} | {prec:.2f} | {rec:.2f} | "
                        f"{np.mean(det['point']):.2f} | {np.mean(det['contextual']):.2f} |")
    head = ["| model | smoothing grid | strict F1 | precision | recall | point recall | contextual recall |",
            "|---|---|---|---|---|---|---|"]
    (OUT / "span_ablation.md").write_text("\n".join(head + rows) + "\n")
    print("\n".join(head + rows))


if __name__ == "__main__":
    df = anomaly_detail()
    breakdown(df)
    fig_models()
    fig_ceiling(df)
    fig_examples(df)
    span_ablation()
