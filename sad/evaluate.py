"""Cross-fitted threshold tuning and result tables.

Channels are split into two folds (stratified by spacecraft). Thresholding
hyperparameters are chosen on one fold (maximising strict sequence F1) and
applied to the other; the two halves are then pooled. No configuration is ever
scored on the channels it was tuned on.

python -m sad.evaluate
"""
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from .data import ROOT, fold_assignment, load_labels
from .detect import Config, detect, ewma, grid
from .metrics import aggregate, channel_counts, prf, threshold_free

SCORES = ROOT / "results" / "scores"
OUT = ROOT / "results"
SEEDS = {"persistence": [0], "moving_avg": [0], "ridge_ar": [0]}   # deterministic baselines
MODEL_ORDER = ["persistence", "moving_avg", "ridge_ar", "iforest", "lstm_ae", "cnn", "gru", "lstm"]
METHODS = {"static_k": "static k·σ", "static_max": "static max-margin", "dynamic": "dynamic (Hundman)", "all": "best of all"}


def available_seeds(model):
    return sorted(int(p.name[1:]) for p in (SCORES / model).glob("s*") if len(list((p).glob("*.npz"))) == 81)


def load_scores(model, seed):
    return {p.stem: (np.load(p)["test"], np.load(p)["cal"]) for p in (SCORES / model / f"s{seed}").glob("*.npz")}


def count_table(scores, labels, cfgs):
    """counts[cfg][chan] -> per-channel metric counts."""
    smooth = {(c, sp): (ewma(scores[c][0], sp), ewma(scores[c][1], sp))
              for c in labels.chan_id for sp in {cfg.span for cfg in cfgs}}
    table = {cfg: {} for cfg in cfgs}
    for _, r in labels.iterrows():
        n = len(scores[r.chan_id][0])
        for cfg in cfgs:
            pred = detect(scores[r.chan_id][0], scores[r.chan_id][1], cfg, smooth[(r.chan_id, cfg.span)])
            table[cfg][r.chan_id] = channel_counts(pred, r.seqs, n)
    return table


def tune(table, cfgs, chans):
    """Config maximising strict F1 pooled over `chans` (ties -> fewer false positives)."""
    def key(cfg):
        a = aggregate([table[cfg][c] for c in chans])
        return (a["f1"], -a["fp"])
    return max(cfgs, key=key)


def crossfit(table, cfgs, folds):
    """Returns (per-channel counts under the cross-fitted choice, chosen cfg per tuning fold)."""
    rows, chosen = {}, {}
    for f in (0, 1):
        tune_ch = [c for c, v in folds.items() if v == f]
        test_ch = [c for c, v in folds.items() if v != f]
        best = tune(table, cfgs, tune_ch)
        chosen[f] = best
        rows.update({c: table[best][c] for c in test_ch})
    return rows, chosen


def evaluate_all():
    labels = load_labels()
    folds = fold_assignment(labels)
    cfgs = grid()
    main, abl, auc, choices = [], [], [], {}
    for model in MODEL_ORDER:
        for seed in SEEDS.get(model, available_seeds(model)):
            scores = load_scores(model, seed)
            table = count_table(scores, labels, cfgs)
            for key, name in METHODS.items():
                sub = [c for c in cfgs if key == "all" or c.method == key]
                rows, chosen = crossfit(table, sub, folds)
                agg = aggregate(list(rows.values()))
                if key == "all":
                    main.append(dict(model=model, seed=seed, **agg))
                    choices[f"{model}/s{seed}"] = {str(f): asdict(c) for f, c in chosen.items()}
                abl.append(dict(model=model, seed=seed, threshold=name, precision=agg["precision"],
                                recall=agg["recall"], f1=agg["f1"], fp=agg["fp"]))
            a = [threshold_free(ewma(scores[r.chan_id][0], 30), r.seqs, len(scores[r.chan_id][0]))
                 for _, r in labels.iterrows()]
            auc.append(dict(model=model, seed=seed, auroc=np.nanmean([x[0] for x in a]),
                            auprc=np.nanmean([x[1] for x in a])))
            print(f"{model} s{seed}: strict F1={main[-1]['f1']:.3f} PA-F1={main[-1]['pa_f1']:.3f}", flush=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(main).to_csv(OUT / "main_results.csv", index=False)
    pd.DataFrame(abl).to_csv(OUT / "threshold_ablation.csv", index=False)
    pd.DataFrame(auc).to_csv(OUT / "threshold_free.csv", index=False)
    (OUT / "chosen_configs.json").write_text(json.dumps(choices, indent=1))
    write_summary()


def _ms(s):
    return f"{s.mean():.3f} ± {s.std(ddof=0):.3f}" if len(s) > 1 else f"{s.iloc[0]:.3f}"


def write_summary():
    main = pd.read_csv(OUT / "main_results.csv")
    abl = pd.read_csv(OUT / "threshold_ablation.csv")
    auc = pd.read_csv(OUT / "threshold_free.csv")
    order = [m for m in MODEL_ORDER if m in set(main.model)]
    lines = ["## Main results (cross-fitted, all 81 channels; mean ± std over seeds)\n",
             "| model | seeds | precision | recall | strict F1 | point-adj. F1 | point-wise F1 | FP seq / 10k steps | median delay | AUROC | AUPRC |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        g, a = main[main.model == m], auc[auc.model == m]
        lines.append(f"| {m} | {len(g)} | {_ms(g.precision)} | {_ms(g.recall)} | {_ms(g.f1)} | {_ms(g.pa_f1)} | "
                     f"{_ms(g.pt_f1)} | {_ms(g.fa_per_10k)} | {_ms(g.median_delay)} | {_ms(a.auroc)} | {_ms(a.auprc)} |")
    lines += ["\n## Threshold ablation (strict F1, cross-fitted; mean ± std over seeds)\n",
              "| model | " + " | ".join(METHODS.values()) + " |", "|---|" + "---|" * len(METHODS)]
    for m in order:
        cells = [_ms(abl[(abl.model == m) & (abl.threshold == t)].f1) for t in METHODS.values()]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    evaluate_all()
