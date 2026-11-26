"""Cross-fitted threshold tuning, multiple fold splits, bootstrap CIs, result tables.

Channels are split into two folds (stratified by spacecraft). Thresholding
hyperparameters are chosen on one fold (maximising strict sequence F1) and
applied to the other; the two halves are then pooled. No configuration is ever
scored on the channels it was tuned on. The whole procedure is repeated for
FOLD_SEEDS different random splits, and channel-level bootstrap CIs quantify the
uncertainty from having only 81 channels / 105 anomalies.

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
FOLD_SEEDS = range(5)
DETERMINISTIC = {"persistence", "moving_avg", "ridge_ar"}
MAIN_MODELS = ["persistence", "moving_avg", "ridge_ar", "iforest", "lstm_ae", "cnn", "gru", "lstm"]
VARIANTS = ["gru_h32", "gru_h128", "gru_w50", "gru_w200", "gru_nocmd"]
METHODS = {"static_k": "static k·σ", "static_max": "static max-margin", "dynamic": "dynamic (Hundman)", "all": "best of all"}
B = 2000


def available_seeds(model):
    d = SCORES / model
    return sorted(int(p.name[1:]) for p in d.glob("s*") if len(list(p.glob("*.npz"))) == 81) if d.exists() else []


def seeds_for(model):
    return [0] if model in DETERMINISTIC else available_seeds(model)


def load_scores(model, seed):
    out = {}
    for p in (SCORES / model / f"s{seed}").glob("*.npz"):
        z = np.load(p)
        out[p.stem] = (z["test"], z["cal"])
    return out


def count_table(scores, labels, cfgs):
    """table[cfg][chan] -> per-channel metric counts."""
    spans = {cfg.span for cfg in cfgs}
    table = {cfg: {} for cfg in cfgs}
    for _, r in labels.iterrows():
        test, cal = scores[r.chan_id]
        sm = {sp: (ewma(test, sp), ewma(cal, sp)) for sp in spans}
        for cfg in cfgs:
            table[cfg][r.chan_id] = channel_counts(detect(test, cal, cfg, sm[cfg.span]), r.seqs, len(test))
    return table


def tune(table, cfgs, chans):
    """Config maximising strict F1 pooled over `chans` (ties -> fewer false positives)."""
    def key(cfg):
        a = aggregate([table[cfg][c] for c in chans])
        return (a["f1"], -a["fp"])
    return max(cfgs, key=key)


def crossfit(table, cfgs, folds):
    """Per-channel counts under the cross-fitted choice, and the config chosen per tuning fold."""
    rows, chosen = {}, {}
    for f in (0, 1):
        best = tune(table, cfgs, [c for c, v in folds.items() if v == f])
        chosen[f] = best
        rows.update({c: table[best][c] for c, v in folds.items() if v != f})
    return rows, chosen


def f1_from_counts(c):
    """c: (..., 3) array of tp, fp, fn sums -> F1."""
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def evaluate_all():
    labels = load_labels()
    chans = labels.chan_id.tolist()
    cfgs = grid()
    folds_by_seed = {fs: fold_assignment(labels, fs) for fs in FOLD_SEEDS}
    main, abl, auc, choices, strict = [], [], [], {}, {}
    for model in MAIN_MODELS + VARIANTS:
        for seed in seeds_for(model):
            scores = load_scores(model, seed)
            table = count_table(scores, labels, cfgs)
            for fs, folds in folds_by_seed.items():
                for key, name in METHODS.items():
                    sub = [c for c in cfgs if key == "all" or c.method == key]
                    rows, chosen = crossfit(table, sub, folds)
                    agg = aggregate(list(rows.values()))
                    if key == "all":
                        main.append(dict(model=model, seed=seed, fold_seed=fs, **agg))
                        strict[(model, seed, fs)] = np.array([[rows[c]["tp"], rows[c]["fp"], rows[c]["fn"]] for c in chans])
                        if fs == 0:
                            choices[f"{model}/s{seed}"] = {str(f): asdict(c) for f, c in chosen.items()}
                    abl.append(dict(model=model, seed=seed, fold_seed=fs, threshold=name, precision=agg["precision"],
                                    recall=agg["recall"], f1=agg["f1"], fp=agg["fp"]))
            a = [threshold_free(ewma(scores[r.chan_id][0], 30), r.seqs, len(scores[r.chan_id][0]))
                 for _, r in labels.iterrows()]
            auc.append(dict(model=model, seed=seed, auroc=np.nanmean([x[0] for x in a]),
                            auprc=np.nanmean([x[1] for x in a])))
            m = [r["f1"] for r in main if r["model"] == model and r["seed"] == seed]
            print(f"{model} s{seed}: strict F1 over {len(m)} fold splits = {np.mean(m):.3f}", flush=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(main).to_csv(OUT / "main_results.csv", index=False)
    pd.DataFrame(abl).to_csv(OUT / "threshold_ablation.csv", index=False)
    pd.DataFrame(auc).to_csv(OUT / "threshold_free.csv", index=False)
    (OUT / "chosen_configs.json").write_text(json.dumps(choices, indent=1))
    bootstrap(strict)
    write_summary()


def bootstrap(strict):
    """Channel-level bootstrap of strict F1 (shared resamples -> paired differences)."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 81, size=(B, 81))
    dist = {}
    for model in {k[0] for k in strict}:
        combos = [v for k, v in strict.items() if k[0] == model]
        dist[model] = np.mean([f1_from_counts(c[idx].sum(1)) for c in combos], axis=0)   # (B,)
    rows = []
    for m, d in dist.items():
        row = dict(model=m, f1_boot_mean=d.mean(), ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5))
        for ref in ("persistence", "gru"):
            diff = d - dist[ref]
            row |= {f"vs_{ref}_diff": diff.mean(), f"vs_{ref}_lo": np.percentile(diff, 2.5),
                    f"vs_{ref}_hi": np.percentile(diff, 97.5), f"vs_{ref}_p_better": (diff > 0).mean()}
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "bootstrap.csv", index=False)


def _ms(s):
    return f"{s.mean():.3f} ± {s.std(ddof=0):.3f}" if len(s) > 1 else f"{s.iloc[0]:.3f}"


def write_summary():
    main, abl = pd.read_csv(OUT / "main_results.csv"), pd.read_csv(OUT / "threshold_ablation.csv")
    auc, bs = pd.read_csv(OUT / "threshold_free.csv"), pd.read_csv(OUT / "bootstrap.csv").set_index("model")
    ms = [m for m in MAIN_MODELS if m in set(main.model)]
    L = ["## Main results (cross-fitted; mean ± std over seeds × 5 random fold splits)\n",
         "| model | runs | precision | recall | strict F1 | 95% CI (channel bootstrap) | point-adj. F1 | point-wise F1 | FP seq / 10k steps | median delay | AUROC | AUPRC |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ms:
        g, a = main[main.model == m], auc[auc.model == m]
        L.append(f"| {m} | {len(g)} | {_ms(g.precision)} | {_ms(g.recall)} | {_ms(g.f1)} | "
                 f"[{bs.loc[m, 'ci_lo']:.2f}, {bs.loc[m, 'ci_hi']:.2f}] | {_ms(g.pa_f1)} | {_ms(g.pt_f1)} | "
                 f"{_ms(g.fa_per_10k)} | {_ms(g.median_delay)} | {_ms(a.auroc)} | {_ms(a.auprc)} |")
    L += ["\n## Paired channel-bootstrap differences in strict F1 (positive = model better)\n",
          "| model | vs persistence: diff [95% CI] | P(better) | vs GRU: diff [95% CI] | P(better) |", "|---|---|---|---|---|"]
    for m in ms:
        if m == "persistence" or m == "gru":
            continue
        r = bs.loc[m]
        L.append(f"| {m} | {r.vs_persistence_diff:+.3f} [{r.vs_persistence_lo:+.3f}, {r.vs_persistence_hi:+.3f}] | "
                 f"{r.vs_persistence_p_better:.2f} | {r.vs_gru_diff:+.3f} [{r.vs_gru_lo:+.3f}, {r.vs_gru_hi:+.3f}] | "
                 f"{r.vs_gru_p_better:.2f} |")
    r = bs.loc["gru"]
    L.append(f"| gru | {r.vs_persistence_diff:+.3f} [{r.vs_persistence_lo:+.3f}, {r.vs_persistence_hi:+.3f}] | "
             f"{r.vs_persistence_p_better:.2f} | – | – |")
    L += ["\n## Threshold ablation (strict F1; mean ± std over seeds × fold splits)\n",
          "| model | " + " | ".join(METHODS.values()) + " |", "|---|" + "---|" * len(METHODS)]
    for m in ms:
        L.append(f"| {m} | " + " | ".join(_ms(abl[(abl.model == m) & (abl.threshold == t)].f1) for t in METHODS.values()) + " |")
    vs = [v for v in VARIANTS if v in set(main.model)]
    if vs:
        L += ["\n## GRU sensitivity (strict F1; reference GRU restricted to the same seeds)\n",
              "| variant | seeds | strict F1 | reference GRU F1 | point-wise F1 | 95% CI |", "|---|---|---|---|---|---|"]
        for v in vs:
            g = main[main.model == v]
            ref = main[(main.model == "gru") & main.seed.isin(set(g.seed))]
            L.append(f"| {v} | {sorted(set(g.seed))} | {_ms(g.f1)} | {_ms(ref.f1)} | {_ms(g.pt_f1)} | "
                     f"[{bs.loc[v, 'ci_lo']:.2f}, {bs.loc[v, 'ci_hi']:.2f}] |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    evaluate_all()
