## Main results (cross-fitted, all 81 channels; mean ± std over seeds)

| model | seeds | precision | recall | strict F1 | point-adj. F1 | point-wise F1 | FP seq / 10k steps | median delay | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 1 | 0.670 | 0.581 | 0.622 | 0.799 | 0.076 | 0.673 | 0.176 | 0.661 | 0.461 |
| moving_avg | 1 | 0.543 | 0.600 | 0.570 | 0.618 | 0.301 | 1.189 | 0.149 | 0.655 | 0.456 |
| ridge_ar | 1 | 0.302 | 0.648 | 0.412 | 0.483 | 0.308 | 3.522 | 0.200 | 0.682 | 0.472 |
| iforest | 3 | 0.387 ± 0.046 | 0.235 ± 0.012 | 0.292 ± 0.023 | 0.289 ± 0.022 | 0.167 ± 0.029 | 0.890 ± 0.135 | 0.181 ± 0.033 | 0.601 ± 0.004 | 0.301 ± 0.002 |

## Threshold ablation (strict F1, cross-fitted; mean ± std over seeds)

| model | static k·σ | static max-margin | dynamic (Hundman) | best of all |
|---|---|---|---|---|
| persistence | 0.437 | 0.478 | 0.622 | 0.622 |
| moving_avg | 0.565 | 0.525 | 0.677 | 0.570 |
| ridge_ar | 0.511 | 0.419 | 0.617 | 0.412 |
| iforest | 0.298 ± 0.011 | 0.302 ± 0.008 | 0.327 ± 0.026 | 0.292 ± 0.023 |
