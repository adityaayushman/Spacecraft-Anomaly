
### Overall

| model | n | recall (detected) | ceiling sep_max | ceiling sep_p99 |
|---|---|---|---|---|
| persistence | 105 | 0.58 | 0.72 | 0.90 |
| moving_avg | 105 | 0.60 | 0.67 | 0.77 |
| ridge_ar | 105 | 0.65 | 0.60 | 0.79 |
| iforest | 105 | 0.24 | 0.33 | 0.44 |

### By anomaly class

| model | cls | n | recall (detected) | ceiling sep_max | ceiling sep_p99 |
|---|---|---|---|---|---|
| persistence | contextual | 43 | 0.33 | 0.47 | 0.77 |
| persistence | point | 62 | 0.76 | 0.90 | 1.00 |
| moving_avg | contextual | 43 | 0.30 | 0.40 | 0.51 |
| moving_avg | point | 62 | 0.81 | 0.85 | 0.95 |
| ridge_ar | contextual | 43 | 0.35 | 0.37 | 0.63 |
| ridge_ar | point | 62 | 0.85 | 0.76 | 0.90 |
| iforest | contextual | 43 | 0.16 | 0.40 | 0.49 |
| iforest | point | 62 | 0.29 | 0.29 | 0.40 |

### By length tertile

| model | length_bucket | n | recall (detected) | ceiling sep_max | ceiling sep_p99 |
|---|---|---|---|---|---|
| persistence | short | 41 | 0.66 | 0.71 | 0.85 |
| persistence | medium | 29 | 0.45 | 0.59 | 0.90 |
| persistence | long | 35 | 0.60 | 0.86 | 0.97 |
| moving_avg | short | 41 | 0.71 | 0.61 | 0.76 |
| moving_avg | medium | 29 | 0.52 | 0.59 | 0.66 |
| moving_avg | long | 35 | 0.54 | 0.80 | 0.89 |
| ridge_ar | short | 41 | 0.71 | 0.61 | 0.78 |
| ridge_ar | medium | 29 | 0.55 | 0.45 | 0.72 |
| ridge_ar | long | 35 | 0.66 | 0.71 | 0.86 |
| iforest | short | 41 | 0.20 | 0.24 | 0.37 |
| iforest | medium | 29 | 0.21 | 0.38 | 0.45 |
| iforest | long | 35 | 0.31 | 0.40 | 0.51 |

### By spacecraft

| model | spacecraft | n | recall (detected) | ceiling sep_max | ceiling sep_p99 |
|---|---|---|---|---|---|
| persistence | MSL | 36 | 0.50 | 0.64 | 0.92 |
| persistence | SMAP | 69 | 0.62 | 0.77 | 0.90 |
| moving_avg | MSL | 36 | 0.56 | 0.53 | 0.75 |
| moving_avg | SMAP | 69 | 0.62 | 0.74 | 0.78 |
| ridge_ar | MSL | 36 | 0.58 | 0.44 | 0.89 |
| ridge_ar | SMAP | 69 | 0.68 | 0.68 | 0.74 |
| iforest | MSL | 36 | 0.28 | 0.28 | 0.36 |
| iforest | SMAP | 69 | 0.22 | 0.36 | 0.48 |
