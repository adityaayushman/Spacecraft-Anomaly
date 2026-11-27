
### Overall

| model | n | recall (detected) | ceiling sep_max | ceiling sep_p99 |
|---|---|---|---|---|
| persistence | 105 | 0.58 | 0.72 | 0.90 |
| moving_avg | 105 | 0.60 | 0.67 | 0.77 |
| ridge_ar | 105 | 0.65 | 0.60 | 0.79 |
| iforest | 105 | 0.24 | 0.33 | 0.44 |
| lstm_ae | 105 | 0.41 | 0.47 | 0.58 |
| cnn | 105 | 0.55 | 0.61 | 0.78 |
| gru | 105 | 0.56 | 0.61 | 0.79 |
| lstm | 105 | 0.51 | 0.58 | 0.72 |
| gru_h32 | 105 | 0.56 | 0.62 | 0.76 |
| gru_h128 | 105 | 0.61 | 0.59 | 0.77 |
| gru_h256 | 105 | 0.60 | 0.62 | 0.81 |
| gru_w50 | 105 | 0.53 | 0.60 | 0.76 |
| gru_w200 | 105 | 0.53 | 0.62 | 0.78 |
| gru_nocmd | 105 | 0.52 | 0.62 | 0.85 |

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
| lstm_ae | contextual | 43 | 0.09 | 0.30 | 0.37 |
| lstm_ae | point | 62 | 0.63 | 0.58 | 0.73 |
| cnn | contextual | 43 | 0.28 | 0.42 | 0.65 |
| cnn | point | 62 | 0.74 | 0.74 | 0.87 |
| gru | contextual | 43 | 0.28 | 0.37 | 0.65 |
| gru | point | 62 | 0.76 | 0.77 | 0.89 |
| lstm | contextual | 43 | 0.19 | 0.40 | 0.56 |
| lstm | point | 62 | 0.74 | 0.71 | 0.84 |
| gru_h32 | contextual | 43 | 0.26 | 0.40 | 0.60 |
| gru_h32 | point | 62 | 0.77 | 0.77 | 0.87 |
| gru_h128 | contextual | 43 | 0.47 | 0.42 | 0.63 |
| gru_h128 | point | 62 | 0.71 | 0.71 | 0.87 |
| gru_h256 | contextual | 43 | 0.49 | 0.47 | 0.72 |
| gru_h256 | point | 62 | 0.68 | 0.73 | 0.87 |
| gru_w50 | contextual | 43 | 0.30 | 0.35 | 0.58 |
| gru_w50 | point | 62 | 0.69 | 0.77 | 0.89 |
| gru_w200 | contextual | 43 | 0.21 | 0.40 | 0.63 |
| gru_w200 | point | 62 | 0.76 | 0.77 | 0.89 |
| gru_nocmd | contextual | 43 | 0.23 | 0.40 | 0.77 |
| gru_nocmd | point | 62 | 0.73 | 0.77 | 0.90 |

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
| lstm_ae | short | 41 | 0.51 | 0.27 | 0.51 |
| lstm_ae | medium | 29 | 0.31 | 0.45 | 0.52 |
| lstm_ae | long | 35 | 0.37 | 0.71 | 0.71 |
| cnn | short | 41 | 0.54 | 0.56 | 0.78 |
| cnn | medium | 29 | 0.41 | 0.55 | 0.69 |
| cnn | long | 35 | 0.69 | 0.71 | 0.86 |
| gru | short | 41 | 0.61 | 0.56 | 0.71 |
| gru | medium | 29 | 0.41 | 0.48 | 0.76 |
| gru | long | 35 | 0.63 | 0.77 | 0.91 |
| lstm | short | 41 | 0.56 | 0.51 | 0.71 |
| lstm | medium | 29 | 0.45 | 0.55 | 0.66 |
| lstm | long | 35 | 0.51 | 0.69 | 0.80 |
| gru_h32 | short | 41 | 0.61 | 0.56 | 0.71 |
| gru_h32 | medium | 29 | 0.41 | 0.52 | 0.69 |
| gru_h32 | long | 35 | 0.63 | 0.77 | 0.89 |
| gru_h128 | short | 41 | 0.56 | 0.51 | 0.68 |
| gru_h128 | medium | 29 | 0.55 | 0.52 | 0.76 |
| gru_h128 | long | 35 | 0.71 | 0.74 | 0.89 |
| gru_h256 | short | 41 | 0.56 | 0.56 | 0.71 |
| gru_h256 | medium | 29 | 0.59 | 0.55 | 0.83 |
| gru_h256 | long | 35 | 0.66 | 0.74 | 0.91 |
| gru_w50 | short | 41 | 0.54 | 0.59 | 0.71 |
| gru_w50 | medium | 29 | 0.38 | 0.48 | 0.72 |
| gru_w50 | long | 35 | 0.66 | 0.71 | 0.86 |
| gru_w200 | short | 41 | 0.59 | 0.56 | 0.71 |
| gru_w200 | medium | 29 | 0.34 | 0.48 | 0.72 |
| gru_w200 | long | 35 | 0.63 | 0.80 | 0.91 |
| gru_nocmd | short | 41 | 0.56 | 0.61 | 0.78 |
| gru_nocmd | medium | 29 | 0.45 | 0.48 | 0.79 |
| gru_nocmd | long | 35 | 0.54 | 0.74 | 0.97 |

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
| lstm_ae | MSL | 36 | 0.42 | 0.36 | 0.44 |
| lstm_ae | SMAP | 69 | 0.41 | 0.52 | 0.65 |
| cnn | MSL | 36 | 0.47 | 0.47 | 0.67 |
| cnn | SMAP | 69 | 0.59 | 0.68 | 0.84 |
| gru | MSL | 36 | 0.50 | 0.44 | 0.72 |
| gru | SMAP | 69 | 0.59 | 0.70 | 0.83 |
| lstm | MSL | 36 | 0.47 | 0.44 | 0.61 |
| lstm | SMAP | 69 | 0.54 | 0.65 | 0.78 |
| gru_h32 | MSL | 36 | 0.47 | 0.44 | 0.64 |
| gru_h32 | SMAP | 69 | 0.61 | 0.71 | 0.83 |
| gru_h128 | MSL | 36 | 0.56 | 0.42 | 0.75 |
| gru_h128 | SMAP | 69 | 0.64 | 0.68 | 0.78 |
| gru_h256 | MSL | 36 | 0.58 | 0.47 | 0.81 |
| gru_h256 | SMAP | 69 | 0.61 | 0.70 | 0.81 |
| gru_w50 | MSL | 36 | 0.47 | 0.47 | 0.72 |
| gru_w50 | SMAP | 69 | 0.57 | 0.67 | 0.78 |
| gru_w200 | MSL | 36 | 0.47 | 0.44 | 0.69 |
| gru_w200 | SMAP | 69 | 0.57 | 0.71 | 0.83 |
| gru_nocmd | MSL | 36 | 0.50 | 0.44 | 0.75 |
| gru_nocmd | SMAP | 69 | 0.54 | 0.71 | 0.90 |
