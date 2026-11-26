| model | smoothing grid | strict F1 | precision | recall | point recall | contextual recall |
|---|---|---|---|---|---|---|
| persistence | spans 10-60 (main) | 0.622 | 0.67 | 0.58 | 0.76 | 0.33 |
| persistence | spans 10-500 | 0.622 | 0.67 | 0.58 | 0.76 | 0.33 |
| moving_avg | spans 10-60 (main) | 0.570 | 0.54 | 0.60 | 0.81 | 0.30 |
| moving_avg | spans 10-500 | 0.641 | 0.76 | 0.55 | 0.73 | 0.30 |
| gru | spans 10-60 (main) | 0.596 | 0.63 | 0.56 | 0.76 | 0.28 |
| gru | spans 10-500 | 0.541 | 0.49 | 0.60 | 0.79 | 0.33 |
| lstm | spans 10-60 (main) | 0.532 | 0.55 | 0.51 | 0.74 | 0.19 |
| lstm | spans 10-500 | 0.571 | 0.54 | 0.61 | 0.77 | 0.37 |
| cnn | spans 10-60 (main) | 0.358 | 0.26 | 0.55 | 0.74 | 0.28 |
| cnn | spans 10-500 | 0.423 | 0.34 | 0.57 | 0.77 | 0.28 |
