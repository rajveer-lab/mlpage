# Ablation and parameter sweep (validation traces)

Score: mean over traces and memory sizes of F_Learned / F_LRU (lower is better, 1.0 = LRU).

## Ablation

| Variant | Faults vs LRU |
|---|---|
| all features | 0.9826 |
| without age | 0.9824 |
| without count | 1.1146 |
| without gap1 | 0.9823 |
| without gap2 | 0.9834 |
| without gap3 | 0.9827 |
| without mean_gap | 0.9813 |
| without age_over_mean_gap | 1.0027 |
| only recency | 2.5047 |
| only frequency | 2.1629 |
| only gap history | 1.0307 |
| design: full (guard on) | 0.9213 |
| design: no DAgger rounds | 0.9589 |
| design: no ghost table | 0.9712 |
| design: tree instead of logistic | 0.9351 |

Feature variants run without the guard, so the model's own quality shows.

## Sweep

| Parameter | Value | Faults vs LRU |
|---|---|---|
| k | 4 | 0.9489 |
| k | 8 | 0.9290 |
| k | 16 | 0.9213 |
| k | 32 | 0.9201 |
| k | 64 | 0.9211 |
| threshold | 0.3 | 0.9212 |
| threshold | 0.5 | 0.9213 |
| threshold | 0.6 | 0.9212 |
| threshold | 0.7 | 0.9215 |
| threshold | 0.8 | 0.9223 |
| threshold | 0.9 | 0.9919 |
| horizon | 0.5 | 0.9302 |
| horizon | 1 | 0.9245 |
| horizon | 2 | 0.9213 |
| horizon | 4 | 0.9197 |
| horizon | 8 | 0.9190 |

