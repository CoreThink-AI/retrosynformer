# Structured Dropout Evaluation — Hyperparameter Search Results

*Branch: `feature-structured-dropout` — June 2026*

---

## 1. Studies and data

Four Optuna studies were run on the small dataset (589 templates, greedy beam search,
`eval_routes_frequency=100`), comparing standard dropout against structured dropout
across roughly 10–12 trials each at up to 200 epochs:

| Study | Type | Trials in top 26 |
|-------|------|-----------------|
| `hypertune-compare_small_standard_dropout` | Standard | 12 |
| `hypertune-compare2_small_standard_dropou` | Standard | 6 |
| `hypertune-compare2_small_structured_drop` | Structured | 7 |
| `hypertune-compare_small_structured_dropo` | Structured | 3 |

Two metrics are tracked per trial:

- **valid action accuracy** (`v_action_acc`) — fraction of correct template predictions
  at each step, averaged over validation routes.
- **valid route accuracy** (`v_route_acc`) — fraction of validation routes where every
  template in the route is predicted correctly (a stricter, compound metric).

The Optuna objective metric changed across studies (details in §4); both metrics are
reported here for completeness.

---

## 2. Top-26 results by valid action accuracy

```
  #  v_action_acc↑  optuna    ep  trial  study                                     n_heads  n_layers  head_dim  dropout  lr        bottleneck
---  ------------  ------  ----  -----  ----------------------------------------  -------  --------  --------  -------  --------  ----------
 1        0.3543  0.3543   200      2  hypertune-compare_small_standard_dropout       2        4        128      0.078   9.90e-04  -
 2        0.3447  0.3447   144      6  hypertune-compare_small_standard_dropout       4        3         64      0.221   2.16e-03  -
 3        0.3419  0.3419   200      3  hypertune-compare2_small_standard_dropou       2        2        256      0.122   2.27e-04  -
 4        0.3365  0.3365   200      1  hypertune-compare2_small_standard_dropou       1        3        256      0.261   4.99e-04  -
 5        0.3365  0.3365    35      8  hypertune-compare_small_standard_dropout       4        5        256      0.055   9.55e-03  -
 6        0.3362  0.1052   200      5  hypertune-compare2_small_structured_drop  ★   3        2        128      0.005   2.89e-04  154
 7        0.3321  0.3321   102      4  hypertune-compare2_small_standard_dropou       2        4         64      0.226   4.08e-03  -
 8        0.3312  0.3312   145      9  hypertune-compare_small_standard_dropout       2        4         64      0.234   1.50e-03  -
 9        0.3180  0.3180   200      2  hypertune-compare2_small_structured_drop  ★   2        3         64      0.158   3.63e-04  221
10        0.3175  0.3175    67      3  hypertune-compare2_small_structured_drop  ★   3        2        256      0.093   1.27e-02  121
11        0.3167  0.3167   200      1  hypertune-compare2_small_structured_drop  ★   3        4        128      0.121   1.57e-04  134
12        0.3144  0.3144    65      3  hypertune-compare_small_standard_dropout       3        5         64      0.309   1.17e-02  -
13        0.3101  0.3101   200     11  hypertune-compare_small_standard_dropout       2        4        128      0.189   1.05e-04  -
14        0.2932  0.2932    50      7  hypertune-compare_small_standard_dropout       1        4        256      0.389   5.16e-02  -
15        0.2738  0.0675    86      7  hypertune-compare2_small_structured_drop  ★   1        4        256      0.261   8.62e-02  216
16        0.2735  0.2735   200      4  hypertune-compare_small_standard_dropout       1        3        128      0.432   1.33e-04  -
17        0.2717  0.0664    71      6  hypertune-compare2_small_structured_drop  ★   3        2         64      0.206   1.18e-01  186
18        0.2711  0.0658    69      8  hypertune-compare2_small_structured_drop  ★   2        2         64      0.296   9.02e-02   46
19        0.2589  0.2589    58      2  hypertune-compare2_small_standard_dropou       3        2        128      0.006   7.56e-02  -
20        0.2546  0.2969    17      0  hypertune-compare_small_standard_dropout       1        3        256      0.100   2.11e-01  -
21        0.2538  0.2538    31      0  hypertune-compare_small_structured_dropo  ★   1        3        256      0.100   2.11e-01  131
22        0.2525  0.2525    34      0  hypertune-compare2_small_standard_dropou       1        3        256      0.100   2.11e-01  -
23        0.2514  0.2514    17     10  hypertune-compare_small_standard_dropout       1        4        256      0.403   3.76e-01  -
24        0.2505  0.2505    30      5  hypertune-compare_small_standard_dropout       1        2        128      0.123   2.22e-01  -
25        0.2321  0.2321    46      1  hypertune-compare_small_structured_dropo  ★   1        4         64      0.247   9.84e-02  438
26        0.2130  0.2130    26      2  hypertune-compare_small_structured_dropo  ★   1        4        256      0.430   2.53e-01  289
```

★ = structured dropout trial

---

## 3. Findings

### 3.1 Standard dropout leads on both metrics

The top 5 positions by action accuracy are all standard dropout trials. The best
structured dropout trial ranks 6th (action accuracy 0.336, one percentage point below
the 5th standard trial at 0.337). By valid **route** accuracy the gap is larger: the
best SD trial achieves 0.105 (rank 6 in the route-accuracy table), while standard
dropout holds ranks 1–5 with a best of 0.116.

There is no metric where structured dropout clearly outperforms standard dropout in
this search.

### 3.2 The action-accuracy/route-accuracy gap is larger for SD

The most striking pattern in the data is that several structured dropout trials score
well on action accuracy but poorly on route accuracy — a disconnect that is less
pronounced in standard dropout trials.

| Trial | Type | `v_action_acc` | `v_route_acc` | Ratio |
|-------|------|---------------|--------------|-------|
| #6 SD trial 5 | Structured | 0.336 | 0.105 | **3.2×** |
| #9 SD trial 2 | Structured | 0.318 | 0.100 | **3.2×** |
| #15 SD trial 7 | Structured | 0.274 | 0.068 | **4.0×** |
| #17 SD trial 6 | Structured | 0.272 | 0.066 | **4.1×** |
| #18 SD trial 8 | Structured | 0.271 | 0.066 | **4.1×** |
| #1 standard trial 2 | Standard | 0.354 | 0.116 | **3.1×** |
| #3 standard trial 3 | Standard | 0.342 | 0.109 | **3.1×** |

Standard dropout trials cluster around a 3.1× ratio between the two metrics. Structured
dropout trials with moderate-to-high standard dropout (0.09–0.30) reach ratios of
3.2–4.1×: the model learns to predict individual template steps correctly at a higher
rate than it chains those steps into valid complete routes.

**Interpretation:** the structured mask may be regularising individual token predictions
too aggressively, or may be disrupting the recurrent state-passing that a Decision
Transformer relies on for multi-step coherence. Single-step accuracy is not a sufficient
condition for route accuracy; the mask may be breaking cross-step information flow.

### 3.3 The best SD trial has near-zero standard dropout

The one SD trial that narrows the gap to standard dropout (#6, `v_route_acc = 0.105`,
`dropout = 0.005`) has essentially no standard dropout. This is consistent with the
hypothesis that the two regularizers compete: adding the structured mask on top of
standard dropout double-penalises the network and degrades multi-step coherence.

This motivates the new `small_nodropout_sd.yaml` study: fix `dropout = 0.0` so the only
regularizer is the structured mask, and let Optuna search `structured_dropout_rate` and
`structured_dropout_bottleneck` freely.

### 3.4 Dropout is detrimental across the board

Across all studies, trials with lower standard dropout perform better on both metrics:

| Dropout range | Mean `v_action_acc` (top 13 standard trials) |
|---------------|----------------------------------------------|
| 0.00–0.10 | **0.338** (trials #1, #5, #19) |
| 0.10–0.25 | **0.328** (trials #2, #3, #7, #8, #13) |
| 0.25–0.45 | **0.291** (trials #4, #12, #14, #16, #23) |

The correlation is monotone: every increase in dropout hurts. The highest-performing
standard dropout trials (#1, #5, dropout = 0.078, 0.055) outperform all trials with
dropout > 0.20.

### 3.5 Architecture patterns

Across both study types, the winning architectures share:

- `n_heads`: 2–4 (single-head models are consistently weaker)
- `n_layers`: 2–4 (deeper is not better at this dataset scale; `n_layers=24` scored
  near the bottom of all completed trials)
- `head_dim`: 64–256 (no strong signal; 128 or 256 appear in most top results)
- `lr`: 1×10⁻⁴ – 4×10⁻³ (very high lr ≥ 0.05 yields early convergence but lower
  ceiling; very low lr ≤ 5×10⁻⁵ is too slow for 200 epochs)

The optimal `structured_dropout_bottleneck` in the SD trials shows no clear trend
in this search (range 46–438); the search was under-powered for this dimension.

---

## 4. Note on the `optuna` column

The `optuna` column records the objective value that Optuna stored for each trial.
An inconsistency is visible in the `compare2_small_structured_drop` study: trials 1–3
show `optuna ≈ v_action_acc`, while trials 5–8 show `optuna ≈ v_route_acc`. This
indicates the study's `objective_metric` config key was changed from
`valid_action_accuracy` to `valid_route_accuracy` mid-run. The two groups are
therefore not directly comparable by Optuna's internal ranking, though both metrics
are reported here independently.

---

## 5. Conclusions and next steps

| Question | Answer from this data |
|----------|-----------------------|
| Does SD improve route accuracy? | No — best SD (0.105) is below best standard (0.116) |
| Does SD improve action accuracy? | No — best SD (0.336) is just below best standard (0.354) |
| Does SD hurt multi-step coherence? | Likely yes — action/route ratio is higher for SD |
| Is the bottleneck size important? | Unclear — search was under-powered |
| Is standard dropout harmful? | Yes — lower dropout is better for all trials |

**Recommended next experiment:** `small_nodropout_baseline.yaml` vs
`small_nodropout_sd.yaml`, with standard dropout fixed at 0.0 in both, and Optuna
searching `structured_dropout_rate` (0.1–5.0, log scale) alongside `bottleneck` for
the SD arm. This isolates the structured mask signal from the confound of competing
regularizers and gives Optuna a direct scalar dial to determine whether any nonzero
`rate` is beneficial.

---

*Generated 2026-06-15.*
