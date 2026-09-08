# STRICT_LARGE vs MILD_LARGE: five-seed robustness checkpoint

Date: 2026-09-08

Source log: `reports/ml/22_large_strict_mild_multiseed_20260908.txt`

## Scope and controls

This is a GPS/speed-only controlled Validation diagnostic, not a final performance report. The seeds `316, 42, 2026, 5858, 1234` were fixed before the LARGE results were observed (`SEEDS_SELECTED_AFTER_PERFORMANCE=False`).

- Runtime: Python 3.13.15, PyTorch 2.11.0+cu128, CUDA available, Tesla T4, device `cuda`
- Reference: `models/transformer/model_utils.py:TrajectoryTransformer` at commit `2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959`
- STRICT_LARGE SHA256: `5CC827DAB88D17F68C5EF7F1A09127A81E8B5F683990EDAB0CC7D00302C275FA`
- MILD_LARGE SHA256: `E7638533CC73B91E5F9BFB065344B7993D9E666E1CD3A4D5B97DEF48863AC4C8`
- Train: 1,500 samples per arm, 300 per class
- Composition per class: 271 common strict samples plus 29 strict-only or 29 mild-only samples
- Validation: the same 150 samples in both arms, 30 per class (`VALIDATION_IDENTICAL=True`)
- Internal Test: not used
- Scaling: a separate `StandardScaler` was fitted to each arm's Train split only
- Model: window size 200, feature size 1, 5 classes, d_model 128, 8 heads, 4 layers, 4 KV heads, dropout 0.1, 728,494 parameters
- Optimization: batch size 64, AdamW, learning rate 0.0002, weight decay 0.0001, gradient clipping 1.0, CrossEntropyLoss
- Training: at most 50 epochs, patience 7; checkpoint selected by minimum Validation loss

## Validation-loss checkpoint results

| Seed | Arm | Best epoch | Stop epoch | Val loss | Accuracy | Macro F1 | Highest observed Macro F1 (epoch, diagnostic only) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 316 | STRICT_LARGE | 22 | 29 | 0.776611 | 0.766667 | 0.765152 | 0.765152 (22) |
| 316 | MILD_LARGE | 27 | 34 | 0.751752 | 0.766667 | 0.759809 | 0.760740 (21) |
| 42 | STRICT_LARGE | 21 | 28 | 0.735705 | 0.766667 | 0.758163 | 0.769593 (19) |
| 42 | MILD_LARGE | 22 | 29 | 0.799457 | 0.740000 | 0.735986 | 0.745141 (8) |
| 2026 | STRICT_LARGE | 23 | 30 | 0.740904 | 0.713333 | 0.710760 | 0.757984 (8) |
| 2026 | MILD_LARGE | 3 | 10 | 0.851085 | 0.700000 | 0.695024 | 0.718672 (2) |
| 5858 | STRICT_LARGE | 13 | 20 | 0.813594 | 0.720000 | 0.719215 | 0.738039 (19) |
| 5858 | MILD_LARGE | 21 | 28 | 0.757086 | 0.733333 | 0.731661 | 0.759241 (22) |
| 1234 | STRICT_LARGE | 21 | 28 | 0.791869 | 0.740000 | 0.722806 | 0.740465 (24) |
| 1234 | MILD_LARGE | 15 | 22 | 0.771742 | 0.733333 | 0.721517 | 0.748537 (14) |

The highest observed Macro F1 is diagnostic only and is not the checkpoint criterion.

## Class recall and F1 at the Validation-loss checkpoint

| Seed | Arm | WALK R/F1 | BIKE R/F1 | CAR R/F1 | BUS R/F1 | SUBWAY R/F1 |
|---:|---|---|---|---|---|---|
| 316 | STRICT_LARGE | 0.966667 / 0.878788 | 0.800000 / 0.857143 | 0.566667 / 0.576271 | 0.700000 / 0.700000 | 0.800000 / 0.813559 |
| 316 | MILD_LARGE | 0.966667 / 0.852941 | 0.800000 / 0.827586 | 0.466667 / 0.518519 | 0.733333 / 0.733333 | 0.866667 / 0.866667 |
| 42 | STRICT_LARGE | 1.000000 / 0.882353 | 0.766667 / 0.821429 | 0.466667 / 0.528302 | 0.733333 / 0.733333 | 0.866667 / 0.825397 |
| 42 | MILD_LARGE | 0.900000 / 0.843750 | 0.800000 / 0.872727 | 0.466667 / 0.509091 | 0.666667 / 0.689655 | 0.866667 / 0.764706 |
| 2026 | STRICT_LARGE | 0.900000 / 0.843750 | 0.800000 / 0.813559 | 0.533333 / 0.533333 | 0.566667 / 0.596491 | 0.766667 / 0.766667 |
| 2026 | MILD_LARGE | 0.900000 / 0.805970 | 0.733333 / 0.814815 | 0.400000 / 0.480000 | 0.733333 / 0.628571 | 0.733333 / 0.745763 |
| 5858 | STRICT_LARGE | 0.933333 / 0.848485 | 0.733333 / 0.814815 | 0.466667 / 0.500000 | 0.733333 / 0.647059 | 0.733333 / 0.785714 |
| 5858 | MILD_LARGE | 0.866667 / 0.825397 | 0.733333 / 0.830189 | 0.500000 / 0.526316 | 0.700000 / 0.688525 | 0.866667 / 0.787879 |
| 1234 | STRICT_LARGE | 0.966667 / 0.852941 | 0.766667 / 0.836364 | 0.300000 / 0.382979 | 0.833333 / 0.735294 | 0.833333 / 0.806452 |
| 1234 | MILD_LARGE | 0.966667 / 0.840580 | 0.766667 / 0.821429 | 0.366667 / 0.448980 | 0.733333 / 0.676923 | 0.833333 / 0.819672 |

## Seed-wise MILD_LARGE minus STRICT_LARGE

| Seed | Accuracy delta | Macro F1 delta |
|---:|---:|---:|
| 316 | 0.000000 | -0.005343 |
| 42 | -0.026667 | -0.022177 |
| 2026 | -0.013333 | -0.015736 |
| 5858 | +0.013333 | +0.012446 |
| 1234 | -0.006667 | -0.001289 |

## Aggregate verification

Standard deviations are population standard deviations (`ddof=0`). The values were independently recalculated from the seed-level results printed in the log and agree with the logged aggregates within six-decimal output precision.

| Metric | STRICT_LARGE mean +/- std | MILD_LARGE mean +/- std | MILD - STRICT mean +/- std | MILD wins |
|---|---:|---:|---:|---:|
| Accuracy | 0.741333 +/- 0.022470 | 0.734667 +/- 0.021250 | -0.006667 +/- 0.013333 | 1/5 |
| Macro F1 | 0.735219 +/- 0.022049 | 0.728799 +/- 0.021049 | -0.006420 +/- 0.011987 | 1/5 |

Accuracy was tied at seed 316. MILD_LARGE was higher on both metrics only at seed 5858. No statistical significance is claimed.

## Small versus LARGE interpretation

The previous equal-size 645-Train diagnostic recorded higher Validation Accuracy and Macro F1 for MILD_EXPANDED in all five predeclared seeds. In this equal-size 1,500-Train five-seed diagnostic, that direction was not reproduced.

The current evidence is insufficient to judge the 2-second-gap mild release as a performance-improving preprocessing policy. This does not establish that STRICT is statistically significantly better, nor does it support a general claim that MILD is harmful.

## Limitations

- Validation contains only 30 samples per class; one sample changes class recall by about 0.033333.
- The same Validation split has been reused across multiple development experiments.
- No statistical significance is claimed.
- These are not Test-set results; Internal Test remains unused.
- This is a GPS/speed-only diagnostic.
- The input domain is not the same as real iPhone CANOPY usage.
- This is not final AI Hub performance, CANOPY service performance, or paper-reproduction performance.
- A CUBLAS workspace warning appears in the log, but the run ends with `TRAINING_STATUS=COMPLETED`.
