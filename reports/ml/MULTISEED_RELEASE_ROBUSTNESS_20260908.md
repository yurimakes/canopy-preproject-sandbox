# STRICT_EXPANDED vs MILD_EXPANDED: five-seed robustness checkpoint

Date: 2026-09-08

Source log: `reports/ml/20_strict_mild_multiseed_20260908.txt`

## Scope and controls

This checkpoint records a controlled Validation diagnostic. The seeds `316, 42, 2026, 5858, 1234` were fixed before the robustness results were observed (`SEEDS_SELECTED_AFTER_PERFORMANCE=False`).

- Runtime: Python 3.13.15, PyTorch 2.11.0+cu128, CUDA available, Tesla T4, device `cuda`
- Reference: `models/transformer/model_utils.py:TrajectoryTransformer` at commit `2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959`
- STRICT_EXPANDED SHA256: `54CC016782D14343578E738C84B0CC2AC3CBFFE1871C3A1D005F87B272229A8B`
- MILD_EXPANDED SHA256: `2C3A968D209C41ED23C89935E65D441946E2FE344E862B29E4954444B82F43AC`
- Train: 645 samples per arm, 129 per class
- Validation: the same 150 samples in both arms, 30 per class (`VALIDATION_IDENTICAL=True`)
- Internal Test: not used
- Scaling: a separate `StandardScaler` was fitted to each arm's Train split only
- Model: window size 200, feature size 1, 5 classes, d_model 128, 8 heads, 4 layers, 4 KV heads, dropout 0.1, 728,494 parameters
- Optimization: batch size 64, AdamW, learning rate 0.0002, weight decay 0.0001, gradient clipping 1.0, CrossEntropyLoss
- Training: at most 50 epochs, patience 7; checkpoint selected by minimum Validation loss

## Validation-loss checkpoint results

| Seed | Arm | Best epoch | Stop epoch | Val loss | Accuracy | Macro F1 | Highest observed Macro F1 (epoch, diagnostic only) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 316 | STRICT_EXPANDED | 13 | 20 | 0.889174 | 0.700000 | 0.694504 | 0.727674 (20) |
| 316 | MILD_EXPANDED | 24 | 31 | 0.842050 | 0.726667 | 0.720097 | 0.746257 (12) |
| 42 | STRICT_EXPANDED | 19 | 26 | 0.920757 | 0.693333 | 0.691908 | 0.723678 (21) |
| 42 | MILD_EXPANDED | 26 | 33 | 0.835088 | 0.733333 | 0.712704 | 0.730662 (21) |
| 2026 | STRICT_EXPANDED | 25 | 32 | 0.856674 | 0.700000 | 0.699541 | 0.722858 (27) |
| 2026 | MILD_EXPANDED | 26 | 33 | 0.785504 | 0.733333 | 0.730856 | 0.749836 (22) |
| 5858 | STRICT_EXPANDED | 13 | 20 | 0.891257 | 0.693333 | 0.690630 | 0.717909 (19) |
| 5858 | MILD_EXPANDED | 9 | 16 | 0.904266 | 0.740000 | 0.740126 | 0.740126 (9) |
| 1234 | STRICT_EXPANDED | 31 | 38 | 0.865884 | 0.706667 | 0.709021 | 0.723568 (37) |
| 1234 | MILD_EXPANDED | 19 | 26 | 0.865751 | 0.720000 | 0.710025 | 0.721155 (12) |

The highest observed Macro F1 is diagnostic only and is not the checkpoint criterion.

## Class recall and F1 at the Validation-loss checkpoint

| Seed | Arm | WALK R/F1 | BIKE R/F1 | CAR R/F1 | BUS R/F1 | SUBWAY R/F1 |
|---:|---|---|---|---|---|---|
| 316 | STRICT_EXPANDED | 0.966667 / 0.852941 | 0.766667 / 0.836364 | 0.400000 / 0.428571 | 0.633333 / 0.633333 | 0.733333 / 0.721311 |
| 316 | MILD_EXPANDED | 0.966667 / 0.852941 | 0.800000 / 0.842105 | 0.433333 / 0.490566 | 0.700000 / 0.656250 | 0.733333 / 0.758621 |
| 42 | STRICT_EXPANDED | 0.933333 / 0.835821 | 0.766667 / 0.836364 | 0.433333 / 0.433333 | 0.600000 / 0.620690 | 0.733333 / 0.733333 |
| 42 | MILD_EXPANDED | 1.000000 / 0.869565 | 0.800000 / 0.813559 | 0.300000 / 0.400000 | 0.800000 / 0.738462 | 0.766667 / 0.741935 |
| 2026 | STRICT_EXPANDED | 0.833333 / 0.806452 | 0.800000 / 0.827586 | 0.500000 / 0.508475 | 0.700000 / 0.688525 | 0.666667 / 0.666667 |
| 2026 | MILD_EXPANDED | 0.866667 / 0.825397 | 0.766667 / 0.821429 | 0.500000 / 0.566038 | 0.800000 / 0.695652 | 0.733333 / 0.745763 |
| 5858 | STRICT_EXPANDED | 0.966667 / 0.816901 | 0.733333 / 0.830189 | 0.400000 / 0.436364 | 0.666667 / 0.606061 | 0.700000 / 0.763636 |
| 5858 | MILD_EXPANDED | 0.966667 / 0.828571 | 0.700000 / 0.807692 | 0.566667 / 0.576271 | 0.700000 / 0.666667 | 0.766667 / 0.821429 |
| 1234 | STRICT_EXPANDED | 0.833333 / 0.793651 | 0.766667 / 0.793103 | 0.566667 / 0.531250 | 0.633333 / 0.655172 | 0.733333 / 0.771930 |
| 1234 | MILD_EXPANDED | 0.966667 / 0.828571 | 0.766667 / 0.836364 | 0.366667 / 0.440000 | 0.766667 / 0.686567 | 0.733333 / 0.758621 |

## Seed-wise MILD_EXPANDED minus STRICT_EXPANDED

| Seed | Accuracy delta | Macro F1 delta |
|---:|---:|---:|
| 316 | +0.026667 | +0.025592 |
| 42 | +0.040000 | +0.020796 |
| 2026 | +0.033333 | +0.031315 |
| 5858 | +0.046667 | +0.049496 |
| 1234 | +0.013333 | +0.001003 |

## Aggregate verification

Standard deviations are population standard deviations (`ddof=0`). The aggregate values below were independently recalculated from the seed-level values printed in the log and agree with the logged aggregates within the six-decimal output precision.

| Metric | STRICT_EXPANDED mean ± std | MILD_EXPANDED mean ± std | MILD - STRICT mean ± std | MILD wins |
|---|---:|---:|---:|---:|
| Accuracy | 0.698667 ± 0.004989 | 0.730667 ± 0.006799 | +0.032000 ± 0.011470 | 5/5 |
| Macro F1 | 0.697121 ± 0.006688 | 0.722761 ± 0.011292 | +0.025641 ± 0.015695 | 5/5 |

Because seed-level metrics and deltas are printed independently to six decimal places, subtraction of displayed values can differ from a logged delta by 0.000001 at a rounding boundary (observed for seed 316 and seed 1234 Macro F1). The aggregate verification allowed only that display-precision tolerance.

## Interpretation and limitations

In this diagnostic with five predeclared seeds and the same 150-sample Validation split, MILD_EXPANDED showed higher Accuracy and Macro F1 than STRICT_EXPANDED for every seed.

This result is limited as follows:

- Validation contains only 30 samples per class; one sample changes class recall by about 0.033333.
- The same Validation split is being reused across development experiments.
- Seed 1234's Macro F1 delta is very small (+0.001003).
- No statistical significance is claimed.
- This is not final AI Hub performance, CANOPY service performance, or paper-reproduction performance.
- A CUBLAS workspace warning appears in the log, but the logged run ends with `TRAINING_STATUS=COMPLETED`.
