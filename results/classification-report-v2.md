# Stylometric Classification Results (v2 - Corrected Methodology)

**Generated:** 2026-02-02T05:35:03.469935+00:00

---

## Methodology Corrections Applied

This analysis addresses issues identified in GPT-5.2 Pro review:

1. **Content-suppressed features:** Function words only (no theological terms)
2. **Proper baselines:** Majority class (not random)
3. **Imbalance handling:** Class weights + balanced accuracy
4. **Statistical significance:** Permutation test
5. **Apples-to-apples:** Downsampled experiments

---

## Data Summary

- **Total blocks:** 244
- **Features:** 169 (function words only)
- **Classes:** JACOB, MORMON, MORONI, NEPHI

### Class Distribution

| Voice | Blocks | % |
|-------|--------|---|
| MORMON | 171 | 70.1% |
| NEPHI | 39 | 16.0% |
| MORONI | 22 | 9.0% |
| JACOB | 12 | 4.9% |

---

## Baselines

| Baseline | Value |
|----------|-------|
| Majority class (MORMON) | 70.1% |
| Random (uniform) | 25.0% |
| Trivial macro-F1 | 0.206 |

---

## Primary Results (Function Words Only)

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Accuracy | 43.9% | vs 70.1% majority |
| **Balanced Accuracy** | **21.6%** | vs 25% random |
| **Macro F1** | **0.236** | vs 0.206 trivial |

### Per-Class Performance

| Voice | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------| 
| JACOB | 0.000 | 0.000 | 0.000 | 12 |
| MORMON | 0.779 | 0.556 | 0.648 | 171 |
| MORONI | 0.000 | 0.000 | 0.000 | 22 |
| NEPHI | 0.286 | 0.308 | 0.296 | 39 |

### Confusion Matrix

```
Pred →    JACOB  MORMON  MORONI   NEPHI
True ↓
 JACOB        0       2       3       7
MORMON        9      95      52      15
MORONI        0      14       0       8
 NEPHI        7      11       9      12
```

---

## Statistical Significance (Permutation Test)

- **Observed balanced accuracy:** 0.318
- **Null distribution mean:** 0.318 ± 0.000
- **p-value:** 1.0000

**Interpretation:** The classification performance is NOT statistically significant (p ≥ 0.05).

---

## Downsampled Experiment (Apples-to-Apples)

To address class imbalance, we repeatedly sampled 12 blocks per class.

- **Resamples:** 100
- **Macro F1:** 0.115 (95% CI: [0.046, 0.217])
- **Balanced Accuracy:** 11.2% (95% CI: [4.2%, 20.8%])

---

## Top Discriminating Features (Function Words)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | this | 0.3812 |
| 2 | wherefore | 0.2902 |
| 3 | yea | 0.2642 |
| 4 | must | 0.2538 |
| 5 | therefore | 0.2504 |
| 6 | yourselves | 0.2441 |
| 7 | where | 0.2231 |
| 8 | after | 0.2170 |
| 9 | my | 0.2128 |
| 10 | i | 0.1974 |
| 11 | also | 0.1935 |
| 12 | even | 0.1922 |
| 13 | again | 0.1863 |
| 14 | your | 0.1858 |
| 15 | upon | 0.1826 |

---

## Interpretation

**Finding:** The classification performance is not statistically significant.
There is insufficient evidence of stylistic differentiation using function words alone.

**Limitations:**

- Jacob (n=12) treated as exploratory due to small sample
- Cannot distinguish genuine multi-authorship from skilled mimicry
- Translation layer effects cannot be separated
- Topic/genre confounds partially controlled but not eliminated

---

*Methodology per GPT-5.2 Pro consultation. See docs/decisions/classification-methodology-corrections.md*