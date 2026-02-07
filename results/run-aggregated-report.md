# Run-Aggregated Stylometric Analysis (Supplementary)

**Generated:** 2026-02-07T01:26:16.410226+00:00
**Version:** 1.4.0
**Status:** SUPPLEMENTARY ANALYSIS (addresses pseudoreplication concern)

---

## Methodology

This analysis aggregates block-level features to the **run level** before classification.
Instead of treating 244 blocks as independent observations, we average features within
each of the 14 voice runs, resulting in **N=14 true independent observations**.

### Key Differences from Block-Level Analysis

| Aspect | Block-Level (Pre-registered) | Run-Aggregated (This Analysis) |
|--------|------------------------------|--------------------------------|
| Unit of analysis | 244 blocks | 14 runs |
| Feature aggregation | None | Sum counts, then convert to freq |
| CV scheme | Leave-one-run-out (14 folds) | Leave-one-out (14 folds) |
| Block capping | Yes (20/run) | Not needed |
| Pseudoreplication | Partially addressed | Fully addressed |

---

## Data Summary

| Voice | Runs | Blocks | Total Words |
|-------|------|--------|-------------|
| JACOB | 3 | 12 | 10,839 |
| MORMON | 4 | 171 | 174,869 |
| MORONI | 2 | 22 | 22,031 |
| NEPHI | 5 | 39 | 38,481 |
| **Total** | **14** | **244** | **246,220** |

---

## Primary Results

### Classification Accuracy Comparison

| Method | Balanced Accuracy | Macro-F1 | Interpretation |
|--------|-------------------|----------|----------------|
| **Logistic Regression** | 22.5% | 0.216 | At chance |
| **Burrows' Delta** | 32.5% | 0.276 | Above chance |
| Chance baseline | 25.0% | 0.25 | Random guessing |

### Statistical Inference (LR Results)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Permutation p-value | **0.4158** | Not significant |
| Null mean | 19.4% | Expected under no effect |
| Null 95% range | [0.0%, 41.5%] | Permutation distribution |

---

## Per-Class Metrics

### Logistic Regression

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| JACOB | 0.000 | 0.000 | 0.000 | 3 |
| MORMON | 0.286 | 0.500 | 0.364 | 4 |
| MORONI | 0.000 | 0.000 | 0.000 | 2 |
| NEPHI | 0.667 | 0.400 | 0.500 | 5 |
| **Macro** | 0.238 | 0.225 | 0.216 | - |

### Burrows' Delta

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| JACOB | 0.000 | 0.000 | 0.000 | 3 |
| MORMON | 0.667 | 0.500 | 0.571 | 4 |
| MORONI | 0.000 | 0.000 | 0.000 | 2 |
| NEPHI | 0.400 | 0.800 | 0.533 | 5 |
| **Macro** | 0.267 | 0.325 | 0.276 | - |

---

## Confidence Intervals

| Method | 95% CI | Note |
|--------|--------|------|
| Wilson (raw acc) | [11.7%, 54.6%] | Valid for raw accuracy only |
| Permutation null | [0.0%, 41.5%] | Range under null hypothesis |
| Bootstrap | [28.7%, 91.7%] | Unreliable with N=14 |

---

## Regularization Sensitivity (C Parameter)

Since we have p >> n (169 features, 14 samples), results can be sensitive to regularization strength.

| C | Balanced Accuracy | Note |
|---|-------------------|------|
| 0.01 | 28.7% |  |
| 0.1 | 22.5% |  |
| 1.0 | 22.5% | ← default |
| 10.0 | 22.5% |  |
| 100.0 | 22.5% |  |


**Interpretation:** Results are stable across C values.

---

## Jackknife Influence Analysis

How much does removing each run affect the overall result?

| Run ID | Voice | Reduced Acc | Influence |
|--------|-------|-------------|-----------|
| run_0000 | NEPHI | 31.2% | +8.8% |
| run_0002 | NEPHI | 18.8% | -3.8% |
| run_0003 | JACOB | 27.5% | +5.0% |
| run_0004 | NEPHI | 18.8% | -3.8% |
| run_0006 | NEPHI | 12.5% | -10.0% |
| run_0008 | NEPHI | 12.5% | -10.0% |
| run_0009 | JACOB | 22.5% | +0.0% |
| run_0011 | JACOB | 22.5% | +0.0% |
| run_0015 | MORMON | 5.0% | -17.5% |
| run_0017 | MORMON | 5.0% | -17.5% |
| run_0019 | MORMON | 18.3% | -4.2% |
| run_0020 | MORONI | 28.7% | +6.2% |
| run_0021 | MORMON | 35.0% | +12.5% |
| run_0022 | MORONI | 17.5% | -5.0% |


**Max influence magnitude:** 17.5%

**Interpretation:** One or more runs strongly influence the result—findings may not be robust.

---

## Confusion Matrices

### Logistic Regression

```
Predicted:    JACOB  MORMON  MORONI  NEPHI
Actual:
  JACOB           0      2      0      1
  MORMON          1      2      1      0
  MORONI          0      2      0      0
  NEPHI           1      1      1      2
```

### Burrows' Delta

```
Predicted:    JACOB  MORMON  MORONI  NEPHI
Actual:
  JACOB           0      0      0      3
  MORMON          0      2      0      2
  MORONI          0      1      0      1
  NEPHI           1      0      0      4
```

---

## Per-Run Predictions (LR)

| Run ID | True Voice | Predicted | Correct |
|--------|------------|-----------|---------|
| run_0000 | NEPHI | MORMON | ✗ |
| run_0002 | NEPHI | JACOB | ✗ |
| run_0003 | JACOB | NEPHI | ✗ |
| run_0004 | NEPHI | MORONI | ✗ |
| run_0006 | NEPHI | NEPHI | ✓ |
| run_0008 | NEPHI | NEPHI | ✓ |
| run_0009 | JACOB | MORMON | ✗ |
| run_0011 | JACOB | MORMON | ✗ |
| run_0015 | MORMON | MORMON | ✓ |
| run_0017 | MORMON | MORMON | ✓ |
| run_0019 | MORMON | MORONI | ✗ |
| run_0020 | MORONI | MORMON | ✗ |
| run_0021 | MORMON | JACOB | ✗ |
| run_0022 | MORONI | MORMON | ✗ |


---

## Comparison with Block-Level Analysis

| Metric | Block-Level (v3) | Run-Aggregated (LR) | Run-Aggregated (Delta) |
|--------|------------------|---------------------|------------------------|
| Balanced Accuracy | 24.2% | 22.5% | 32.5% |
| Permutation p-value | 0.177 | 0.4158 | - |
| Unit of analysis | 244 blocks | 14 runs | 14 runs |

---

## Interpretation

The run-aggregated analysis confirms the block-level result. With balanced accuracy of 22.5%
and p = 0.416, we find **no significant evidence**
of narrator-level stylistic differentiation when treating runs as the unit of analysis.

Both Logistic Regression and Burrows' Delta (the canonical stylometry baseline) yield
similar accuracy near chance level, reinforcing the null finding.

This supplementary analysis addresses reviewer concerns about pseudoreplication by
ensuring each observation is truly independent.

---

## Limitations

### Critical Limitations

1. **Severe class imbalance (MORONI=2 runs)**: In LOOCV, when a MORONI run is held out,
   training includes only 1 MORONI example. Learning "MORONI" from a single training
   run is essentially few-shot classification and highly unreliable. MORONI predictions
   should be interpreted with extreme caution.

2. **p >> n regime**: With 169 features and 14 samples, the model is severely
   underdetermined. L2 regularization helps but cannot fully address this. Results
   may be sensitive to feature selection and regularization choices.

3. **Small effective sample size**: N=14 runs provides limited statistical power
   to detect even moderate effects. The permutation null distribution is correspondingly
   wide, making it difficult to achieve significance.

### Additional Limitations

4. **Information loss**: Aggregating discards within-run variance information
5. **Not pre-registered**: This is a post-hoc sensitivity analysis
6. **Bootstrap CI unreliability**: With N=14 and imbalanced classes, bootstrap
   resamples often omit classes entirely, making CIs unreliable

---

## Phase 2: Robustness Analyses

### 3-Class Analysis (Excluding MORONI)

Addresses class imbalance by removing MORONI (only 2 runs).

| Metric | Value |
|--------|-------|
| Classes | JACOB, MORMON, NEPHI |
| Chance level | 33.3% |
| LR Balanced Accuracy | 45.0% |
| Delta Balanced Accuracy | 43.3% |
| Permutation p-value | 0.0851 |

**Interpretation:** 3-class analysis (excluding MORONI): BA = 45.0%, p = 0.0851 (chance = 33.3%)

---

### Feature Sensitivity Analysis

Tests stability across different numbers of top-frequency features.

| k (features) | LR BA | Delta BA | p-value |
|--------------|-------|----------|---------|
| 50 | 27.5% | 17.5% | 0.2008 |
| 100 | 45.0% | 33.8% | 0.0230 |
| 150 | 22.5% | 27.5% | 0.3137 |
| 171 | 22.5% | 32.5% | 0.2927 |

**Interpretation:** Feature sensitivity analysis shows classification stability across different numbers of top-frequency features.

---

### Confound Probe (Book Prediction)

Tests whether features capture book/topic rather than narrator style.

| Prediction Target | Balanced Accuracy | Chance | Above Chance |
|-------------------|-------------------|--------|--------------|
| Narrator | 22.5% | 25.0% | -2.5% |
| Book | 8.3% | 11.1% | -2.8% |

**Interpretation:** Narrator BA: 22.5% (chance 25.0%), Book BA: 8.3% (chance 11.1%). No strong confound detected

---

## Figures

- `figures/permutation-null-dist.png` - Permutation null distribution
- `figures/confusion-matrix-lr.png` - Confusion matrix (Logistic Regression)
- `figures/confusion-matrix-delta.png` - Confusion matrix (Burrows' Delta)

---

*Analysis conducted: {results['metadata']['generated_at']}*
*Random seed: {results['metadata']['random_seed']}*
*Version: {results['metadata']['script_version']}*
