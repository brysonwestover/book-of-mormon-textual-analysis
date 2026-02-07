# Methodology Documentation: run_aggregated_analysis.py v1.5.1

**Version:** 1.5.1
**Date:** 2026-02-07
**Status:** Ready for archiving (pre-production run)

---

## 1. Study Design

### 1.1 Purpose
Supplementary stylometric analysis addressing pseudoreplication concerns by aggregating features at the run level. This treats contiguous narrator segments ("runs") as the independent unit of analysis.

### 1.2 Unit of Analysis
- **Run:** A contiguous segment of text attributed to a single narrator
- **N = 14 runs total** (across 4 narrators)
- **N = 12 runs in primary analysis** (MORONI excluded due to n=2)

### 1.3 Data Structure
| Narrator | Runs | Blocks | Words | In Primary? |
|----------|------|--------|-------|-------------|
| JACOB | 3 | 12 | 10,839 | Yes |
| MORMON | 4 | 171 | 174,869 | Yes |
| MORONI | 2 | 22 | 22,031 | **No** (n=2) |
| NEPHI | 5 | 39 | 38,481 | Yes |
| **Total** | **14** | **244** | **246,220** | 12 |

---

## 2. Primary Analysis (Pre-specified, Confirmatory)

### 2.1 Specification
- **Classes:** 3-class (JACOB, MORMON, NEPHI)
- **Excluded:** MORONI (n=2, insufficient for reliable classification)
- **Classifier:** Logistic Regression
  - Regularization: L2 (Ridge), l1_ratio=0
  - C = 1.0 (inverse regularization strength)
  - class_weight = 'balanced'
  - solver = 'lbfgs'
  - max_iter = 2000
- **Features:** All 171 function words (no selection)
- **Validation:** Leave-one-out cross-validation (LOO-CV)
- **Metric:** Balanced accuracy (macro-averaged recall)
- **BA Computation:** From pooled confusion matrix via `sklearn.metrics.balanced_accuracy_score`
  - Collects all LOO-CV predictions, computes single confusion matrix, derives BA
  - BA = mean(per-class recall) = (1/k) × Σ(TP_i / (TP_i + FN_i))

### 2.2 Primary Inference
- **Method:** BLOCKED permutation test (within book-strata)
- **Rationale:** Narrator-book collinearity (~68%) violates exchangeability; blocked permutation respects this structure
- **Permutations:** 10,000 (production) / 100 (quick mode)
- **Alpha:** 0.05 (one-sided)
- **p-value formula:** (# null ≥ observed + 1) / (n_permutations + 1) [Phipson & Smyth 2010]

### 2.3 Strata Definition
Runs grouped by **primary book** (first book if run spans multiple):
- Strata are defined by the book in which the run primarily occurs
- Within-stratum permutation: labels shuffled only among runs in the same book-stratum
- Degenerate permutations (identical to original) are counted and reported

### 2.4 Reference Analysis
- **Unrestricted permutation** reported for comparison only
- **NOT used for inference** because exchangeability is violated
- Large difference between blocked and unrestricted p-values confirms violation

---

## 3. Exploratory Analyses (FDR-corrected)

All exploratory p-values corrected using **Benjamini-Hochberg FDR**.

### 3.1 Exploratory Tests
1. 4-class LR analysis (includes MORONI)
2. 4-class Burrows' Delta
3. 3-class Burrows' Delta
4. Feature sensitivity (k=50, 100, 150)
5. C sensitivity (0.01, 0.1, 1.0, 10.0, 100.0)
6. Narrator vs Book comparison
7. Book prediction p-value

### 3.2 Burrows' Delta Implementation
```
For test sample x and class centroid c_k:
  Delta(x, c_k) = (1/p) * sum_{j=1}^{p} |z_x,j - z_c,j|
where:
  z = (value - mean) / std
  mean, std computed on TRAINING data only (ddof=1)
  Features with zero variance: std set to 1
Prediction: argmin_k Delta(x, c_k)
```
This is "classic" Burrows' Delta (2002).

---

## 4. Sensitivity Analyses

### 4.1 Unrestricted vs Blocked Permutation
- Compare p-values to assess exchangeability violation
- Large difference confirms narrator-book confounding

### 4.2 Feature Ranking Inside Folds
- Features ranked by mean frequency in TRAINING data only (each fold)
- Prevents potential leakage from test samples

### 4.3 Jackknife Influence
- Remove-one-run analysis
- Identifies if results are "carried" by 1-2 influential runs

---

## 5. Uncertainty Quantification

### 5.1 Bootstrap CI (Balanced Accuracy) - DESCRIPTIVE ONLY
- Stratified resampling at run level
- 1,000 bootstrap samples
- **Status:** Descriptive/stability measure only; NOT for comparing to 33.3% chance
- **Rationale:** Bootstrap CI does not respect blocked permutation structure; comparison to theoretical chance (33.3%) is inappropriate when using blocked null. The proper baseline is the blocked-null mean (~42-44%).
- **Caveat:** May be optimistic with N=12 due to LOO duplicate leakage

### 5.2 Wilson CI (Per-class Recall)
- Binomial confidence intervals for each class's recall
- **Caveat:** Assumes independent Bernoulli draws; may be too narrow if predictions correlated

### 5.3 Permutation-based CI - PRIMARY
- Derived from blocked permutation null distribution
- Reports null mean, std, and 95% range
- **Use this as the primary uncertainty/baseline measure**

---

## 6. Collinearity Assessment

### 6.1 Metric Definition
```
Collinearity index = 1 - (unique narrator-book pairs / max possible pairs)
```
Computed from contingency table of runs by narrator × book.

### 6.2 Interpretation
- **LOW (<20%):** Narrator and book largely independent
- **MODERATE (20-50%):** Some overlap
- **HIGH (>50%):** Narrator strongly predicts book position

### 6.3 Current Value
- **68.8% collinearity** - HIGH
- Interpretation: Narrator-book confounding is substantial; cannot fully separate effects

---

## 7. Reproducibility

### 7.1 Random Seeds
- `RANDOM_SEED = 42`
- `numpy.random.RandomState(42)` used throughout
- `random.seed(42)` for Python random

### 7.2 Software
- Python 3.12+
- scikit-learn (LogisticRegression, StandardScaler, balanced_accuracy_score)
- numpy, matplotlib

### 7.3 Solver Settings
- solver = 'lbfgs'
- max_iter = 2000
- warm_start = False

---

## 8. Known Limitations

1. **Sample size:** N=12 runs limits statistical power
2. **High-dimensional:** 171 features with ~11 training samples per fold (p >> n)
3. **Confounding:** 68% narrator-book collinearity; cannot fully separate effects
4. **MORONI excluded:** Only 2 runs; excluded from confirmatory analysis
5. **Generalization:** Results apply to this text; may not generalize
6. **Bootstrap CI:** May be misaligned with blocked permutation null
7. **Wilson CIs:** Assume independence that may not hold

---

## 9. Interpretation Guidelines

### 9.1 Null Result Interpretation
- **DO say:** "We did not detect performance above the blocked-null baseline"
- **DON'T say:** "There is no narrator effect" (absence of evidence ≠ evidence of absence)

### 9.2 Baseline for Comparison
- **Use:** Mean of blocked permutation null distribution
- **Not:** Theoretical chance (1/k classes) - this ignores block structure

### 9.3 What This Analysis Can/Cannot Show
**CAN show:**
- Whether narrator labels predict stylometric features beyond book-strata structure

**CANNOT show:**
- Whether narrators are "different authors"
- Causal attribution of style differences
- Generalization beyond this specific text

---

## 10. Audit Trail

| Version | Date | Changes |
|---------|------|---------|
| v1.0.0 | 2026-02-05 | Initial implementation |
| v1.2.0 | 2026-02-05 | Count-based aggregation, +1 p-value correction |
| v1.3.0 | 2026-02-05 | Zero-word guard, bootstrap caveats |
| v1.4.0 | 2026-02-06 | Burrows' Delta, per-class metrics, visualizations |
| v1.5.0 | 2026-02-06 | FDR correction, blocked permutation (as sensitivity), methodology card |
| v1.5.1 | 2026-02-07 | **Blocked permutation as PRIMARY**, bootstrap CI, per-class Wilson CIs, RNG documentation |

---

## 11. Files for Archiving

- `scripts/run_aggregated_analysis.py` (v1.5.1)
- `data/text/processed/bom-voice-blocks.json` (input data)
- `results/run-aggregated-results.json` (output - after production run)
- `results/figures/*.png` (visualizations)
- `docs/METHODOLOGY-v1.5.1.md` (this file)
- `docs/private/AUDIT-*.md` (audit documentation)
