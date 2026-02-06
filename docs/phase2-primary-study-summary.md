# Primary Study (Phase 2.0) Summary

*Generated: February 5, 2026*

---

## The Question

Can machine learning classifiers distinguish between the 4 claimed ancient narrators (Mormon, Nephi, Moroni, Jacob) based on function-word writing style?

---

## Study Design

| Parameter | Value |
|-----------|-------|
| Text blocks | 1000 words each |
| Features | 169 function words |
| Classifier | Logistic Regression (balanced) |
| CV method | Leave-one-run-out (14 runs) |
| Metric | Run-weighted balanced accuracy |
| Inference | 100,000 restricted permutations |

---

## Data Structure

| Narrator | Runs | Blocks |
|----------|------|--------|
| Mormon | 4 | 45 (capped from 171) |
| Nephi | 5 | 37 |
| Moroni | 2 | 21 |
| Jacob | 3 | 12 |
| **Total** | **14** | **115** |

---

## Key Results

### Primary Metric

**Observed accuracy: 24.2%** (chance = 25%)

### Permutation Test

**p = 0.177**
- Not significant at any conventional threshold
- The observed accuracy is completely typical under the null hypothesis
- Null distribution: mean = 18.2%, 95th percentile = 30.1%

### Per-Narrator Breakdown

| Narrator | Run accuracies | Pattern |
|----------|---------------|---------|
| Nephi | 15%, 38%, 100%, 67%, 100% | Wildly inconsistent |
| Mormon | 60%, 56%, 17%, 0% | Inconsistent |
| Jacob | 0%, 0%, 0% | Never identified correctly |
| Moroni | 0%, 0% | Never identified correctly |

### Uncertainty Quantification

**Bootstrap 95% CI: [5.4%, 39.5%]** — includes chance level

---

## Supplementary Analyses

### TOST Equivalence Testing
- Equivalence bounds: 25% ± 15% (i.e., 10% to 40%)
- 90% CI: [10.1%, 40.9%]
- **TOST p = 0.06** (near-equivalence, just missed formal threshold)

### Bayes Factor
- **BF01 = 2.85** (weak evidence favoring the null hypothesis)

---

## Interpretation

### 1. No detectable stylometric signal
The classifier performs at chance level, indistinguishable from random guessing.

### 2. Highly inconsistent predictions
The wild variability across runs (0% to 100%) indicates the classifier is not detecting any stable pattern — it's essentially noise.

### 3. Two narrators never identified
Jacob and Moroni were never correctly classified across any run, suggesting their "styles" are not distinguishable from the others.

### 4. Converging evidence

| Analysis | Result | Interpretation |
|----------|--------|----------------|
| Point estimate | 24.2% vs 25% chance | Essentially at chance |
| Permutation test | p = 0.177 | Not unusual under null |
| Bootstrap CI | [5.4%, 39.5%] | Includes chance |
| TOST | p = 0.06 | Near-equivalence to chance |
| Bayes factor | BF01 = 2.85 | Weak evidence for null |

---

## Bottom Line

**The classifier cannot distinguish between the four narrators.**

The results are exactly what we would expect if all the text were written by a single author (or by authors with indistinguishable function-word usage patterns).

---

## Technical Details

### Source Files
- Results: `results/classification-results-v3.json`
- TOST: `results/tost-equivalence-results.json`

### Pre-Registration
- OSF pre-registration: `docs/osf-preregistration.md`
- Methodology followed as specified in Section 3
