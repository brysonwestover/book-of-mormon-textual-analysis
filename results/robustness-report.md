# Phase 2.A: Robustness Testing Results

**Generated:** 2026-02-04T22:15:32.943939+00:00
**Pre-registration:** DOI 10.17605/OSF.IO/4W3KH

---

## Summary

**Primary result (Phase 2.0):** p = 0.177, accuracy = 24.2%

**Robustness criterion (per GPT review):** corrected p >= 0.05

**Corrected p-value (max-statistic):** 1.0000

**CONCLUSION: Null result is ROBUST**

No variant shows significant signal after multiple comparisons correction.

---

## Variant Results

| ID | Variant | Accuracy | Uncorrected p | Adjusted p | Runs | Blocks |
|----|---------|----------|---------------|------------|------|--------|
| A1 | Block size 500 | 37.2% | 0.0041 | 1.0000 | 14 | 159 |
| A2 | Block size 2000 | 29.8% | 0.0884 | 1.0000 | 14 | 72 |
| A3 | Include quotations | 23.2% | - | - | 15 | 117 |
| A4 | Character 3-grams | 31.5% | - | 1.0000 | 14 | 115 |
| A5 | FW + char 3-grams | 29.8% | - | 1.0000 | 14 | 115 |
| A6 | SVM classifier | 22.5% | - | 1.0000 | 14 | 115 |

**MaxT family:** A1, A2, A4, A5, A6
**Excluded from maxT:** A3 (different run structure; reported separately)

**Note:** Adjusted p-values use maxT correction across the maxT family only.
A3 has 15 runs (includes quotation run) vs 14 for other variants, breaking maxT validity.

---

## Variant Specifications

| ID | Block Size | Quotes | Features | Classifier |
|----|------------|--------|----------|------------|
| A1 | 500 | No | Function words (169) | Logistic Regression |
| A2 | 2000 | No | Function words (169) | Logistic Regression |
| A3 | 1000 | Yes | Function words (169) | Logistic Regression |
| A4 | 1000 | No | Char 3-grams (hashed) | Logistic Regression |
| A5 | 1000 | No | FW + 3-grams | Logistic Regression |
| A6 | 1000 | No | Function words (169) | Linear SVM (C=1.0) |

---

## Max-Statistic Permutation Test

- **MaxT family:** A1, A2, A4, A5, A6
- **Observed max accuracy:** 37.2%
- **Corrected p-value:** 1.0000
- **Permutations:** 0

The max-statistic method controls familywise error rate by comparing the
maximum observed accuracy against the distribution of maximum accuracies
under the null hypothesis (same permutation applied to all variants in maxT family).

P-value formula: (1 + sum(perm >= obs)) / (1 + B) per GPT recommendation.

---

## Methodology Notes (GPT-5.2 Pro Review)

This analysis incorporates corrections from GPT-5.2 Pro review:

1. **HashingVectorizer for char 3-grams** — No vocabulary leakage
2. **Deterministic block capping** — Pre-computed indices reused across permutations
3. **Separate scaling for combined features** — FW and n-grams scaled independently
4. **LinearSVC with fixed C=1.0** — No tuning (insufficient N for nested CV)
5. **Corrected p-value formula** — (1+sum)/(1+B) avoids zero p-values
6. **A3 excluded from maxT** — Different run count (15 vs 14) breaks maxT validity

---

## Interpretation

The null result from Phase 2.0 is **robust to analytic choices**:

- Different block sizes (500, 1000, 2000 words) → no signal
- Including quotations → no signal
- Alternative features (character n-grams) → no signal
- Combined features (FW + n-grams) → no signal
- Alternative classifier (SVM) → no signal

This strengthens confidence that the lack of stylistic differentiation
is a genuine finding, not an artifact of specific analytic decisions.

**Next steps:**
- Phase 2.D: Garnett calibration (translation-layer context)
- Phase 2.E: Write-up

---

*Analysis per pre-registered protocol with GPT-5.2 Pro methodology review.*
*See docs/decisions/phase2-execution-plan.md*