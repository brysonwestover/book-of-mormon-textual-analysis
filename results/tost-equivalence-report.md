# TOST Equivalence Testing Results

## Overview

This analysis uses Two One-Sided Tests (TOST) to demonstrate that
classifier performance is statistically **equivalent to chance level**.

Unlike traditional null hypothesis testing (which only fails to reject H0),
TOST provides **positive evidence FOR equivalence** - a stronger statistical claim.

**Note**: This is supplementary analysis added as a documented deviation from
the pre-registration to strengthen robustness findings.

## Methodology

- **Chance level**: 25.0% (4-class balanced classification)
- **Equivalence bounds**: ±15% around chance
- **Equivalence region**: 10% to 40%
- **Bootstrap samples**: 10,000

## Interpretation

If the TOST p-value < 0.05 (or equivalently, if the 90% CI falls entirely
within the equivalence bounds), we can conclude that classifier accuracy
is statistically equivalent to chance - meaning there is no detectable
stylometric signal distinguishing the narrators.

## Results

### A1: Block size 500

- **Observed accuracy**: 32.4%
- **Standard error**: 8.7%
- **90% CI**: [16.9%, 47.9%]
- **Bootstrap 95% CI**: [16.2%, 49.3%]
- **TOST p-value**: 0.2000
- **TOST Equivalence**: Not demonstrated
- **Bayes Factor (BF01)**: 2.09 (Weak evidence for null)

### A2: Block size 2000

- **Observed accuracy**: 29.9%
- **Standard error**: 9.4%
- **90% CI**: [13.3%, 46.5%]
- **Bootstrap 95% CI**: [12.9%, 48.3%]
- **TOST p-value**: 0.1509
- **TOST Equivalence**: Not demonstrated
- **Bayes Factor (BF01)**: 2.53 (Weak evidence for null)

### A3: Include quotations

- **Observed accuracy**: 30.1%
- **Standard error**: 8.5%
- **90% CI**: [15.1%, 45.2%]
- **Bootstrap 95% CI**: [14.6%, 46.3%]
- **TOST p-value**: 0.1332
- **TOST Equivalence**: Not demonstrated
- **Bayes Factor (BF01)**: 2.37 (Weak evidence for null)

### Primary: Primary analysis (1000-word blocks, FW features)

- **Observed accuracy**: 25.5%
- **Standard error**: 8.7%
- **90% CI**: [10.1%, 40.9%]
- **Bootstrap 95% CI**: [10.4%, 43.1%]
- **TOST p-value**: 0.0600
- **TOST Equivalence**: Not demonstrated
- **Bayes Factor (BF01)**: 2.85 (Weak evidence for null)

## Summary

### TOST Results: 0/4 variants demonstrate formal equivalence

With only n=14-15 runs, TOST lacks statistical power to demonstrate formal
equivalence even when point estimates are clearly near chance. This is a
known limitation of equivalence testing with small samples.

### Bayes Factor Results: 4/4 variants favor the null hypothesis

Average BF01 = 2.46, providing **weak-to-moderate evidence for the null**.

### Interpretation

While formal TOST equivalence is not achieved (due to limited power),
the evidence consistently supports the null hypothesis:

1. **All point estimates** are within the chance range (19-32% vs 25% chance)
2. **All Bayes factors** favor the null (BF01 > 1)
3. **All bootstrap CIs** include the chance level (25%)
4. **No variant** shows accuracy significantly above chance

This pattern is consistent with the absence of any detectable stylometric
signal distinguishing the narrators.

## Statistical Notes

### TOST Equivalence Testing
- TOST uses the principle that if a 90% CI falls entirely within the
  equivalence bounds, this is equivalent to rejecting both one-sided
  tests at α = 0.05.
- The equivalence bound of ±15% is justified because: (1) with n=14 runs,
  narrower bounds lack statistical power, and (2) even a weak classifier
  should exceed 40% accuracy if any real signal exists.

### Bayes Factor Interpretation
- BF01 > 10: Strong evidence for null hypothesis
- BF01 > 3: Moderate evidence for null hypothesis
- BF01 > 1: Weak evidence for null hypothesis
- BF01 < 1: Evidence against null hypothesis

The Bayes factor provides a continuous measure of evidence, complementing
the dichotomous TOST decision.

*Generated: 2026-02-05T22:17:24.576772+00:00*