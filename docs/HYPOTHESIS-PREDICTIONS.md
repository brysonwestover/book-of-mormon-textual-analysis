# Hypothesis Framework: Predictions and Constraints

This document maps each hypothesis to its predictions about stylometric detectability, enabling principled interpretation of results.

**Last Updated:** 2026-02-06
**Related:** METHODOLOGY.md, LIMITATIONS.md

---

## The Research Question

**Can we detect stylometric differences among the claimed narrators of the Book of Mormon (Mormon, Nephi, Moroni, Jacob) using function-word frequencies?**

This is NOT a direct test of historicity. It is a test of **stylometric separability** under specific conditions:
- English function-word feature space
- Single translator/author producing the English text
- Leave-one-work-out cross-validation
- Run-level permutation inference

---

## Hypothesis Predictions Table

| Hypothesis | Description | Predicts Detectable Narrator Differences? | Reasoning |
|------------|-------------|-------------------------------------------|-----------|
| **H1** | Modern single-author composition | **No / Weak** | Single author produces homogeneous function-word profile across all voices |
| **H2** | Modern multi-source composition | **No / Weak** | Single author compiling sources; author's function-word signature dominates |
| **H3** | Modern collaborative composition | **Possibly Yes** | Multiple 19th-c authors might have distinct profiles, but close collaboration could homogenize |
| **H4** | Ancient multi-author + single translator | **No / Weak** | Translation layer erases/homogenizes source author function-word patterns |
| **H5** | Deliberate ancient-style pseudepigraphy | **No** | Intentional stylistic imitation; author suppresses own variation |
| **H0** | Indeterminate (method insufficient) | **N/A** | Cannot distinguish signal absence from method limitation |

---

## Interpreting Results

### If Significant Narrator Separation Detected (Accuracy >> Chance)

This would provide evidence **against**:
- H1 (single author should not produce separable voices)
- H4 (translation should homogenize)
- H5 (successful pseudepigraphy should not be detectable)

This would be **consistent with**:
- H3 (multiple modern collaborators with distinct styles)
- Partially H2 (if sources have preserved stylistic signatures)

### If No Significant Separation Detected (Accuracy ≈ Chance)

This is **consistent with** (does not distinguish among):
- H1 (single author)
- H2 (single author + sources)
- H4 (ancient multi-author + translation homogenization)
- H5 (successful pseudepigraphy)

This provides evidence **against**:
- Strong version of H3 (multiple distinct modern collaborators)
- Any hypothesis predicting easily detectable narrator variation

---

## Our Results

### Primary Finding (Phase 2.0)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Balanced Accuracy | 24.2% | At chance (25% for 4 classes) |
| Permutation p-value | 0.177 | Not significant |
| TOST p-value | 0.06 | Near-equivalence to chance |
| Bayes Factor BF₀₁ | 2.85 | Weak evidence for null |

**Interpretation:** No statistically significant stylometric differentiation detected.

### Method Validation (Phase 2.D Garnett)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 58.2% | 25 points above chance |
| p-value | 0.0016 | Highly significant |

**Interpretation:** The method CAN detect authorial signal through a single translator's voice. This makes the BoM null result **informative** rather than merely inconclusive.

---

## Constraints on Hypotheses

Given our results, we can state:

### Hypotheses Consistent with Results

1. **H1 (Single modern author):** Highly consistent. A single author would produce the observed homogeneous profile.

2. **H2 (Single author + sources):** Consistent. Author's function-word signature dominates any source material.

3. **H4 (Ancient + translator):** Consistent. The Garnett calibration shows translation CAN preserve authorial signal, but the BoM result could still reflect translation homogenization under different conditions (genre, editorial process, etc.).

4. **H5 (Pseudepigraphy):** Consistent. Successful voice mimicry would produce uniform function-word patterns.

### Hypotheses Disfavored by Results

1. **Strong H3 (Multiple distinct modern collaborators):** Disfavored. Multiple independent authors typically produce detectable stylistic differences (as seen in KJV control, Garnett corpus).

### Hypotheses Not Addressed

- Whether events described actually occurred
- Divine involvement or lack thereof
- Author identities
- Historical authenticity in any religious sense

---

## The Underdetermination Problem

**Multiple hypotheses fit the observed data equally well.**

Our stylometric test cannot uniquely identify the correct hypothesis because:

1. **H1 and H4 make the same prediction** for function-word analysis (no separation expected)

2. **Translation effects are underdetermined:** We demonstrated translation CAN preserve signal (Garnett), but cannot prove the BoM translation process was comparable

3. **Absence of evidence vs. evidence of absence:** Our sensitivity analysis suggests we could detect large effects (~25 pts above chance), but cannot rule out small effects below our detection threshold

---

## Scope of Inference Statement

> "This study evaluates narrator separability in function-word usage under a pre-registered computational pipeline. It does not adjudicate historicity; it quantifies the presence or absence of a specific stylometric signature that some (but not all) origin hypotheses would predict. The null result is consistent with single-author composition (H1), translation-layer homogenization (H4), and successful pseudepigraphy (H5), but does not uniquely identify any of these."

---

## What Would Change These Conclusions?

### Evidence that would favor multi-authorship hypotheses:

- Significant narrator separation in robustness variants (Phase 2.A)
- Genre-controlled analysis showing separation within sermon/narrative categories
- Alternative feature spaces (syntax, discourse) showing narrator differences
- External evidence of multiple authorial hands

### Evidence that would strengthen single-author conclusion:

- Consistent null results across all robustness variants (Phase 2.A in progress)
- Strong similarity to known single-author pseudepigraphic texts
- Computational evidence of unified compositional process

---

## References

- METHODOLOGY.md - Full analytical framework
- LIMITATIONS.md - Explicit scope boundaries
- docs/decisions/expected-results-if-ancient.md - Detailed H4 predictions
- docs/REPRODUCIBILITY.md - How to reproduce these analyses
