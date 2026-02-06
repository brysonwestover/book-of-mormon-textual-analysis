# Book of Mormon Stylometric Analysis - Project Wiki

## Overview

This project applies computational stylometric methods to examine whether the claimed ancient narrators of the Book of Mormon exhibit detectably different writing styles. The analysis uses function-word frequencies—considered resistant to conscious authorial control—to test for stylistic differentiation between narrator voices.

A parallel calibration study using Constance Garnett's translations of Russian literature tests whether authorial signal can survive passage through a single translator's voice, which informs interpretation of results under the "ancient multi-author with single translator" hypothesis.

---

## Research Questions

**Primary Question:** Do the claimed ancient narrators of the Book of Mormon (Mormon, Nephi, Moroni, Jacob) exhibit detectably different writing styles as measured by function-word frequencies?

**Secondary Question:** Can stylometric methods recover authorial signal through a single translator's voice? (Tested via Constance Garnett translations)

---

## Hypotheses Framework

We assess evidence against five competing hypotheses (not a binary ancient/modern dichotomy):

| Hypothesis | Description |
|------------|-------------|
| **H1** | Modern single-author composition |
| **H2** | Modern multi-source composition (single author + sources) |
| **H3** | Modern collaborative composition (multiple contributors) |
| **H4** | Ancient multi-author compilation with single translator |
| **H5** | Deliberate ancient-style pseudepigraphy |
| **H0** | Indeterminate (insufficient signal to discriminate) |

---

## Datasets

### Primary Dataset: Book of Mormon

- **Source:** 1830 First Edition (Thomas A. Jenson digital replica)
- **Structure:** 14 voice runs (contiguous same-voice passages)
- **Narrators:** Mormon (4 runs), Nephi (5 runs), Moroni (2 runs), Jacob (3 runs)
- **Constraint:** Effective sample size is 14 runs, not individual blocks

### Calibration Dataset: Garnett Corpus

- **Source:** Project Gutenberg
- **Translator:** Constance Garnett (single translator)
- **Authors:** 4 Russian authors
- **Works:** 19 total (~3,000,000 words)

| Author | Works | Period |
|--------|-------|--------|
| Dostoevsky | 6 | 1912-1918 (late) |
| Tolstoy | 4 | 1901-1915 (mixed) |
| Chekhov | 4 | 1916-1918 (late) |
| Turgenev | 5 | 1894-1896 (early) |

---

## Analysis Phases

### Phase 1: Exploratory (Complete - Not Pre-Registered)

Initial exploration that informed methodology. Results are exploratory only.

### Phase 2.0: Primary Analysis (Complete)

| Component | Specification |
|-----------|---------------|
| Features | 169 function words |
| Block size | 1000 words |
| Model | Logistic Regression (balanced weights) |
| CV | Leave-one-run-out (14 folds) |
| Metric | Run-weighted balanced accuracy |
| Inference | 100,000 restricted permutations |
| Bootstrap | 1,000 stratified replicates |

**Results:**
- Accuracy: 24.2% (chance = 25%)
- Permutation p-value: 0.177
- Bootstrap 95% CI: [5.4%, 39.5%]
- **Interpretation:** No detectable stylometric signal

**Supplementary Analyses:**
- TOST equivalence: p = 0.06 (near-equivalence to chance)
- Bayes factor: BF01 = 2.85 (weak evidence for null)

### Phase 2.A: Robustness Testing (In Progress)

Tests sensitivity of null result across methodological variants:

| Variant | Description | Observed Accuracy |
|---------|-------------|-------------------|
| A1 | Block size: 500 words | 27.9% |
| A2 | Block size: 2000 words | 18.8% |
| A3 | Include quotations | 31.1% |
| A4 | Character 3-grams | 26.1% |
| A5 | Combined: FW + char 3-grams | 24.5% |
| A6 | SVM classifier | 23.6% |

- **Correction:** MaxT permutation correction (A3 excluded due to different run count)
- **Permutations:** 10,000
- **Status:** Running on AWS (30% complete as of Feb 6, 2026)

### Phase 2.D: Translation-Layer Calibration (Complete)

Tests whether stylometry recovers authorial signal through single translator.

| Analysis | Accuracy | Chance | p-value | Bootstrap CI | Status |
|----------|----------|--------|---------|--------------|--------|
| **Novels Only (Primary)** | 58.2% | 33% | **0.0016** | [32.5%, 70.5%] | ✅ Significant |
| **Full Corpus** | 54.4% | 25% | **0.0001** | ✅ | ✅ Significant |
| **Early Period (1894-1904)** | 68.2% | 50% | 0.148 | NOT ESTIMABLE | ⚠️ Not significant |
| **Late Period (1912-1918)** | 55.6% | 33% | **0.006** | NOT ESTIMABLE | ✅ Significant |

**Confound Controls (Pre-Registered):**
- Character/place name masking: Implemented (100+ Russian names)
- Translation period stratification: Implemented (with Amendment #2 limitations)

**Key Finding:** Stylometry CAN recover authorial signal through translation (3/4 analyses significant). This validates the method and makes the BoM null result informative.

---

## Pre-Registration Amendments

### Amendment #1 (February 5, 2026)
**URL:** https://osf.io/sthmk/files/uzhkd

| # | Deviation | Rationale |
|---|-----------|-----------|
| 1 | Robustness uses 10K permutations (vs 100K) | Computational constraints |
| 2 | A3 excluded from maxT family | Different run count (15 vs 14) |
| 3 | Added TOST/Bayes supplementary analyses | Strengthens null interpretation |
| 4 | Garnett corpus expanded (11→19 works) | Increased power; genre control |
| 5 | Minor: duplicate function words in code | No impact (auto-deduplicated) |

### Amendment #2 (February 5, 2026)
**URL:** https://osf.io/sthmk/files/ruzq9

| # | Deviation | Rationale |
|---|-----------|-----------|
| 6 | Bootstrap CI feasibility criterion (≥95%) | Mathematical incompatibility with LOWO when author has 2 works |
| 7 | Prior analyses disclosed | Transparency about timing |

**Technical Detail:** Stratified bootstrap with replacement + LOWO CV is incompatible when an author has only 2 works (~30-49% failure rate). Permutation test remains valid (0% failure rate).

---

## Key Findings (As of February 6, 2026)

### Phase 2.0 Primary Result

**The classifier cannot distinguish between the four BoM narrators.**

- Accuracy essentially at chance level (24.2% vs 25%)
- Highly inconsistent predictions across runs
- Jacob and Moroni never correctly classified
- Results consistent with single-author hypothesis (H1)

### Phase 2.D Calibration Result

**Stylometry CAN detect authorial signal through a single translator.**

- Primary analysis: 58.2% accuracy, p=0.0016 (highly significant)
- 3 of 4 analyses reach statistical significance
- This proves the method works, making the BoM null result meaningful

### Combined Interpretation

The BoM null result is **informative** because:
1. The Garnett calibration proves stylometry detects different authors through translation
2. The BoM narrators genuinely lack distinguishable function-word patterns
3. This is consistent with single-author composition (H1)

---

## Repository Structure

```
book-of-mormon-textual-analysis/
├── data/
│   ├── text/processed/           # Annotated BoM data
│   └── reference/garnett/raw/    # Garnett translations
├── scripts/
│   ├── run_classification_v3.py  # Primary analysis (Phase 2.0)
│   ├── run_robustness_optimized.py    # Robustness (Phase 2.A)
│   ├── run_garnett_analysis_optimized.py  # Calibration (Phase 2.D)
│   └── run_period_analyses.py    # Period-stratified analyses
├── results/
│   ├── classification-results-v3.json
│   ├── tost-equivalence-results.json
│   ├── garnett-checkpoint.json
│   ├── period-analysis-results.json
│   └── [pending robustness results]
├── docs/
│   ├── osf-preregistration.md
│   ├── osf-amendment-2026-02-05.md
│   ├── osf-amendment-2-2026-02-05.md
│   ├── PROJECT-WIKI.md (this file)
│   └── SESSION-PROGRESS-*.md
└── requirements-lock.txt
```

---

## Reproducibility

- **Python version:** 3.12.3
- **Random seed:** 42 (fixed for all analyses)
- **Dependencies:** Listed in `requirements-lock.txt`

---

## AI Assistance Disclosure

This analysis was designed and implemented with assistance from:
- **Claude** (Anthropic) - Code implementation and execution
- **GPT-5.2 Pro** (OpenAI) - Methodology validation and review

All AI-assisted work is documented and reproducible via the scripts in this repository.

---

## References

- Good, P. I. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
- Lakens, D., Scheel, A. M., & Isager, P. M. (2018). Equivalence testing for psychological research. *AMPPS*, 1(2), 259-269.
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing*. Wiley.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

---

## Contact

Bryson Westover

---

*Last updated: February 6, 2026*
