# Pre-Registration: Book of Mormon Stylometric Analysis (Phase 2)

**Registration Date:** 2026-02-03
**Investigators:** Bryson Westover
**Analysis conducted by:** Claude (Anthropic) with GPT-5.2 Pro methodology consultation

---

## 1. Study Information

### 1.1 Title
Stylometric Analysis of Claimed Narrator Voices in the Book of Mormon: A Function-Word Classification Study with Translation-Layer Calibration

### 1.2 Research Questions

**Primary Question:** Do the claimed ancient narrators of the Book of Mormon (Mormon, Nephi, Moroni, Jacob) exhibit detectably different writing styles as measured by function-word frequencies?

**Secondary Question:** Can stylometric methods recover authorial signal through a single translator's voice? (Tested via Constance Garnett translations of Russian literature)

### 1.3 Hypotheses Framework

We assess evidence against five competing hypotheses (not a binary ancient/modern dichotomy):

- **H1:** Modern single-author composition
- **H2:** Modern multi-source composition (single author + sources)
- **H3:** Modern collaborative composition (multiple contributors)
- **H4:** Ancient multi-author compilation with single translator
- **H5:** Deliberate ancient-style pseudepigraphy
- **H0:** Indeterminate (insufficient signal to discriminate)

### 1.4 Prior Work (Exploratory Phase)

Phase 1 analysis (exploratory, not pre-registered) found:
- Block-level balanced accuracy: 21.6% (below 25% chance)
- Permutation p-value: 1.0

However, the permutation test was **invalid** due to a technical bug (zero variance in null distribution). This pre-registration covers the corrected analysis (Phase 2).

---

## 2. Data Specification

### 2.1 Primary Dataset: Book of Mormon

**Source:** 1830 First Edition (Thomas A. Jenson digital replica from Internet Archive)
**File:** `data/text/processed/bom-verses-annotated-v3.json`

**Annotation Schema:**
- `frame_narrator`: Editorial compiler of the plates
- `voice`: Surface speaker for stylometric analysis
- `quote_source`: Source of quotations (ISAIAH, ZENOS, MALACHI, MATTHEW, or null)

**Inclusion Criteria:**
- Voices: MORMON, NEPHI, MORONI, JACOB (≥10 blocks each)
- Quote status: `original` only (excludes quoted material)
- Block size: 1000 words (primary), 500/2000 (robustness)

**Data Structure:**
- 14 voice runs (contiguous same-voice passages)
- 244 blocks at 1000-word target size
- Run distribution: MORMON (4 runs, 171 blocks), NEPHI (5 runs, 39 blocks), MORONI (2 runs, 22 blocks), JACOB (3 runs, 12 blocks)

**Critical Constraint:** Effective sample size is **14 runs**, not 244 blocks. Run_0015 contains 146 blocks (60% of total). All inference must be run-level to avoid pseudoreplication.

### 2.2 Translation-Layer Calibration Dataset: Garnett Corpus

**Source:** Project Gutenberg
**Translator:** Constance Garnett (single translator)
**Authors:** 4 Russian authors (Dostoevsky, Tolstoy, Chekhov, Turgenev)

**Works (11 total, 2,219,293 words):**

| Author | Works | Words | Translation Years |
|--------|-------|-------|-------------------|
| Dostoevsky | 4 | 852,184 | 1912-1918 |
| Tolstoy | 2 | 919,128 | 1901-1904 |
| Chekhov | 3 | 181,953 | 1916-1917 |
| Turgenev | 2 | 266,028 | 1894-1895 |

**Purpose:** Test whether stylometry can recover author identity through a single translator's voice. This calibrates interpretation of BoM null results.

---

## 3. Primary Analysis Specification

### 3.1 Features

**Feature Set:** 169 function words (content-suppressed)
- Articles, pronouns, prepositions, conjunctions, auxiliary verbs
- Archaic forms (ye, thee, thou, hath, doth, wherefore, etc.)
- Excludes all theological/content terms (god, lord, christ, nephi, etc.)

**Feature Extraction:** Frequency per 1000 words for each function word

### 3.2 Model

**Algorithm:** Logistic Regression
- Solver: L-BFGS
- Class weights: Balanced (inverse frequency)
- Max iterations: 1000
- Random seed: 42

**Preprocessing:** StandardScaler (z-score normalization)

### 3.3 Cross-Validation

**Method:** Leave-one-run-out (14-fold)
- Each fold holds out one complete voice run
- Prevents train/test leakage from adjacent blocks

**Block Capping:** Maximum 20 blocks per run in training to prevent run_0015 domination

### 3.4 Primary Metric

**Run-Weighted Balanced Accuracy:**
1. For each held-out run, compute proportion of blocks correctly classified
2. For each voice class, average run-level accuracies across runs of that class
3. Average across classes (balanced)

This metric treats runs as the unit of analysis, respecting the exchangeability structure.

**Chance Baseline:** 25% (4 classes, balanced)

### 3.5 Statistical Inference

**Permutation Test:**
- Permute voice labels at the **run level** (not block level)
- Restricted permutations preserving class run-counts (4/5/2/3)
- Total valid permutations: 14!/(4!×5!×2!×3!) = 2,522,520
- Method: Monte Carlo with 100,000 draws
- **One-sided p-value** (testing above-chance classification)

**Bootstrap Confidence Interval:**
- Stratified resampling at run level (with replacement within each voice)
- 1,000 bootstrap replicates
- 95% CI: 2.5th and 97.5th percentiles

---

## 4. Decision Thresholds

### 4.1 Primary Decision Rule

| Outcome | Criterion | Interpretation |
|---------|-----------|----------------|
| **Significant signal** | p < 0.05 AND run-weighted balanced accuracy > 30% | Evidence of stylistic differentiation; investigate source |
| **Null result** | p ≥ 0.05 OR balanced accuracy ≤ 30% | No evidence of differentiation; proceed to robustness |
| **Indeterminate** | 95% CI spans both chance and meaningful effect | Insufficient power; acknowledge limitation |

### 4.2 Effect Size Interpretation

| Run-Weighted Balanced Accuracy | Interpretation |
|-------------------------------|----------------|
| ≤ 25% | At or below chance |
| 26-35% | Weak effect (marginally above chance) |
| 36-50% | Moderate effect |
| > 50% | Strong effect |

---

## 5. Robustness Analyses (Phase 2.A)

These are **sensitivity analyses**, not independent hypothesis tests. We do not claim significance based on the best-looking variant.

### 5.1 Pre-Specified Variants

| ID | Variant | Rationale |
|----|---------|-----------|
| A1 | Block size: 500 words | More samples, noisier features |
| A2 | Block size: 2000 words | Cleaner signal, fewer samples |
| A3 | Include quotations | Test if exclusion policy affects results |
| A4 | Character 3-grams | Alternative feature family |
| A5 | Combined: FW + char 3-grams | Maximum feature richness |
| A6 | SVM classifier | Alternative algorithm |

### 5.2 Multiple Comparisons Correction

**Method:** Max-statistic permutation correction
- For each permutation, compute scores for all 6 variants
- Record maximum score across variants
- Corrected p-value = proportion where max(permuted) ≥ max(observed)

### 5.3 Robustness Criterion

The null result is considered **robust** if:
- All 6 variants have uncorrected p ≥ 0.05, OR
- Corrected (max-statistic) p ≥ 0.05

---

## 6. Translation-Layer Calibration (Phase 2.D)

### 6.1 Pre-Registered Predictions

| ID | Prediction | If True |
|----|------------|---------|
| D1 | Garnett corpus: author classification accuracy >> chance | Translation does NOT inherently erase authorial signal; BoM null is informative |
| D2 | Garnett corpus: author classification accuracy ≈ chance | Translation homogenization is plausible; BoM null may be expected under H4 |

### 6.2 Methodology (Matched to BoM)

- Same block sizes: 1000 words (primary)
- Same features: Function words (adapted for non-archaic English)
- Same model: Logistic Regression with balanced weights
- Grouping: By work (novel/story), not random blocks
- CV: Leave-one-work-out
- Metric: Work-weighted balanced accuracy

### 6.3 Confound Controls

- **Translation period:** Report results stratified by early/mid/late Garnett career
- **Content leakage:** Mask character names and place names before feature extraction

---

## 7. Exploratory vs. Confirmatory

### 7.1 Confirmatory Analyses (Pre-Registered)

- Primary classification (Section 3)
- Permutation test and bootstrap CI (Section 3.5)
- Decision based on thresholds (Section 4)
- Robustness variants A1-A6 (Section 5)
- Garnett calibration (Section 6)

### 7.2 Exploratory Analyses (Not Pre-Registered)

The following may be conducted but will be clearly labeled as exploratory:
- Genre-controlled analysis (if null result suggests confounding)
- Burrows' Delta and other distance-based methods
- Visualization (PCA, clustering)
- Post-hoc investigation of specific narrator pairs

### 7.3 Phase 1 Status

All Phase 1 results (v1 and v2 classification) are **exploratory** and were conducted before this pre-registration. They informed the methodology but do not constitute confirmatory evidence.

---

## 8. Code and Data Availability

### 8.1 Repository

All code, data, and analysis scripts are version-controlled at:
https://github.com/brysonwestover/book-of-mormon-textual-analysis

### 8.2 Key Files

| File | Description |
|------|-------------|
| `scripts/run_classification_v3.py` | Primary analysis script |
| `scripts/run_robustness.py` | Robustness variants (Phase 2.A) |
| `scripts/run_garnett_analysis.py` | Translation-layer calibration (Phase 2.D) |
| `data/text/processed/bom-verses-annotated-v3.json` | Annotated BoM data |
| `data/reference/garnett/raw/manifest.json` | Garnett corpus manifest |
| `docs/decisions/phase2-execution-plan.md` | Full execution plan |

### 8.3 Reproducibility

- Python version: 3.12.3
- Dependencies: Listed in `requirements-lock.txt`
- Random seed: 42 (fixed for all analyses)

---

## 9. Limitations (Acknowledged in Advance)

1. **Small effective N:** Only 14 runs; power is limited
2. **Severe imbalance:** Mormon dominates (70% of blocks, 4 of 14 runs)
3. **Translation layer:** Under H4, surface stylistic signal may not reflect underlying authors
4. **Genre confounds:** Narrator correlates with discourse type (partially controlled)
5. **Single annotator:** Voice annotations are silver-label (single expert), not gold-label (multiple annotators with IAA)

---

## 10. Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-03 | 1.0 | Initial pre-registration |

---

## 11. Funding and Conflicts

**Funding:** None
**Conflicts of Interest:** None declared
**AI Assistance:** Analysis designed and implemented with Claude (Anthropic) and methodology validated by GPT-5.2 Pro (OpenAI)
