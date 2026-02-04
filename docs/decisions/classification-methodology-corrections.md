# Classification Methodology Corrections

**Date:** 2026-02-01
**Status:** Implemented
**Consulted:** GPT-5.2 Pro (methodology validation)

---

## Summary

After initial classification experiments, GPT-5.2 Pro review identified critical methodological issues that invalidated our initial interpretation. This document records the corrections.

---

## Critical Error: Baseline Calculation

### The Problem

Our initial analysis reported:
- SVM Linear accuracy: 50%
- "Baseline": 25% (assuming random 4-class)
- Interpretation: "2x baseline suggests detectable differences"

**This was incorrect.**

With class distribution:
- Mormon: 171 blocks (70.1%)
- Nephi: 39 blocks (16.0%)
- Moroni: 22 blocks (9.0%)
- Jacob: 12 blocks (4.9%)

The correct **majority-class baseline** is **70%** (always predict Mormon).

### Corrected Interpretation

Our 50% accuracy is **below the majority baseline**, meaning the classifier is actually performing worse than a trivial "always predict Mormon" rule. This cannot be interpreted as evidence of stylistic differentiation.

### Fix

- Report **macro-F1** and **balanced accuracy** as primary metrics
- Always include majority-class baseline in reporting
- Use class weights to handle imbalance

---

## Issue 2: Feature Selection Problems

### The Problem

Top discriminating features from initial analysis:
1. "wherefore" (function word)
2. "therefore" (function word)
3. "lord" (char n-gram) ← **content word**
4. "upon", "my", "now"

**GPT Assessment:** Features like "lord" and discourse markers like "wherefore/therefore" are **topic/genre markers** (sermon vs narrative), not necessarily authorship signals.

### Risk

We may be detecting **genre/topic differences** rather than **authorial style**.

### Fix

1. **Run content-suppressed analysis:** Function words only, excluding theological lexicon
2. **Feature ablation:** Compare results with/without content words
3. **Unmasking analysis:** Iteratively remove top features to test robustness
4. **Cross-genre validation:** Test if attribution holds across different genres

---

## Issue 3: Class Imbalance

### The Problem

| Voice | Blocks | % |
|-------|--------|---|
| Mormon | 171 | 70.1% |
| Nephi | 39 | 16.0% |
| Moroni | 22 | 9.0% |
| Jacob | 12 | 4.9% |

With this imbalance:
- Raw accuracy is dominated by majority class
- Small classes (Jacob, Moroni) may never be correctly predicted
- Cross-validation folds may have 0 instances of minority classes

### Fix

1. **Use class weights** (`class_weight="balanced"` in sklearn)
2. **Report macro-F1 and balanced accuracy** (not raw accuracy)
3. **Downsampled experiments:** Repeatedly sample Mormon to match smallest class
4. **Use StratifiedGroupKFold** to ensure class representation in folds
5. **Treat Jacob as exploratory** (n=12 is insufficient for strong claims)

---

## Issue 4: Potential Data Leakage

### The Problem

Two leakage risks identified:
1. **N-gram vocabulary selection** on full corpus before CV
2. **Scaler fitting** on full data instead of training folds only

### Fix

1. Use sklearn `Pipeline` to ensure all preprocessing happens within CV folds
2. Vocabulary selection must be fit on training data only
3. Scaler must be fit on training data only

---

## Issue 5: Sentence Features

### The Problem

Sentence-level features (mean words/sentence, mean chars/sentence) depend on punctuation, which is an **editorial layer** in historical texts, not necessarily authorial.

### Fix

- Drop sentence features from primary analysis
- Include as sensitivity analysis only

---

## Issue 6: Vocabulary Richness Features

### The Problem

TTR (type-token ratio) and hapax ratio are highly **topic/genre sensitive** and unstable markers of authorship in short chunks.

### Fix

- Drop vocabulary richness features from primary analysis
- Include as sensitivity analysis only

---

## Issue 7: Statistical Significance

### The Problem

Bootstrap CIs alone don't answer "is there any real signal?"

### Fix

- Add **permutation test** at group level
- Permute author labels by `run_id`, rebuild full CV pipeline
- Build null distribution for macro-F1/balanced accuracy

---

## Revised Analysis Plan

### Primary Analysis (Content-Suppressed)

**Features:**
- Function words only (no content words)
- Mask theological terms if using char n-grams
- Exclude sentence features
- Exclude vocabulary richness features

**Evaluation:**
- Metrics: Macro-F1, balanced accuracy, per-class precision/recall
- CV: StratifiedGroupKFold (or GroupKFold with stratification check)
- Class weights: balanced
- Significance: Permutation test (group-level)

**Baselines:**
- Majority class (70%)
- Macro-F1 of trivial predictor

### Sensitivity Analyses

1. **Feature ablation:**
   - All features vs function words only
   - With vs without theological lexicon
   - With vs without sentence features

2. **Downsampling:**
   - Repeatedly sample Mormon to match Jacob (n=12)
   - Report distribution of macro-F1 across resamples

3. **Unmasking:**
   - Iteratively remove top features
   - Track how quickly separation degrades

4. **Cross-genre:**
   - If genre labels available, test within-genre and cross-genre

### Exploratory Analysis

- Jacob treated as exploratory (n=12 insufficient)
- Pairwise comparisons (Jacob vs Mormon, etc.) with bootstrap CIs
- Distance/clustering analysis

---

## Addressing Hypothesis H3 (Intentional Mimicry)

### GPT Assessment

> "Stylometry cannot reliably detect mimicry vs genuine multi-authorship... H2 vs H3 is typically underdetermined without external constraints."

### What We CAN Test

1. **Do multiple stable, topic-robust stylistic clusters exist?**
   - If YES → consistent with distinct style sources (H1 or competent H3)
   - If NO → weak evidence for distinct authorship

2. **Same-author verification:**
   - Are cross-"author" pairs as different as expected for different authors?
   - Or do they look more like within-author variation?

3. **Unmasking diagnostic:**
   - If separation depends on few salient features → consistent with topic/genre OR "painted-on" voices
   - If separation degrades gradually → consistent with many weak authorial cues

### Framing

Include H3 as a **sensitivity/interpretation section**, not a primary confirmable hypothesis. Report:
- Whether stylistic clusters are topic-robust
- Whether signal persists under content suppression
- Unmasking degradation curve

---

## Minimum Sample Size Guidelines

### Per GPT Consultation

| Measure | Minimum | Preferred |
|---------|---------|-----------|
| Words per block | 1,000 | 1,500-2,000 |
| Blocks per author | 12 (exploratory) | ≥20 (stable) |
| Total words per author | 10,000 | ≥20,000 |

### Current Data Status

| Voice | Blocks | Words | Status |
|-------|--------|-------|--------|
| Mormon | 171 | 174,463 | ✅ Primary |
| Nephi | 39 | 38,431 | ✅ Primary |
| Moroni | 22 | 21,998 | ⚠️ Marginal |
| Jacob | 12 | 10,814 | ⚠️ Exploratory only |

---

## Implementation Checklist

- [x] Document corrections (this file)
- [ ] Update `run_classification.py` with:
  - [ ] Class weights
  - [ ] Macro-F1 and balanced accuracy as primary metrics
  - [ ] Correct baseline reporting (majority class)
  - [ ] Function-word-only feature set
  - [ ] Permutation test
  - [ ] Downsampling experiments
- [ ] Update `extract_features.py` with:
  - [ ] Option to exclude content words
  - [ ] Option to mask theological terms
- [ ] Add unmasking analysis
- [ ] Update PROJECT-STATUS.md
- [ ] Update results report

---

## References

- GPT-5.2 Pro methodology consultation (2026-02-01)
- Burrows, J. (1987). Word patterns and story shapes
- Stamatatos, E. (2009). A survey of modern authorship attribution methods
- Koppel, M. et al. (2009). Computational methods in authorship attribution
- Eder, M. et al. (2016). Stylometry with R: A package for computational text analysis
