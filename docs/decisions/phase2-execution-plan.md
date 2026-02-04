# Phase 2 Execution Plan: Stylometric Extensions

**Created:** 2026-02-03
**Status:** DRAFT - Awaiting user approval before execution
**Consultation:** GPT-5.2 Pro (2 rounds)

---

## Executive Summary

Phase 1 stylometric analysis found no statistically significant differentiation among BoM narrators using function words. However, the permutation test was **invalid** (zero variance in null distribution). Before claiming a null result, we must:

1. Fix the statistical inference
2. Conduct robustness testing
3. Calibrate against translation-layer comparator (Garnett corpus)

This document specifies the complete Phase 2 execution plan with pre-registered decisions.

---

## Critical Finding: Phase 1 Inference Invalid

### Problem Identified

The Phase 1 permutation test produced:
```
observed_score: 0.3177
null_mean: 0.3177 (identical!)
null_std: 5.55e-17 (zero)
p_value: 1.0
```

**Root cause:** sklearn's `permutation_test_score` permutes at sample level, but with GroupKFold, samples within runs stay together. With only 14 runs and one run (run_0015) containing 60% of blocks, effective permutation space is near-zero.

### Structural Constraints

| Voice | Runs | Blocks | Dominant Run |
|-------|------|--------|--------------|
| MORMON | 4 | 171 | run_0015: 146 blocks (60% of total) |
| NEPHI | 5 | 39 | run_0000: 22 blocks |
| MORONI | 2 | 22 | run_0020: 21 blocks |
| JACOB | 3 | 12 | run_0003: 6 blocks |
| **Total** | **14** | **244** | |

**Key insight (GPT):** The effective sample size is **14 runs**, not 244 blocks. Any analysis treating blocks as independent observations is pseudoreplicated.

---

## Pre-Registered Specifications

### Primary Metric

**Run-weighted balanced accuracy:**
1. For each held-out run, compute predictions for all its blocks
2. Compute run-level accuracy (proportion correct within run)
3. Compute balanced accuracy across runs (average of per-class run-level accuracies)

This aligns the test statistic with the exchangeability structure.

### Primary Model

- **Algorithm:** Logistic Regression with balanced class weights
- **Features:** 169 function words (content-suppressed)
- **Block size:** 1000 words (primary), 500/2000 (robustness)
- **CV:** Leave-one-run-out (14-fold)
- **Blocks per run cap:** 20 (to prevent run_0015 domination in training)

### Permutation Test

**Restricted group-level permutation:**
- Permute voice labels at the run level
- Restrict to permutations preserving class run-counts (4 MORMON, 5 NEPHI, 2 MORONI, 3 JACOB)
- Number of valid permutations: 14!/(4!×5!×2!×3!) = 2,522,520
- Use exact enumeration if computationally feasible, else Monte Carlo with 100,000 draws
- **One-sided p-value** (testing above-chance classification)

### Decision Thresholds

| Outcome | Criterion | Action |
|---------|-----------|--------|
| Significant signal | p < 0.05 AND balanced accuracy > 30% | Investigate: is it narrator, genre, or confound? |
| Null result | p ≥ 0.05 OR balanced accuracy ≤ 30% | Proceed to robustness + Garnett calibration |
| Indeterminate | CI spans both | Report as underpowered |

---

## Execution Phases

### Phase 2.0: Validation Gate (BLOCKING)

Must complete before any extensions.

#### Task 0.1: Fix Permutation Test

**Script:** `scripts/run_classification_v3.py`

Implementation:
1. Implement run-weighted balanced accuracy
2. Implement restricted group-level permutation
3. Cap blocks per run at 20 for training balance
4. Bootstrap CI for effect size (resample runs with replacement)

**Outputs:**
- `results/classification-results-v3.json`
- `results/classification-report-v3.md`

**Decision:** If corrected p < 0.05 with meaningful effect size, Phase 2 shifts to *explanation* (what's driving the signal?). If p ≥ 0.05, proceed to robustness testing.

#### Task 0.2: Power Analysis

**Script:** `scripts/power_analysis.py`

Implementation:
1. **Learning curve:** Balanced accuracy vs. number of runs in training (leave-k-runs-out for k=1,2,3,...)
2. **Simulation:** Inject known stylistic shift (perturb function word rates by δ), estimate minimum detectable effect (MDE)
3. **Report:** "Given 14 runs, we can detect effects of magnitude ≥ X with 80% power"

**Outputs:**
- `results/power-analysis.json`
- `results/power-analysis-report.md`
- Learning curve plot

---

### Phase 2.A: Robustness Testing

**Timing:** After Phase 2.0, in parallel with Phase 2.D

**Purpose:** Test whether null result is robust to analytic choices. These are **sensitivity analyses**, not hypothesis fishing.

#### Variants (Pre-Specified)

| ID | Variant | Rationale |
|----|---------|-----------|
| A1 | Block size: 500 words | Smaller blocks → more samples, but noisier |
| A2 | Block size: 2000 words | Larger blocks → cleaner signal, but fewer samples |
| A3 | Include quotations | Test if exclusion policy affects results |
| A4 | Character 3-grams | Different feature family, less interpretable |
| A5 | Combined: FW + char 3-grams | Maximum feature richness |
| A6 | SVM classifier | Alternative algorithm sensitivity check |

#### Analysis Plan

1. Run each variant with same run-weighted metric and permutation test
2. Report as **robustness table** (not separate hypothesis tests)
3. **Do not** claim significance based on best-looking variant
4. If ALL variants remain null (p ≥ 0.05), this strengthens the null finding
5. If ANY variant shows signal, investigate whether it's narrator, genre, or artifact

#### Multiple Comparisons Correction

Use **max-statistic permutation correction:**
1. For each permutation, compute scores for all 6 variants
2. Record maximum score across variants
3. Corrected p-value = proportion where max_perm ≥ max_observed

This controls familywise error while respecting variant correlation.

---

### Phase 2.B: Genre-Controlled Analysis

**Timing:** After Phase 2.A, only if A shows stable null

**Purpose:** Test whether genre/topic confounds mask or create apparent signal.

#### Genre Annotation

Add `discourse_type` field to verses with categories:
1. NARRATION — historical chronicle, events
2. SERMON — didactic argument, moral instruction, exhortation
3. PROPHECY — predictive, visionary content
4. EPISTLE — letters, personal address
5. SCRIPTURE_QUOTE — already tagged via quote_source

**Method:** Rule-based heuristics + manual review of edge cases

#### Analysis

1. Filter to "pure" blocks (≥80% single genre)
2. Test within-genre classification:
   - Nephi-sermon vs Mormon-sermon vs Jacob-sermon
   - Nephi-narration vs Mormon-narration
3. If within-genre shows signal but global didn't → genre was masking
4. If within-genre remains null → strengthens overall null

**Constraint:** May have insufficient data for some genre×narrator combinations. Report coverage.

---

### Phase 2.C: Alternative Methods (Targeted)

**Timing:** After Phase 2.A, only for specific diagnostic questions

**Purpose:** Test whether null is method-specific. NOT a fishing expedition.

#### Methods

| Method | Purpose | Run If |
|--------|---------|--------|
| Burrows' Delta | Gold standard stylometric distance | Always (primary alternative) |
| Rolling Delta | Visualize style shifts | If Delta shows anything interesting |
| PCA/clustering | Unsupervised structure | Exploratory visualization only |

#### Burrows' Delta Implementation

1. Compute Delta distance between all block pairs
2. For each narrator, compute mean within-narrator vs between-narrator Delta
3. Test: Is within-narrator Delta < between-narrator Delta?
4. Use permutation test at run level

**Skip:** Neural/deep learning approaches (insufficient data with N=14 runs)

---

### Phase 2.D: Translation-Layer Calibration (Garnett Corpus)

**Timing:** In parallel with Phase 2.A — start immediately after Phase 2.0

**Purpose:** Calibrate expectations for stylometry through single translator. **NOT** a BoM test; contextual evidence.

#### Corpus: Constance Garnett Translations

| Author | Works | Est. Words |
|--------|-------|------------|
| Tolstoy | War and Peace, Anna Karenina, etc. | 500k+ |
| Dostoevsky | Crime and Punishment, Brothers Karamazov, etc. | 500k+ |
| Chekhov | Short stories, plays | 200k+ |
| Turgenev | Fathers and Sons, etc. | 200k+ |

**Source:** Project Gutenberg (public domain)

#### Pre-Registered Predictions

| ID | Prediction | Interpretation if True |
|----|------------|----------------------|
| D1 | Accuracy(author \| Garnett) >> chance | Translation does NOT erase author signal; BoM null is meaningful |
| D2 | Accuracy(author \| Garnett) ≈ chance | Translation homogenization plausible; BoM null is expected |

#### Methodology (Matched to BoM)

1. **Same block sizes:** 500, 1000, 2000 words
2. **Same features:** 169 function words (adapted for non-archaic English)
3. **Same grouping:** Group by work (novel/story), not random blocks
4. **Same CV:** Leave-one-work-out
5. **Same metric:** Run-weighted (work-weighted) balanced accuracy

#### Confound Controls

1. **Garnett career period:** Use leave-time-slice-out (hold out early/mid/late works)
2. **Content/NE leakage:** Mask character names and place names before feature extraction
3. **Topic:** Works span multiple genres; report genre distribution

#### Outputs

- `data/reference/garnett/` — processed corpus
- `results/garnett-classification.json`
- `results/garnett-report.md`

---

### Phase 2.E: Write-Up & Consolidation

**Timing:** After all above phases complete

#### Report Structure

1. **Abstract** — One-paragraph summary
2. **Introduction** — Research question, hypotheses (H1-H5)
3. **Methods**
   - Text acquisition and preprocessing
   - Voice annotation (dual-layer schema)
   - Block derivation (run-based)
   - Feature extraction
   - Classification methodology (run-weighted, permutation)
   - Control analyses (KJV, Finney, Garnett)
4. **Results**
   - Primary finding (corrected Phase 1)
   - Robustness testing (Phase 2.A)
   - Genre analysis (Phase 2.B, if run)
   - Alternative methods (Phase 2.C)
   - Translation-layer calibration (Phase 2.D)
5. **Discussion**
   - Interpretation under each hypothesis
   - What the null does/doesn't prove
   - Limitations (power, translation layer, genre confounds)
6. **Conclusion**
7. **Appendices**
   - Full feature lists
   - All robustness results
   - Code availability

---

## Dependency Graph

```
Phase 2.0 (Validation Gate)
    ├── Task 0.1: Fix Permutation ─────┬──► Phase 2.A (Robustness)
    │                                  │        │
    └── Task 0.2: Power Analysis ──────┘        ├──► Phase 2.B (Genre) [if A null]
                                                │
Phase 2.D (Garnett) ◄──────────────────────────┤    Phase 2.C (Alt Methods)
    [runs in parallel with A]                   │        │
                                                │        │
                                                └────────┴──► Phase 2.E (Write-Up)
```

---

## Decision Gates

### Gate 0: After Phase 2.0

| Finding | Action |
|---------|--------|
| Corrected p < 0.05, effect size meaningful | Shift to explanation: is signal narrator, genre, or artifact? |
| p ≥ 0.05 OR effect near chance | Proceed to robustness (Phase 2.A) |
| Power analysis shows MDE > plausible effects | Acknowledge in limitations; still run robustness |

### Gate 1: After Phase 2.A

| Finding | Action |
|---------|--------|
| ALL variants null (corrected p ≥ 0.05) | Strong null result; proceed to minimal 2.B + 2.E |
| ANY variant shows signal | Investigate: run 2.B and 2.C for diagnostics |
| Results inconsistent across variants | Report as unstable; emphasize limitations |

### Gate 2: After Phase 2.D

| Finding | Interpretation |
|---------|----------------|
| Garnett shows high author separability | Translation does not inherently erase signal; BoM null is informative |
| Garnett shows low/no separability | Translation homogenization is plausible; BoM null is expected under H4 |

---

## Timeline Estimate

| Phase | Tasks | Dependencies |
|-------|-------|--------------|
| 2.0 | Fix permutation, power analysis | None |
| 2.A | 6 robustness variants | 2.0 complete |
| 2.D | Garnett corpus acquisition + analysis | 2.0 complete (parallel with A) |
| 2.B | Genre annotation + analysis | 2.A complete, only if null |
| 2.C | Burrows' Delta | 2.A complete |
| 2.E | Write-up | All above complete |

---

## Files to Create/Modify

### New Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_classification_v3.py` | Corrected classification with run-weighted metrics |
| `scripts/power_analysis.py` | Learning curves and MDE simulation |
| `scripts/run_robustness.py` | Phase 2.A variants |
| `scripts/annotate_genre.py` | Discourse type annotation |
| `scripts/run_genre_analysis.py` | Phase 2.B |
| `scripts/burrows_delta.py` | Phase 2.C |
| `scripts/preprocess_garnett.py` | Garnett corpus preparation |
| `scripts/run_garnett_analysis.py` | Phase 2.D |

### New Data Files

| File | Purpose |
|------|---------|
| `data/reference/garnett/` | Garnett corpus directory |
| `data/text/processed/bom-verses-annotated-v4.json` | With discourse_type field |

### New Results Files

| File | Purpose |
|------|---------|
| `results/classification-results-v3.json` | Corrected Phase 1 |
| `results/power-analysis.json` | Power analysis |
| `results/robustness-results.json` | Phase 2.A |
| `results/genre-analysis.json` | Phase 2.B |
| `results/delta-analysis.json` | Phase 2.C |
| `results/garnett-classification.json` | Phase 2.D |

---

## Approval Checklist

Before executing, confirm:

- [ ] Primary metric (run-weighted balanced accuracy) is correct
- [ ] Permutation test specification (restricted group-level) is correct
- [ ] Robustness variants (A1-A6) are complete and appropriate
- [ ] Garnett corpus is appropriate comparator
- [ ] Decision gates are reasonable
- [ ] No other major methodological gaps

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-03 | 0.1 | Initial draft based on GPT-5.2 Pro consultation |
