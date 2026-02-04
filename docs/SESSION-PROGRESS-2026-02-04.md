# Session Progress Report: 2026-02-04

**Purpose:** Resume point for continuing stylometric analysis of Book of Mormon.

---

## Project Overview

This project applies computational stylometry to test whether claimed narrator voices in the Book of Mormon exhibit detectably different writing styles using function-word classification.

**Pre-registration DOI:** [10.17605/OSF.IO/4W3KH](https://osf.io/4W3KH)

---

## Completed Work

### Phase 1 (Exploratory) ✅

- Voice annotation schema v3 with dual-layer narrator/voice distinction
- Block derivation from verse annotations (244 blocks, 14 voice runs)
- Initial classification (v1, v2) — identified permutation test bug (zero variance)

### Phase 2.0 (Confirmatory) ✅

**Script:** `scripts/run_classification_v3.py`

**Results:**
- Run-weighted balanced accuracy: **24.2%** (at chance, 25%)
- Permutation p-value: **0.177** (not significant)
- Bootstrap 95% CI: [5.4%, 39.5%]
- Permutations: 100,000

**Conclusion:** No statistically significant stylistic differentiation detected.

**Key methodology corrections (GPT-5.2 Pro consultation):**
1. Effective N = 14 runs, not 244 blocks
2. Group-level permutation preserving class run-counts
3. Block capping (max 20/run) to prevent run_0015 domination
4. Stratified bootstrap at run level

**Results files:**
- `results/classification-results-v3.json`
- `results/classification-report-v3.md`

### OSF Pre-Registration ✅

- Created OSF account
- Submitted pre-registration with:
  - `osf-preregistration.md` (full methodology)
  - `phase2-execution-plan.md` (execution plan)
  - `run_classification_v3.py` (analysis code)
- DOI: **10.17605/OSF.IO/4W3KH**
- Status: Public, timestamped, immutable

### Git Commit ✅

- Committed 187 files (all project work)
- Commit hash: `2607355`
- Includes Phase 2.0 results, Garnett corpus, all scripts

### Garnett Corpus Downloaded ✅

- Location: `data/reference/garnett/raw/`
- 11 works by 4 authors (Dostoevsky, Tolstoy, Chekhov, Turgenev)
- 2,219,293 total words
- Single translator: Constance Garnett
- Purpose: Phase 2.D translation-layer calibration

---

## In Progress

### Phase 2.A: Robustness Testing 🔄

**Script:** `scripts/run_robustness.py` ✅ WRITTEN (not yet run)

**Pre-registered variants:**
| ID | Variant | Purpose |
|----|---------|---------|
| A1 | Block size 500 words | More samples, noisier |
| A2 | Block size 2000 words | Cleaner signal, fewer samples |
| A3 | Include quotations | Test exclusion policy |
| A4 | Character 3-grams | Alternative features |
| A5 | FW + char 3-grams | Maximum features |
| A6 | SVM classifier | Alternative algorithm |

**Multiple comparisons:** Max-statistic permutation correction

**GPT-5.2 Pro review corrections incorporated:**
1. HashingVectorizer for char 3-grams (no vocabulary leakage)
2. Pre-computed deterministic block capping
3. Separate standardization for combined features (A5)
4. Corrected p-value formula: (1 + sum) / (1 + B)
5. LinearSVC with fixed C=1.0 for A6 (no tuning)
6. Simplified robustness criterion: corrected p >= 0.05

**Status:** Script written, awaiting execution

**To run:**
```bash
source venv/bin/activate
python scripts/run_robustness.py
```

**Expected runtime:** Several hours (10,000 permutations × 6 variants × 14 CV folds)

---

## Not Yet Started

### Phase 2.B: Genre-Controlled Analysis
- Contingent on Phase 2.A results
- Only run if robustness testing reveals inconsistencies

### Phase 2.C: Alternative Methods (Burrows' Delta)
- After Phase 2.A
- For diagnostic questions only

### Phase 2.D: Garnett Calibration
- Corpus downloaded ✅
- Script not yet written: `scripts/run_garnett_analysis.py`
- Can run in parallel with Phase 2.A

### Phase 2.E: Write-Up
- After all analyses complete

### Power Analysis (Task 0.2)
- Script not yet written: `scripts/power_analysis.py`
- Not pre-registered (methodological assessment, not hypothesis test)

---

## Key Files

### Results
- `results/classification-results-v3.json` — Phase 2.0 results
- `results/classification-report-v3.md` — Phase 2.0 report

### Documentation
- `docs/osf-preregistration.md` — Pre-registration document
- `docs/decisions/phase2-execution-plan.md` — Full execution plan

### Scripts
- `scripts/run_classification_v3.py` — Phase 2.0 (complete)
- `scripts/run_robustness.py` — Phase 2.A (ready to run)
- `scripts/download_garnett.py` — Garnett corpus download (complete)

### Data
- `data/text/processed/bom-voice-blocks.json` — Annotated BoM blocks
- `data/reference/garnett/raw/manifest.json` — Garnett corpus manifest

---

## Decision Points Reached

### Gate 0 (After Phase 2.0): ✅ PASSED
- Corrected p = 0.177 (p ≥ 0.05)
- Effect near chance (24.2% vs 25%)
- **Decision:** Proceed to robustness testing (Phase 2.A)

### Gate 1 (After Phase 2.A): PENDING
- If all variants null (corrected p ≥ 0.05) → Strong null; proceed to 2.D + 2.E
- If any variant shows signal → Investigate source (2.B)

---

## Next Steps (Tomorrow)

1. **Run Phase 2.A robustness testing:**
   ```bash
   source venv/bin/activate
   python scripts/run_robustness.py
   ```

2. **Review results** — Check if null is robust

3. **If robust:** Write `run_garnett_analysis.py` for Phase 2.D

4. **If not robust:** Investigate which variant shows signal, consider Phase 2.B

5. **Git commit** Phase 2.A results

---

## Hypotheses Framework

The analysis assesses evidence against five competing hypotheses:

- **H1:** Modern single-author composition
- **H2:** Modern multi-source composition (single author + sources)
- **H3:** Modern collaborative composition (multiple contributors)
- **H4:** Ancient multi-author compilation with single translator
- **H5:** Deliberate ancient-style pseudepigraphy
- **H0:** Indeterminate (insufficient signal to discriminate)

**Current status:** Null result (no differentiation) is consistent with H1, H4, or H0. Garnett calibration (Phase 2.D) will help distinguish H1 from H4.

---

## Contact & References

- **OSF Pre-registration:** https://osf.io/4W3KH
- **Git commit:** 2607355
- **Methodology:** GPT-5.2 Pro consultation (2 rounds)

---

*Last updated: 2026-02-04 ~23:45 MST*
