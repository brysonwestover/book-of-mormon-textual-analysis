# Audit Response Plan: run_aggregated_analysis.py v1.4.0 → v1.5.0

**Date:** 2026-02-06
**Status:** IMPLEMENTING

---

## Summary of Issues and Responses

### CRITICAL ISSUES

| ID | Issue | Valid? | Response | Implementation |
|----|-------|--------|----------|----------------|
| C1 | N=14 too small, MORONI=2 problematic | YES | Cannot fix N; reframe claims; make 3-class primary | Docs + code |
| C2 | Permutation exchangeability violated | YES | Add blocked permutation; document narrator-book structure | Code change |
| C3 | Per-run aggregation discards power | PARTIAL | This is intentional tradeoff; add sample weighting option | Code change |
| C4 | Multiplicity not addressed | YES | Pre-specify primary; label exploratory; add FDR | Code change |

### MAJOR ISSUES

| ID | Issue | Valid? | Response | Implementation |
|----|-------|--------|----------|----------------|
| M1 | LOO-CV high variance | YES | Unavoidable at N=14; report per-fold variance | Already done |
| M2 | Feature selection leakage | PARTIAL | Move ranking inside folds for sensitivity analysis | Code change |
| M3 | Burrows Delta details incomplete | YES | Add explicit formula and variant documentation | Docs |
| M4 | Confound probe insufficient | YES | Rename; acknowledge limitation; show collinearity | Code + docs |

---

## Detailed Implementation Plan

### 1. Pre-Specify Primary Analysis (C4)

**Change:** Designate ONE primary confirmatory analysis.

**Primary Analysis:**
- **Classes:** 3-class (JACOB, MORMON, NEPHI) - excludes MORONI due to n=2
- **Classifier:** Logistic Regression, L2, C=1.0
- **Features:** All 171 function words (no selection)
- **Validation:** Leave-one-out CV
- **Inference:** Permutation test, 100,000 permutations
- **Metric:** Balanced accuracy
- **Alpha:** 0.05 (one-sided)

**Exploratory Analyses (FDR-corrected):**
- 4-class analysis (includes MORONI)
- Burrows' Delta classifier
- Feature sensitivity (k=50, 100, 150)
- C sensitivity (0.01, 0.1, 1.0, 10.0, 100.0)
- Confound assessment (book prediction)
- Blocked permutation sensitivity

### 2. Blocked Permutation for Exchangeability (C2)

**Problem:** Narrator labels correlate with book position. Unrestricted permutation may test "narrator + topic" not just "narrator."

**Narrator-Book Structure:**
```
NEPHI  → 1 Nephi, 2 Nephi (early books)
JACOB  → 2 Nephi, Jacob (early-middle)
MORMON → Mosiah, Alma, Helaman, 3 Nephi, 4 Nephi, Mormon, Words of Mormon, Moroni (spans many)
MORONI → Ether, Mormon, Moroni (late books)
```

**Implementation:**
1. Create book-strata groupings
2. Implement blocked permutation: shuffle labels WITHIN strata where possible
3. Report BOTH unrestricted and blocked permutation p-values
4. If blocked permutation is infeasible (too few within-stratum observations), document why

**Exchangeability Defense:**
> "Under the null hypothesis of no narrator effect, we assume run labels are exchangeable within book-strata. Because some narrators span multiple books (MORMON) and some books contain multiple narrators (2 Nephi contains both NEPHI and JACOB runs), complete blocking is not possible. We report both unrestricted permutation (which may be anti-conservative if book confounds narrator) and partially-blocked permutation as sensitivity analysis."

### 3. Sample Weighting by Run Length (C3)

**Problem:** Runs vary in word count. Per-1000-word normalization equalizes means but not precision.

**Implementation:**
1. Add optional `sample_weight` parameter to `leave_one_out_cv()`
2. Weight by `sqrt(total_words)` or `total_words` (test both)
3. Report weighted and unweighted results
4. Document tradeoff: weighting improves efficiency but may overweight long runs

### 4. FDR Correction for Exploratory Analyses (C4)

**Implementation:**
1. Collect all exploratory p-values
2. Apply Benjamini-Hochberg FDR correction
3. Report both raw and adjusted p-values
4. Clearly label which analyses are confirmatory vs exploratory

### 5. Feature Ranking Inside Folds (M2)

**Current:** Feature ranking by corpus-wide mean frequency (computed once on all data)

**Change:** For feature sensitivity analysis, compute ranking INSIDE each CV fold

**Implementation:**
1. Modify `feature_sensitivity_analysis()` to accept `rank_inside_fold=True` parameter
2. When True, rank features using only training data in each fold
3. Report both approaches as sensitivity check

### 6. Rename and Clarify Confound Probe (M4)

**Change:**
- Rename: "Confound Probe" → "Narrator vs. Book Comparison"
- Add explicit limitation statement
- Report narrator-book contingency table

### 7. Documentation Updates (M3, all)

**Add to script docstring:**
- Explicit Burrows' Delta formula
- Primary vs exploratory analysis designation
- Exchangeability assumptions and defense
- Known limitations

**Create methodology card:**
- All hyperparameters and their justification
- What was pre-specified vs. explored
- What effects are detectable (power analysis)

---

## Code Changes Required

### New Functions to Add:
```python
def blocked_permutation_test(X, y, observed_score, strata, n_permutations, seed)
    """Permutation test respecting block/strata structure."""

def compute_fdr_correction(p_values)
    """Apply Benjamini-Hochberg FDR correction."""

def get_book_strata(run_info, run_ids)
    """Extract book-based strata for blocked permutation."""

def weighted_leave_one_out_cv(X, y, weights, C)
    """LOO-CV with sample weighting."""
```

### Functions to Modify:
```python
def feature_sensitivity_analysis(..., rank_inside_fold=False)
    """Add option to rank features inside each CV fold."""

def confound_probe_book(...) → narrator_vs_book_comparison(...)
    """Rename and add limitation documentation."""

def main()
    """Restructure to separate primary/exploratory; add FDR; add blocked perm."""
```

### New Output Structure:
```python
results = {
    "metadata": {...},
    "primary_analysis": {
        "description": "3-class LR, pre-specified confirmatory",
        "results": {...},
        "p_value": ...,
        "conclusion": ...
    },
    "exploratory_analyses": {
        "four_class": {"results": ..., "p_value_raw": ..., "p_value_fdr": ...},
        "burrows_delta": {...},
        "feature_sensitivity": {...},
        ...
    },
    "sensitivity_analyses": {
        "blocked_permutation": {...},
        "weighted_samples": {...},
        "feature_ranking_inside_fold": {...}
    },
    "limitations": {
        "sample_size": "N=14 runs limits power to detect small effects",
        "exchangeability": "Narrator-book correlation may inflate significance",
        "confound": "Cannot definitively separate narrator from topic effects"
    }
}
```

---

## Verification Checklist

After implementation, verify:

- [ ] Primary analysis is clearly designated
- [ ] All exploratory analyses have FDR-corrected p-values
- [ ] Blocked permutation implemented and reported
- [ ] Sample weighting option available
- [ ] Feature ranking can be done inside folds
- [ ] Confound comparison renamed and limitations documented
- [ ] Burrows' Delta formula explicitly documented
- [ ] Narrator-book contingency table reported
- [ ] Methodology card created
- [ ] All limitations explicitly stated

---

## Expected Outcomes

After these changes:

1. **An expert reviewer** should be able to look at our methodology and say: "This is the right approach given the constraints. The authors have been appropriately cautious."

2. **An adversarial reviewer** will still note N=14 is small, but cannot claim we've been statistically naive or engaged in p-hacking.

3. **The conclusions** will be appropriately hedged: "We find suggestive evidence consistent with [X], but the small sample size and potential confounds preclude definitive conclusions."
