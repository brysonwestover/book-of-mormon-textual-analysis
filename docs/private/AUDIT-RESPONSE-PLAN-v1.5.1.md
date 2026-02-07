# Audit Response Plan: run_aggregated_analysis.py v1.5.1

**Date:** 2026-02-07
**Status:** IMPLEMENTED

---

## Summary of v1.5.1 Changes (Second Audit Response)

### KEY CHANGE: Blocked Permutation is Now PRIMARY

The v1.5.0 GPT audit identified that **unrestricted permutation violates exchangeability** because narrator labels correlate with book position (~68% collinearity). The script's own output proved this:

| Test | p-value |
|------|---------|
| Unrestricted permutation | 0.069-0.117 |
| Blocked permutation | 0.505-0.529 |

The ~4-8x difference confirms that unrestricted permutation produces **anti-conservative** p-values when exchangeability is violated.

**v1.5.1 Solution:** Blocked permutation is now the PRIMARY inferential test. Unrestricted permutation is demoted to reference only.

---

## Changes Implemented in v1.5.1

### 1. Promoted Blocked Permutation to PRIMARY
- **Before (v1.5.0):** Unrestricted permutation was primary; blocked was "sensitivity"
- **After (v1.5.1):** Blocked permutation is PRIMARY; unrestricted is "reference only"
- **Rationale:** Exchangeability is violated; blocked permutation is the only valid test

### 2. Added Uncertainty Quantification
- Bootstrap 95% CI for balanced accuracy
- Per-class recall with Wilson binomial CIs
- Permutation-based CI from null distribution

### 3. Documented RNG Seeds
- `random_seed: 42` explicitly in metadata
- `rng_note: "numpy.random.RandomState(42) used throughout"`
- Solver settings documented in methodology card

### 4. Defined Collinearity Metric
- Added to docstring:
  ```
  COLLINEARITY METRIC:
    Collinearity index = 1 - (unique narrator-book pairs / max possible pairs)
    HIGH (>50%) indicates narrator strongly predicts book position.
  ```

### 5. Documented Strata Definition
- Primary book (first book if multiple) defines stratum
- Degenerate permutations (identical to original) counted and reported
- Strata sizes printed in output

### 6. Removed Unsupported Power Claim
- **Before:** "can detect large effects (~25+ pts above chance)"
- **After:** "no formal power analysis performed"

### 7. Updated Output Structure
```python
"primary_analysis": {
    "inference_method": "BLOCKED permutation (within book-strata)",
    "blocked_permutation": {...},
    "unrestricted_permutation_reference": {...},  # For comparison only
    "balanced_accuracy_bootstrap_ci": [lower, upper],
    "note": "Blocked permutation is PRIMARY because..."
}
```

---

## Audit Issue Resolution Matrix

| Issue | v1.5.0 Status | v1.5.1 Status |
|-------|--------------|---------------|
| **C1: N=14 too small** | Acknowledged | Acknowledged + removed unsupported power claim |
| **C2: Exchangeability violated** | Partial (blocked as sensitivity) | **RESOLVED** (blocked is PRIMARY) |
| **C3: Per-run aggregation** | Acknowledged | Acknowledged |
| **C4: Multiplicity** | Addressed (FDR) | Addressed (FDR) |
| **M1: LOO-CV high variance** | Acknowledged | + bootstrap CI |
| **M2: Feature selection leakage** | Addressed | Addressed |
| **M3: Burrows' Delta** | Addressed | Addressed |
| **M4: Confound probe** | Renamed + limitations | Renamed + limitations |
| **NEW: RNG documentation** | Missing | **RESOLVED** |
| **NEW: Collinearity definition** | Missing | **RESOLVED** |
| **NEW: Per-class uncertainty** | Missing | **RESOLVED** |

---

## Sample v1.5.1 Output

```
======================================================================
RUN-AGGREGATED STYLOMETRIC ANALYSIS v1.5.1
Publication-Quality Analysis for DSH/LLC (Exchangeability Fix)
======================================================================

>>> PRIMARY ANALYSIS (Confirmatory) <<<
3-class (JACOB, MORMON, NEPHI): BA = 45.0% (chance = 33.3%)
Bootstrap 95% CI: [38.3%, 93.3%]
BLOCKED permutation p-value (PRIMARY): 0.5050
Unrestricted permutation p-value (reference): 0.1176
CONCLUSION: NOT SIGNIFICANT at alpha=0.05

>>> EXCHANGEABILITY CHECK <<<
Blocked p-value (PRIMARY): 0.5050
Unrestricted p-value (ref): 0.1176
WARNING: Large difference (-0.3873) confirms exchangeability violation
```

---

## Publication Readiness Assessment

### What v1.5.1 Achieves
1. **Valid inferential procedure** - Blocked permutation respects exchangeability
2. **Transparent uncertainty** - Bootstrap CI, per-class Wilson CIs
3. **Reproducible** - RNG seeds documented
4. **Honest limitations** - All known issues documented

### What v1.5.1 Does NOT Achieve
1. **Strong authorship claims** - Results are null; no significant separation
2. **Causal inference** - Cannot separate narrator from topic effects
3. **Generalization** - Results specific to this text

### Publication Recommendation
**Publishable as a conservative/null result** with appropriate framing:
> "We find no significant evidence of narrator-specific stylistic signatures
> once accounting for narrator-book confounding via blocked permutation
> (p=0.51). The high collinearity between narrator and book position (68%)
> prevents definitive attribution claims."

---

## Verification Checklist

- [x] Blocked permutation is PRIMARY inferential test
- [x] Unrestricted permutation clearly labeled as reference
- [x] Bootstrap CI for balanced accuracy
- [x] Per-class recall with Wilson CIs
- [x] RNG seeds documented in metadata
- [x] Collinearity metric defined in docstring
- [x] Strata definition documented
- [x] Degenerate permutation handling documented
- [x] Unsupported power claim removed
- [x] All v1.5.0 audit issues addressed or acknowledged
