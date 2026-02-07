# Full-Scale Audit: run_aggregated_analysis.py v1.5.0

**Date:** 2026-02-07
**Auditor:** GPT-5.2-Pro (via Claude orchestration)
**Target:** DSH/LLC Journal

---

## Executive Summary

The v1.5.0 implementation shows **significant methodological improvements** over v1.4.0. However, **critical issues remain** that require addressing before publication. Most importantly: the blocked permutation should be the PRIMARY inferential test (not sensitivity), and the sample output itself demonstrates this need (unrestricted p=0.069 vs blocked p=0.529).

---

## Disposition of v1.4.0 CRITICAL Issues

| Issue | v1.5.0 Response | Audit Judgment | Residual Risk |
|-------|-----------------|----------------|---------------|
| **C1: N=14 too small; MORONI=2** | MORONI excluded; limitations documented | **Partly addressed** | High-dim model (171 features vs 13 training samples/fold); power claim unsupported |
| **C2: Permutation exchangeability** | Blocked permutation added | **Only partially addressed** | Blocked should be PRIMARY, not sensitivity; unblocked p-value is invalid if exchangeability violated |
| **C3: Per-run aggregation discards power** | Acknowledged in limitations | **Acknowledged, not resolved** | Hierarchical models or weighting not explored |
| **C4: Multiplicity not addressed** | FDR correction added | **Largely addressed** | Family of tests not fully enumerated; two p-values (blocked/unblocked) without decision rule |

### C2 Deep Dive (Most Critical)

The sample output demonstrates the issue:
- Unrestricted permutation p-value: **0.069**
- Blocked permutation p-value: **0.5294**

This 8x difference proves that exchangeability is violated. **The unrestricted p-value is not valid.** Yet the primary analysis still uses unrestricted permutation, with blocked listed as "sensitivity."

**Recommendation:** Promote blocked permutation to PRIMARY inferential test.

---

## Disposition of v1.4.0 MAJOR Issues

| Issue | v1.5.0 Response | Audit Judgment |
|-------|-----------------|----------------|
| **M1: LOO-CV high variance** | Still uses LOO | **Not addressed** beyond necessity |
| **M2: Feature selection leakage** | Ranking inside folds added | **Addressed** |
| **M3: Burrows' Delta incomplete** | Formula documented | **Mostly addressed** |
| **M4: Confound probe insufficient** | Renamed with limitations | **Partly addressed** |

---

## What v1.5.0 Does Well (Validated Improvements)

1. **Unit-of-analysis correction**: Moving from 244 blocks to N=14 runs addresses pseudoreplication
2. **Leakage control**: StandardScaler and feature ranking inside CV folds
3. **Permutation p-value**: Using Phipson & Smyth (2010) +1 correction
4. **Multiplicity transparency**: Clear primary vs exploratory distinction with FDR
5. **Explicit limitations**: Documented in docstring and methodology card

---

## Remaining Methodological Concerns

### 3.1 "Runs as independent" is asserted, not demonstrated
The independence claim needs justification or softened wording.

### 3.2 Primary inferential target is ambiguous
Two p-values for the same BA (blocked vs unblocked) without a pre-specified decision rule invites p-value shopping.

### 3.3 High-dimensional model + tiny N
171 features with ~13 training samples per fold is extremely underdetermined.

### 3.4 Balanced accuracy with N=14 is very coarse
Need uncertainty quantification (confusion matrix, per-class recalls with binomial uncertainty, bootstrap CIs).

### 3.5 Power statement is unsupported
"~25+ pts above chance" needs derivation or simulation.

### 3.6 Reproducibility details not stated
RNG seeds and solver settings should be in metadata.

### 3.7 "Collinearity" metric undefined
"68.8% collinearity" needs explicit definition.

### 3.8 Blocked permutation degeneracy handling unclear
What happens with degenerate permutations?

---

## Publication Assessment

### What is now closer to publication-grade
1. Unit-of-analysis correction
2. Leakage control
3. Permutation p-value computation
4. Multiplicity transparency

### What still fails a skeptical DSH/LLC bar
1. **Internal validity compromised by confounding** - blocked p=0.529 vs unblocked p=0.069
2. **N=14 is extremely small** for stylometric inference
3. **Primary inferential procedure not fully pinned down** - which p-value is *the* p-value?

### Bottom Line
- **If intended claim is strong authorship separation**: NOT publishable without additional design
- **If intended claim is conservative/null** ("no significant signal once exchangeability respected"): Much closer to publishable, provided blocked permutation is promoted to primary

---

## Simulated Adversarial Review

### Attack A: "Your p-values are invalid because labels aren't exchangeable"
**Can v1.5.0 withstand?** PARTIALLY. Added blocked permutation but framed as sensitivity.
**Demand:** Make blocked permutation PRIMARY.

### Attack B: "171 predictors with 14 observations is unconstrained"
**Can v1.5.0 withstand?** MODERATELY. Permutation test controls Type I error if exchangeability holds.
**Demand:** Report confidence intervals, confusion matrices.

### Attack C: "Aggregation throws away information"
**Can v1.5.0 withstand?** ONLY AS CONSERVATIVE CHOICE.
**Demand:** Justify vs hierarchical modeling; consider run-length weighting.

### Attack D: "Multiplicity control is underdefined"
**Can v1.5.0 withstand?** BETTER than v1.4.0.
**Demand:** Enumerate exact test family.

### Attack E: "Confound analysis is cosmetic"
**Can v1.5.0 withstand?** IF CLAIM IS MODEST (no signal), yes. If strong, no.
**Demand:** Causal restraint in conclusions.

---

## High-Priority Recommendations

1. **Promote blocked permutation to PRIMARY confirmatory inference**
2. **Define book strata and degenerate-permutation handling precisely**
3. **Add uncertainty quantification** (bootstrap CI, per-class recall intervals)
4. **Justify or remove "~25+ pts" power claim**
5. **Document RNG seeds** in metadata
6. **Define "collinearity" metric** explicitly

---

## Conclusion

v1.5.0 represents substantial progress. The key remaining issue is **inferential framing**: the blocked permutation should be PRIMARY (not sensitivity) because the script's own output proves exchangeability is violated. With this change plus uncertainty quantification, the script approaches publication standards for a conservative/null-result framing.

**Recommendation:** Update v1.5.0 to v1.5.1 with:
- Blocked permutation as primary
- Explicit strata definition
- Bootstrap CIs
- RNG documentation
