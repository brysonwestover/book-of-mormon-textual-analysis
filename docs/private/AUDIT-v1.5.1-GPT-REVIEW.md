# Full-Scale Audit: run_aggregated_analysis.py v1.5.1

**Date:** 2026-02-07
**Auditor:** GPT-5.2-Pro (via Claude orchestration)
**Target:** DSH/LLC Journal

---

## PART 1: AUDIT OF v1.5.1

### 1. Is Blocked Permutation Appropriately Implemented as PRIMARY?

**Conceptually: YES** - Promoting blocked permutation to primary when exchangeability is violated is exactly correct.

**Implementation concerns to verify:**
- What is being permuted? (labels must be permuted before training, not after)
- Does permutation re-run the full modeling pipeline?
- Is the blocking structure explicitly defined?

**Minimum publication requirement:** Explicitly demonstrate that the blocked scheme restores exchangeability for the actual experimental unit being permuted.

### 2. Are Uncertainty Quantifications Appropriate?

**Bootstrap CI:** ⚠️ CONCERNS
- With only 12 runs, naive bootstrap is very unstable
- Wide CI [38.3%, 93.3%] is consistent with instability
- Bootstrap CI lower bound above 33.3% while blocked p=0.51 suggests misalignment

**Recommendation:** Use permutation-based uncertainty or cluster bootstrap that respects dependence structure.

**Wilson CIs for per-class recall:** ⚠️ CONCERNS
- Wilson intervals assume independent Bernoulli draws
- Predictions are correlated within text/book structure
- May be too narrow; consider cluster-robust alternative

**"Chance = 33.3%":** ⚠️ CONCERN
- Under blocked null, expected performance can be >33.3% if block structure preserves information
- Better baseline: mean balanced accuracy under blocked permutations

### 3. Is the Null Result Interpretation Sound?

**YES, with caveats:**
- "NOT SIGNIFICANT at alpha=0.05" is consistent with p=0.505
- Avoid "no effect" framing → use "we did not detect performance above blocked-null baseline"
- Interpret relative to blocked null, not 33.3% chance

### 4. Remaining Methodological Concerns

1. **Unit of analysis + dependence**
   - Clarify what the independent sampling unit is
   - Align uncertainty quantification to that

2. **Permutation validity with model training**
   - Confirm permutations re-train models under permuted labels
   - If not, explicitly state it's a conditional randomization test

3. **Degenerate permutations (26/100)**
   - 100 permutations is low; Monte Carlo error substantial
   - Increase to thousands or do exact test if space is small
   - Sample unique permutations without replacement

4. **Collinearity (68.8%)**
   - Articulate what is and is not identifiable
   - Estimand is effectively "signal within non-confounded portion"

5. **Metric computation**
   - Specify whether BA is per-run then averaged, or from pooled confusion matrix

### 5. Does v1.5.1 Meet DSH/LLC Standards?

**LIKELY ACCEPTABLE WITH REVISIONS:**

1. Precisely define permutation unit, blocking structure, inferential target
2. Ensure permutation test matches modeling pipeline
3. Replace "chance = 33.3%" with blocked-null baseline
4. Strengthen uncertainty quantification to respect dependence
5. Address degeneracy by increasing permutation count

---

## PART 2: ALTERNATIVE METHODS TO INCREASE N

### First Principle
> "You can't manufacture independent N. With contiguous text, independent replicates are whatever units can plausibly be treated as exchangeable draws."

Effective sample size formula:
```
N_eff ≈ (J × m) / (1 + (m-1)ρ)
```
where J=#runs, m=blocks/run, ρ=within-run ICC. If ρ is high, N_eff collapses back to J (≈12-14).

### Method Evaluation Summary

| Method | Increases N_eff? | Avoids Pseudorep? | Addresses Confound? | Recommended? |
|--------|-----------------|-------------------|---------------------|--------------|
| **Hierarchical/mixed-effects** | Modestly | Yes (if AR structure) | No | ✓ for estimation |
| **Cluster-robust inference** | No | Partially | No | Sensitivity only |
| **Block bootstrap** | No | Yes (if run-level) | No | For uncertainty |
| **Sliding windows** | **No** (mirage) | **High risk** | Can worsen | ⚠️ Avoid |
| **Thinned windows** | **Maybe** | If spaced | Neutral | ✓ Best bet |
| **Leave-one-book-out** | No | Yes | Can worsen | ⚠️ Not recommended |
| **Bayesian approaches** | No | Yes | Can represent | ✓ Recommended |
| **Other stylometric methods** | No | Depends | Partially | Robustness checks |

### What CAN Legitimately Increase N_eff?

#### A) Thinned Windows + Explicit Autocorrelation
Instead of 244 adjacent blocks, construct **K widely-spaced windows per run** (e.g., at 10%, 30%, 50%, 70%, 90% of run length):
1. Compute stylometric outcome per window
2. Estimate within-run autocorrelation
3. Choose spacing where correlation is small
4. Fit multilevel model with run random effects

This can move from ~12 to **maybe 20-40** effective points if dependence drops with distance.

#### B) Two-Stage "Run Profiling" Models
- Stage 1: Estimate each run's stylistic profile with uncertainty using its blocks
- Stage 2: Compare narrators at run level, accounting for measurement error

Doesn't increase N, but improves **power per run** by reducing noise.

#### C) Design-Based Controls for Confounding
- Restrict to comparable discourse modes
- Include relative position as covariate
- Residualize against topic proxies

Doesn't increase N, but prevents spurious results from structure.

### Concrete Recommendations for Follow-Up Study

**PRIMARY PATH (Methodologically Strongest):**
1. Keep runs as inferential unit (or thinned windows as repeated measures)
2. Use features less sensitive to topic (function words, char n-grams, POS trigrams)
3. Fit **Bayesian hierarchical model** with:
   - Random intercepts for run
   - AR(1) residual correlation within run
   - Fixed effects for relative position and/or genre
4. Report **effect sizes + uncertainty**, not just p-values

**SECONDARY/SENSITIVITY:**
- Wild cluster bootstrap (cluster=run) for hypothesis tests
- Group-wise CV blocking by run
- Multiple feature families for consistency

**WHAT NOT TO DO:**
- ❌ Treat 244 blocks as IID to get dramatic p-values
- ❌ Leave-one-book-out CV under strong narrator-book collinearity
- ❌ Switch algorithms as if that fixes the sampling problem

### "Best Bet" to Raise N_eff

**Spaced (thinned) windows + hierarchical modeling**, empirically justified by showing autocorrelation decay of stylometric outcome.

---

## Summary

**v1.5.1 Status:** Methodologically sound for a null result with recommended revisions.

**N Problem:** Cannot be "solved" but can be mitigated through:
1. Thinned windows (if autocorrelation decays)
2. Bayesian hierarchical modeling
3. Better uncertainty quantification
4. Explicit confound controls

**Honest Conclusion:** With N=12 and 68% confounding, the design cannot definitively answer the authorship question. A null result is appropriately humble; a "significant" result would require extraordinary caution.
