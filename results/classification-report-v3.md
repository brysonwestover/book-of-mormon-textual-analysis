# Stylometric Classification Results (v3 - Run-Level Inference)

**Generated:** 2026-02-04T06:44:51.352292+00:00

---

## Methodology Corrections (Phase 2.0)

This analysis addresses critical issues identified in GPT-5.2 Pro review:

1. **Run-weighted metrics:** Effective N = 14 runs, not 244 blocks
2. **Group-level permutation:** Permute voice labels at run level
3. **Restricted permutations:** Preserve class run-counts (4/5/2/3)
4. **Capped blocks per run:** Max 20 to prevent run_0015 domination
5. **Bootstrap CI:** Stratified resampling at run level

---

## Data Summary

- **Total runs:** 14
- **Total blocks:** 244 (before capping)
- **Blocks after capping:** 115 (max 20/run)
- **Features:** 169

### Run Distribution by Voice

| Voice | Runs | Blocks (orig) | Blocks (capped) |
|-------|------|---------------|-----------------|
| JACOB | 3 | 12 | 12 |
| MORMON | 4 | 171 | 45 |
| MORONI | 2 | 22 | 21 |
| NEPHI | 5 | 39 | 37 |

---

## Primary Results

### Run-Weighted Balanced Accuracy

- **Observed:** 24.2%
- **Chance baseline:** 25.0% (4 classes)
- **vs Chance:** -0.8 percentage points

### Bootstrap Confidence Interval

- **95% CI:** [5.4%, 39.5%]
- **Bootstrap mean:** 20.8%

### Permutation Test (Group-Level)

- **Observed score:** 0.242
- **Null distribution:** 0.182 ± 0.067
- **Null range:** [0.004, 0.579]
- **p-value (one-sided):** 0.1767
- **Permutations:** 100,000

**Interpretation:** Classification performance is **NOT statistically significant** (p ≥ 0.05).
There is insufficient evidence of stylistic differentiation using function words.

---

## Per-Run Performance

| Run ID | Voice | Blocks | Accuracy |
|--------|-------|--------|----------|
| run_0000 | NEPHI | 20 | 15.0% |
| run_0002 | NEPHI | 8 | 37.5% |
| run_0003 | JACOB | 6 | 0.0% |
| run_0004 | NEPHI | 1 | 100.0% |
| run_0006 | NEPHI | 3 | 66.7% |
| run_0008 | NEPHI | 5 | 100.0% |
| run_0009 | JACOB | 4 | 0.0% |
| run_0011 | JACOB | 2 | 0.0% |
| run_0015 | MORMON | 20 | 60.0% |
| run_0017 | MORMON | 9 | 55.6% |
| run_0019 | MORMON | 12 | 16.7% |
| run_0020 | MORONI | 20 | 0.0% |
| run_0021 | MORMON | 4 | 0.0% |
| run_0022 | MORONI | 1 | 0.0% |

---

## Comparison with v2 Results

| Metric | v2 (Block-Level) | v3 (Run-Level) |
|--------|------------------|----------------|
| Balanced Accuracy | 21.6% | 24.2% |
| Permutation p-value | 1.0 (INVALID) | 0.1767 |
| Null distribution std | ~0 (BUG) | 0.0669 |

---

## Key Structural Insight

The effective sample size is **14 runs**, not 244 blocks.

Run_0015 alone contains 146 of 244 blocks (60%). Any analysis that treats
blocks as independent observations is pseudoreplicated. The v2 permutation
test failed because it permuted at the wrong level.

---

## Implications for Phase 2

**No significant signal.** Proceed to:
- Phase 2.A: Robustness testing (confirm null is stable)
- Phase 2.D: Garnett calibration (context for translation layer)

---

*Methodology per GPT-5.2 Pro consultation. See docs/decisions/phase2-execution-plan.md*