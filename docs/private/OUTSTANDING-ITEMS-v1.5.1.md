# Outstanding Items: v1.5.1 Audit Response

**Date:** 2026-02-07
**Status:** AUDITED - Ready for archive (no production run)
**Final Audit:** PASS (see AUDIT-FINAL-v1.5.1-GPT.md)

---

## Items Addressed in v1.5.1

| Item | Status | Notes |
|------|--------|-------|
| Blocked permutation as PRIMARY | ✅ Done | Unrestricted demoted to reference |
| Bootstrap CI for BA | ✅ Done | With caveat about N=12 |
| Per-class recall with Wilson CIs | ✅ Done | With independence caveat |
| RNG seeds documented | ✅ Done | In metadata and methodology card |
| Collinearity metric defined | ✅ Done | In docstring and methodology |
| Strata definition documented | ✅ Done | Primary book determines stratum |
| Unsupported power claim removed | ✅ Done | No longer claims "~25+ pts" |
| Permutation re-runs full pipeline | ✅ Verified | LOO-CV called inside each perm |

---

## Outstanding Items (Not Yet Addressed)

### HIGH PRIORITY (Should fix before production)

| Item | Issue | Suggested Fix | Status |
|------|-------|---------------|--------|
| **1. Blocked-null baseline** | Reports "chance = 33.3%" but should use blocked-null mean | Add `blocked_null_mean` to output and use in comparisons | ✅ **FIXED** |
| **2. Permutation count** | 100 quick / 10,000 prod may be low for blocked | Increase blocked to match primary (100k) or enumerate exact | ✅ GPT confirmed (see below) |
| **3. Degenerate permutation handling** | Counted but included in null | Document that this is conservative; consider sampling unique only | ✅ GPT confirmed (see below) |

### MEDIUM PRIORITY (Document/caveat, don't necessarily fix)

| Item | Issue | Action |
|------|-------|--------|
| **4. Bootstrap CI misalignment** | Lower bound > 33% while p=0.51 | ✅ GPT confirmed: permutation-based CI is primary; bootstrap is descriptive only |
| **5. Wilson CI independence** | Assumes independent draws | Already caveated in code |
| **6. BA computation method** | Not explicitly documented | Add to methodology: BA computed from pooled confusion matrix via sklearn |

### LOW PRIORITY (Nice to have)

| Item | Issue | Action |
|------|-------|--------|
| **7. Exact FDR test family** | List not fully enumerated | Add explicit list to methodology |
| **8. Equivalence testing** | p=0.51 doesn't prove "no effect" | Consider adding TOST or Bayes factor in future |

---

## Questions for GPT Confirmation (RESOLVED)

### Q1: Degenerate Permutation Handling ✅
**Current:** Degenerate permutations (identical to original) are counted toward the null distribution.

**Question:** Is this appropriate, or should we sample unique permutations only?

**GPT-5.2-Pro Answer (2026-02-07):**
> **Include them.** The identity permutation is part of the permutation group; excluding it would make the test anti-conservative. Including degenerates is the correct approach. The only exception would be if you're trying to approximate an exact test and need unique permutations, but for Monte Carlo permutation tests, including degenerates is standard and conservative.

**Conclusion:** Current implementation is correct. No changes needed.

### Q2: Permutation Count for Blocked Test ✅
**Current:** 10,000 permutations for blocked test in production.

**Question:** Is this sufficient given the restricted permutation space within strata?

**GPT-5.2-Pro Answer (2026-02-07):**
> **If the permutation space is small, enumerate exactly.** With blocked permutation and small strata (some with 1-2 runs), the actual permutation space may be small enough to enumerate. If so, compute the exact p-value. If the space is too large to enumerate, 10,000 permutations is adequate for Monte Carlo error < 0.01. The key diagnostic is whether you're seeing many degenerate permutations (which would indicate a small effective space).

**Conclusion:** 10,000 is adequate. Current implementation already counts degenerates (~26/100 in quick mode), which provides diagnostic information. For exact inference, would need to enumerate, but 10k gives sufficient precision for p=0.51.

### Q3: Bootstrap vs Permutation-Based CI ✅
**Current:** Report both bootstrap CI and permutation-based CI.

**Question:** Given the apparent misalignment (bootstrap lower > 33% while p=0.51), should we:
- (a) Keep both with caveats
- (b) Recommend permutation-based as primary
- (c) Drop bootstrap CI entirely

**GPT-5.2-Pro Answer (2026-02-07):**
> **Recommendation: (b)** — make the blocked permutation analysis primary, treat bootstrap CI as secondary/descriptive only.
>
> **Why the misalignment isn't a contradiction:**
> - Bootstrap CI compares to 33.3% (unrestricted theoretical chance)
> - Blocked permutation compares to ~42-44% (blocked-null mean)
> - Bootstrap CI [38.3%, 93.3%] **includes** 42-44%, so it's consistent with p=0.51
>
> **Key insight:** Use blocked-null mean (~42-44%) as the baseline, not 33.3%. The "misalignment" disappears when using the correct null reference.
>
> **Recommended reporting:**
> 1. Observed BA (45.0%)
> 2. Blocked permutation p-value (0.51)
> 3. Blocked-null mean and 95% range (the proper baseline)
> 4. Bootstrap CI only as descriptive stability measure, labeled "(descriptive only)"

**Conclusion:** Fixed output to show blocked-null mean as baseline. Bootstrap CI now labeled "descriptive only".

---

## Actions Completed

1. ✅ **Fix #1:** Added blocked-null mean to summary output as comparison baseline
2. ✅ **GPT confirmation on Q1-Q3:** All three questions confirmed (see above)
3. ✅ **Bootstrap CI labeled:** Now shows "(descriptive only)" in output
4. ⏸️ **Production run:** NOT performed (user decision - archiving as-is)

---

## Files Ready for Archive

Archive to OSF/Zenodo (no production run needed - quick mode output is sufficient for pre-registration):
- [x] `scripts/run_aggregated_analysis.py` (v1.5.1)
- [x] `data/text/processed/bom-voice-blocks.json` (input data)
- [ ] `results/run-aggregated-results.json` (quick mode output exists)
- [ ] `results/figures/*.png` (quick mode figures exist)
- [x] `docs/METHODOLOGY-v1.5.1.md`
- [x] `docs/private/AUDIT-*.md`
- [x] `docs/private/OUTSTANDING-ITEMS-v1.5.1.md`

**Note:** User opted to archive without production run. Quick mode output (100 permutations) is sufficient for demonstrating methodology. Full 100k permutation run can be performed later as needed.
