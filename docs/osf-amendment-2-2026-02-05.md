# OSF Pre-Registration Amendment #2

**Date**: February 5, 2026
**Study**: Stylometric Analysis of Book of Mormon Narrators
**Original Pre-Registration**: https://osf.io/4w3kh (DOI: 10.17605/OSF.IO/4W3KH)
**Amendment #1**: https://osf.io/sthmk/files/uzhkd

---

## Summary

This second amendment documents a methodological limitation discovered during Phase 2.D (Translation-Layer Calibration) execution and establishes a feasibility criterion for bootstrap confidence intervals. This amendment is filed **after** completion of the primary analyses (novels_only, full_corpus) but **before** any period-stratified inferential results were obtained or examined.

---

## Deviation 6: Bootstrap CI Feasibility Criterion (Phase 2.D Period Stratification)

### Trigger for Amendment (What Happened)

While executing the pre-registered Phase 2.D period-stratified analysis, the pipeline terminated with the error:

> "This solver needs samples of at least 2 classes in the data, but the data contains only one class: np.str_('Turgenev')"

This indicated that in some bootstrap × leave-one-work-out (LOWO) iterations, the training split contained only a single author class, making classification undefined for those iterations.

### What We Examined (Diagnostic Only)

To determine whether this was an implementation bug or an inherent feasibility issue, we performed:

1. **Forensic CV audit**: Verified the LOWO implementation is correct by generating a fold manifest showing all 7 early-period folds have 2 classes when run without bootstrap resampling.

2. **Root cause analysis**: Identified that stratified bootstrap with replacement can sample the same work twice when an author has only 2 works (e.g., [work_A, work_A] instead of [work_A, work_B]), creating a bootstrap sample with only 1 unique work_id for that author. When LOWO then holds out that work_id, all blocks from that author are removed from training → single-class error.

3. **Feasibility diagnostics**: Logged execution success/failure across bootstrap replicates:
   - Early period (1894-1904): ~30% failure rate (2 authors: Tolstoy 2 works, Turgenev 5 works)
   - Late period (1912-1918): 49.4% failure rate (3 authors: Chekhov 4, Dostoevsky 6, Tolstoy 2 works)

4. **Permutation test verification**: Confirmed that permutation testing (which preserves exact author counts) has 0% failure rate and remains feasible.

### What We Did NOT Examine (No Inferential Outcomes)

Prior to filing this amendment, we did **not** compute, inspect, or record:
- Period-stratified classification accuracies
- Period-stratified p-values
- Period-stratified confidence intervals
- Any other inferential performance summaries for early- or late-period analyses

The feasibility diagnostic computed only success/failure flags for bootstrap replicates, not classification outcomes. The amendment is motivated solely by feasibility/execution considerations, not by any knowledge of period-stratified results.

### Change to Analysis Plan (Decision Rule)

We add the following pre-registered feasibility criterion:

> **Bootstrap CI Feasibility Rule**: A period-stratified analysis will report bootstrap confidence intervals only if ≥95% of bootstrap replicates complete successfully (defined as "no single-class error in any LOWO fold"). If this criterion is not met:
>
> (a) The non-feasibility will be reported transparently with failure rates
> (b) Point estimates and permutation p-values will still be reported (permutation test is feasible)
> (c) Bootstrap CI will be reported as "NOT ESTIMABLE under pre-registered procedure"
> (d) Any alternative bootstrap approach would be labeled as exploratory

### Rationale for 95% Threshold

The 95% threshold ensures that:
- The bootstrap distribution is not substantially distorted by conditioning on "successful" replicates
- CIs are not anti-conservative due to removing high-instability resamples
- Results are comparable to standard bootstrap practice

This threshold is conventional in bootstrap literature for ensuring stable inference.

### Application to Period-Stratified Analyses

| Period | Authors | Works Distribution | Failure Rate | Bootstrap CI Status |
|--------|---------|-------------------|--------------|---------------------|
| Early (1894-1904) | 2 | Tolstoy: 2, Turgenev: 5 | ~30% | **NOT ESTIMABLE** |
| Late (1912-1918) | 3 | Chekhov: 4, Dostoevsky: 6, Tolstoy: 2 | 49.4% | **NOT ESTIMABLE** |

For both period-stratified analyses:
- Point estimate (LOWO accuracy): WILL be computed and reported
- Permutation p-value: WILL be computed and reported (0% failure rate verified)
- Bootstrap CI: NOT ESTIMABLE - failure rates exceed 5% threshold

---

## Timing Disclosure

This amendment is filed:
- **AFTER** completion of novels_only and full_corpus analyses (results: 58.2%, p=0.0016 and 54.4%, p=0.0001 respectively)
- **AFTER** observing the execution error and conducting feasibility diagnostics
- **BEFORE** any period-stratified inferential results (accuracies, p-values) were computed or examined

The decision to add a feasibility criterion was made based on understanding WHY the method fails, not based on any knowledge of what the period-stratified results would show.

---

## Deviation 7: Completed Analyses Status

### Statement

The novels_only and full_corpus analyses completed successfully before the period stratification error was encountered. These results are:
- novels_only: 58.2% work-weighted balanced accuracy, p=0.0016, 95% CI [32.5%, 70.5%]
- full_corpus: 54.4% work-weighted balanced accuracy, p=0.0001

### Justification

These analyses:
1. Used the exact pre-registered methodology
2. Have sufficient works per author (4-6 per author) for stable bootstrap (failure rate effectively 0%)
3. Were computed before any code modifications related to the period-stratified issue
4. Timestamps are preserved in checkpoint files

---

## Audit Trail

The following materials document this amendment:

1. **Error log**: `results/garnett-analysis.log` - contains original crash traceback
2. **Forensic audit script**: Inline Python code documented in session transcript
3. **Feasibility diagnostic output**: Documented in session transcript showing 30% and 49.4% failure rates
4. **Checkpoint file**: `results/garnett-checkpoint.json` - contains timestamped novels_only and full_corpus results
5. **Git commit history**: All code changes version-controlled

Random seed (42) and computing environment (Python 3.12.3, scikit-learn, joblib) are documented in the repository.

---

## Summary Table (Cumulative with Amendment #1)

| # | Deviation | Pre-Reg | Actual | Rationale | Impact |
|---|-----------|---------|--------|-----------|--------|
| 1 | Permutation count (2.A) | 100,000 implied | 10,000 | Computational constraints | None |
| 2 | A3 in maxT (2.A) | All 6 variants | 5 variants | Different run count | A3 reported separately |
| 3 | TOST/Bayes (2.0) | Not specified | Added | Supplementary evidence | Strengthens conclusions |
| 4 | Garnett corpus (2.D) | 11 works | 19 works | Increased power | Strengthens calibration |
| 5 | Duplicate FW (2.0) | 169 | 169 (code had 171) | Auto-deduplicated | None |
| 6 | Bootstrap feasibility (2.D) | Not specified | ≥95% success required | Statistical validity | Period CIs not estimable |
| 7 | Prior analyses disclosed | N/A | novels_only, full_corpus complete | Transparency | None |

---

## Researcher Statement

This amendment documents a contingency rule introduced after encountering an execution failure, before examining period-stratified inferential results. The feasibility criterion is based on method validity considerations (avoiding biased CIs from selective reporting of successful replicates), not on any knowledge of what the period results would show.

I confirm that:
- No period-stratified accuracies or p-values were computed or examined prior to this amendment
- The diagnostic information examined was limited to execution success/failure rates and work counts
- The novels_only and full_corpus results were already known when drafting this amendment (disclosed above)
- All materials are version-controlled with timestamps

**Signed**: Bryson Westover
**Date**: February 5, 2026

---

## References

- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge University Press.
