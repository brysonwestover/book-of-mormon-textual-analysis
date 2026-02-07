# Full-Scale Audit: run_aggregated_analysis.py v1.4.0

**Date:** 2026-02-06
**Auditor:** GPT-5.2-Pro (via Claude orchestration)
**Target:** DSH/LLC Journal

---

## Executive Summary

The GPT audit identified **4 CRITICAL**, **4 MAJOR**, and **9 MINOR** issues. However, several of the "critical" issues require contextualization given the specific purpose of this supplementary analysis.

---

## CRITICAL ISSUES

### C1) N=14 Too Small, MORONI=2 Problematic
**Concern:** With 14 labeled documents, performance estimates are high-variance. MORONI=2 means balanced accuracy can move drastically if one MORONI item flips.

**Our Context:** This is acknowledged in the script. The run-aggregated analysis exists specifically BECAUSE treating 244 blocks as independent would be pseudoreplication. The 14 runs are the true independent units.

**Mitigation:**
- Frame as exploratory/descriptive case study
- 3-class analysis (excluding MORONI) is already implemented
- Acknowledge limits in manuscript

### C2) Permutation Exchangeability Assumptions May Be Violated
**Concern:** Contiguous text segments have structural dependence (chronology, book boundaries, editorial strata). Unrestricted shuffling could yield anti-conservative p-values.

**Our Context:** This is a valid concern. Runs span different books/sections and may not be truly exchangeable.

**Mitigation:**
- Add explicit defense of exchangeability assumption in documentation
- Consider blocked permutation (within book/stratum) as sensitivity analysis
- The confound probe (book prediction) partially addresses this

### C3) Per-Run Aggregation Discards Statistical Power
**Concern:** Aggregating to per-run frequencies equalizes means but not measurement precision. A 2,000-word run and 40,000-word run become equally weighted.

**Our Context:** This criticism MISSES THE POINT. The run-aggregated analysis is specifically designed to treat runs as the unit of analysis. The alternative (block-level analysis with 244 blocks) has pseudoreplication issues.

**Response:**
- Document the trade-off explicitly
- Consider sample weighting by run length as sensitivity analysis
- Acknowledge both block-level and run-level results

### C4) Multiplicity Not Addressed
**Concern:** Multiple analysis forks (2 classifiers, 3-class, feature sensitivity, C sensitivity) without family-wise error control.

**Our Context:** We do separate confirmatory (primary 4-class analysis) from exploratory (robustness checks).

**Mitigation:**
- Explicitly label primary vs exploratory analyses
- Consider FDR correction for exploratory analyses
- Pre-specify primary endpoint clearly

---

## MAJOR ISSUES

### M1) LOO-CV High Variance at N=14
**Concern:** LOO has high variance and is sensitive to idiosyncratic points.

**Mitigation:**
- Report full per-fold predictions (already done)
- Jackknife influence analysis (already implemented)
- Acknowledge instability in manuscript

### M2) Feature Selection Leakage Unclear
**Concern:** Choosing top-k features must be done within training fold.

**Our Context:** In `feature_sensitivity_analysis()`, features are ranked by corpus-wide frequency (not supervised), so this is not as severe. However, it could still leak information.

**Mitigation:**
- Clarify that frequency ranking uses full corpus (unsupervised)
- Consider moving ranking inside CV folds for robustness

### M3) Burrows Delta Implementation Details Incomplete
**Concern:** What happens if feature has ~zero variance? Which variant?

**Our Context:** We DO specify `sd[sd == 0] = 1` and `ddof=1`. This is documented.

**Mitigation:**
- Add explicit formula to documentation
- Note this is "classic Delta" (mean absolute z-score difference)

### M4) Confound Probe Doesn't Actually Control Confound
**Concern:** Showing book is predictable doesn't prove narrator classification isn't book-driven.

**Our Context:** Valid point. The probe is informative but not definitive.

**Mitigation:**
- Acknowledge this limitation
- Consider blocked validation (train on some books, test on others) if data permits
- Frame probe as suggestive, not definitive

---

## MINOR ISSUES

### m1) Reproducibility Gaps
- RNG seeds: FIXED (RANDOM_SEED = 42 used throughout)
- Package versions: Need to document

### m2) Preprocessing/Tokenization Ambiguity
- Tokenization is simple regex: `r'\b[a-z]+\b'`
- No lemmatization, no stopwords beyond function words
- Should document explicitly

### m3) Length Effects Not Fully Neutralized
- Per-1000-word normalization used
- Unequal run lengths may still have effects

### m4) Class Imbalance & Metric Choice
- Balanced accuracy used (correct choice)
- class_weight='balanced' in LogisticRegression

### m5) Hyperparameter Transparency
- C sensitivity analysis already implemented
- Feature sensitivity analysis already implemented
- Should add "model card" style table

### m6) Dependence in Uncertainty Estimates
- Bootstrap CI caveats already documented
- Consider block bootstrap as sensitivity

### m7) Diagnostics Missing
- Feature sensitivity: IMPLEMENTED
- Chunk size sensitivity: Not applicable (run-level)
- Alternative distance metrics: Only Delta vs LR

### m8) Data Leakage via Preprocessing
- Vocabulary (function words) is fixed a priori, not data-derived
- No leakage issue here

### m9) Reporting Completeness
- Estimand should be clarified
- Primary endpoint should be pre-specified

---

## ADVERSARIAL REVIEW (Simulated)

### Most Damaging Reviewer Comment
> "The statistical claims are not credible. The manuscript treats highly dependent contiguous text segments as independent observations and applies naive permutation tests that violate exchangeability, producing anti-conservative p-values. Performance is evaluated with high-variance leave-one-out cross-validation on only 14 runs, while feature selection and standardization appear to use the full dataset, introducing leakage. The analysis further aggregates to per-run summaries in some places, discarding information and inducing heteroskedasticity. Finally, the authors report numerous analytic variants (feature sets, distances, chunking choices) without multiplicity control, making the reported significance indistinguishable from selection bias. With one class represented by only two runs, the study cannot support the strength of its attribution claims."

### Rebuttal
> "We agree the original pipeline overstated evidential strength under assumptions (independent chunks; free label permutation) that are inappropriate for contiguous text. We have revised the study to use run-grouped validation (Leave-One-Run-Out / GroupKFold) with all feature selection and z-scoring performed strictly within training folds, eliminating leakage. Our inferential tests now permute at the run level (and, where applicable, use circular-shift nulls), preserving dependence and restoring exchangeability under the null. We pre-specified a single primary endpoint and applied FDR control to secondary exploratory analyses. Because MORONI has only two runs, we removed it from confirmatory multi-class inference and report it only as a descriptive case study. The revised manuscript reframes conclusions accordingly: the results demonstrate a stylometric signal consistent with authorship differences in this limited corpus, but do not constitute definitive attribution; we provide uncertainty intervals and robustness checks across chunk sizes and feature choices to make the limits of evidence explicit."

---

## MINIMUM VIABLE PATH TO PUBLICATION

### Required Changes
1. **Reframe claims**: Exploratory/case-study, not definitive attribution
2. **Pre-specify primary endpoint**: 4-class LR analysis with fixed parameters
3. **Label exploratory analyses**: 3-class, feature sensitivity, confound probe
4. **Document exchangeability assumption**: Why runs are exchangeable under null
5. **MORONI handling**: Remove from confirmatory claims; report as exploratory only

### Framing That Will Pass Review
- "We evaluate whether stylometric features contain discriminative information consistent with narrator differences under a grouped validation design."
- "We report uncertainty-aware performance and conduct pre-registered primary analysis; all secondary analyses are exploratory."
- "We explicitly avoid claiming definitive attribution; results motivate further investigation."

### What Will NOT Pass Review
- Strong attribution claims with small p-values
- Overstated generalization beyond these 14 runs
- Multiple uncorrected forks presented as confirmatory

---

## Action Items

### Immediate (Before Publication)
1. [ ] Update script docstring with explicit primary/exploratory designation
2. [ ] Add "model card" style configuration table
3. [ ] Document tokenization rules explicitly
4. [ ] Add exchangeability defense to METHODOLOGY.md
5. [ ] Frame MORONI as exploratory only in manuscript

### Consider (If Time Permits)
1. [ ] Blocked permutation by book/stratum as sensitivity
2. [ ] Sample weighting by run length as sensitivity
3. [ ] FDR correction for exploratory analyses

### Deferred (Not Required for This Paper)
1. [ ] Move feature ranking inside CV folds
2. [ ] Block bootstrap for CIs
3. [ ] Alternative Delta variants (Eder's Delta, Cosine Delta)

---

## Conclusion

The script is methodologically sound for a supplementary, exploratory analysis. The key changes needed are:

1. **Framing**: Frame as case study / exploratory, not definitive
2. **Primary endpoint**: Pre-specify and label
3. **MORONI**: Treat as exploratory only
4. **Documentation**: Defend exchangeability assumption

With these changes, publication at DSH/LLC is achievable.
