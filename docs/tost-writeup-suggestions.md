# TOST Analysis Write-Up Suggestions

*Generated from GPT-5.2-Pro consultation, February 5, 2026*

---

## Core Results (Primary Analysis)

> Using function-word features on 1000-word blocks, 4-class narrator classification achieved 25.5% accuracy, essentially at chance (25%). A 100,000-iteration permutation test yielded p = 0.247, indicating that the observed accuracy is not unusually high under the null of no narrator-specific signal.

---

## Equivalence Framing (TOST)

> We additionally conducted equivalence testing (TOST) with a pre-registered practical equivalence region of 25% ± 15% (10%–40%). The 90% confidence interval for accuracy was [10.1%, 40.9%], narrowly exceeding the upper equivalence bound, and thus did not meet the criterion for formal equivalence at α = 0.05 (TOST p = 0.060). This result is consistent with chance-level performance but remains too imprecise to rule out effects modestly larger than the equivalence margin.

---

## Bayesian Support

> Bayes factors modestly favored the null model (BF01 = 2.85), indicating weak evidence for no discriminative signal relative to an alternative.

---

## Robustness Checks / Variants

> Sensitivity analyses varying block size (500 vs. 2000 words) and including quotations produced accuracies between 29.9% and 32.4%, with similarly wide 90% confidence intervals overlapping chance. None of these specifications provided evidence of reliable above-chance classification, nor did they meet the predefined equivalence criterion; Bayes factors in all variants weakly favored the null (BF01 ≈ 2.1–2.5).

---

## Interpretation (Careful Substantive Conclusion)

> Overall, across multiple reasonable preprocessing choices, we find no positive evidence that the four narrator labels are associated with a stable, machine-detectable function-word stylometric signature in this dataset. However, due to limited effective sample size in cross-validation (14–15 runs) and correspondingly wide uncertainty intervals, these analyses do not tightly bound the magnitude of any potential narrator-specific effect.

---

## Optional: Acknowledging the Margin Choice

> Notably, the equivalence margin (±15% absolute accuracy) is conservative in the sense that it only targets the exclusion of relatively large deviations from chance; smaller effects cannot be excluded given current uncertainty.

---

## Key Interpretive Points

### What "no positive evidence" means
- The permutation test (p = 0.247) shows the observed accuracy is not unusual under the null
- The TOST analysis shows we're *near* equivalence but can't formally claim it
- The Bayes factors (~2-3) provide weak evidence *for* the null hypothesis

### What we can and cannot claim
**Can claim**: No evidence of narrator-distinguishing stylometric signal was detected
**Cannot claim**: Narrators are definitively indistinguishable (uncertainty too wide)

### The pattern across variants
- No monotone improvement with block size (500→1000→2000: 32.4%→25.5%→29.9%)
- Including quotations doesn't help (30.1%)
- This looks like sampling noise, not a real signal

---

## Statistical Notes for Methods Section

### TOST Methodology
- Equivalence bounds: 25% ± 15% (i.e., 10% to 40%)
- Justification: With n=14 runs, narrower bounds lack power; 40% still represents meaningful above-chance performance
- 90% CI interpretation: If entirely within bounds, equivalence is demonstrated at α=0.05

### Bayes Factor Calculation
- Approximate JZS-style Bayes factor for one-sample t-test
- Prior scale: 0.1 (default Cauchy width)
- BF01 > 1 favors null; BF01 > 3 = moderate evidence; BF01 > 10 = strong evidence

### Limitations
1. Low effective sample size (n=14-15 CV runs)
2. Wide equivalence margin only excludes large effects
3. CV fold overlap means SEs are approximate
