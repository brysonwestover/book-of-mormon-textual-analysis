# Expected Stylometric Results If Ancient Multi-Author Claim Were True

**Date:** 2026-02-01
**Status:** Reference document
**Consulted:** GPT-5.2 Pro

---

## Summary

This document describes what stylometric results we would EXPECT to see if the Book of Mormon were genuinely an ancient text with multiple distinct authors, versus what our null result suggests.

---

## What We Actually Found

| Metric | Result |
|--------|--------|
| Balanced Accuracy | 21.6% (BELOW 25% chance) |
| Permutation p-value | 1.0 (not significant) |
| Jacob recall | 0% |
| Moroni recall | 0% |
| Signal source | Topic/genre (disappeared when content words removed) |

---

## What We Would EXPECT If Ancient Multi-Authorship Were True

### Two Versions of the Claim

**Version A: Voices survive translation (strong, testable)**

If multiple ancient authors existed AND their stylistic habits survived into the English output, we should expect:

| Expected Pattern | Our Result |
|-----------------|------------|
| Narrator-consistent, replicable separation on function words | ❌ No separation |
| High out-of-sample classification accuracy | ❌ Below chance |
| Robustness to preprocessing choices | ❌ Not tested yet |
| Persistence after controlling for genre/topic | ❌ Signal disappeared |
| Coherence across segment sizes | ❌ Not tested yet |
| Non-zero recall for each narrator | ❌ Jacob=0%, Moroni=0% |

**Version B: Voices homogenized by translation (weaker claim)**

If multiple ancient authors existed but the English reflects a single strong translator/editor:

| Expected Pattern | Our Result |
|-----------------|------------|
| Little to no narrator-specific function-word signal | ✅ Matches |
| Differences are genre/topic-driven, not idiolects | ✅ Matches |
| Detection only possible with non-function-word features | ? Not tested |

---

## The Translation Layer Problem

### Key Insight from GPT

> "Function words are excellent for authorship when the surface language is the authors' own. They are much less diagnostic when there is a strong translation/register imposition or editorial smoothing."

### What Translation Studies Show

1. **Translators leave detectable fingerprints** (Baker 2000; Koppel & Ordan 2011)
2. **Translated texts can cluster by translator rather than source author**
3. **Function words are particularly vulnerable** to translator normalization
4. **Biblical/KJV register further reduces degrees of freedom** (formulaic phrasing)

### Implication

A single translator (Joseph Smith) producing a sustained pseudo-biblical register would be expected to homogenize function-word distributions across all "narrators."

---

## What Hypotheses Are MORE Consistent With Our Null Result?

1. **Single English production voice dominates**
   - One author/translator's function-word habits overwhelm any source variation

2. **Translation/normalization erases source-author signals**
   - Biblical English register compresses variation in articles, prepositions, auxiliaries

3. **Narrator partitions are not stylometrically natural units**
   - Too short, interleaved with quotes, defined by framing rather than continuous composition

4. **Topic/genre effects masquerade as "voice"**
   - War narrative vs sermon vs epistle vs prophecy

5. **Strong formulaicity flattens idiolect**
   - Repeated phrases, stock transitions, scriptural pastiche

---

## What Hypotheses Are LESS Consistent With Our Null Result?

1. **Distinct narrator-specific English idiolects are present and strong**
   - If robust fingerprints existed, standard methods should recover them

2. **"Narrator" corresponds closely to independent authorial generators**
   - Null result pushes against simple mapping

3. **Multiple authors with no harmonizing layer**
   - Persistent lack of function-word separation is hard to explain

---

## Falsifiability Analysis

### What Would Have SUPPORTED Ancient Multi-Authorship?

1. Stable narrator clustering across methods (Delta, SVM, etc.)
2. High cross-validated accuracy predicting held-out chapters
3. Separation remains when matching on genre (sermon vs sermon)
4. Independence from topic confounds
5. External calibration showing method works on comparable texts

### What Would Have FALSIFIED It?

Strictly, stylometry on English translation cannot cleanly falsify "ancient multi-authorship" because the hypothesis can retreat to "multiple ancient authors, but translator homogenized style."

However, these would strongly DISCONFIRM "separable narrator idiolects":
1. Narrator labels unlearnable while text is self-consistent as one "author"
2. Alignment with specific modern authorial profile
3. Separation tracks dictation/production conditions rather than narrators

### Is This a Fair Test?

**Fair test of:** "If there were multiple authors, English should retain narrator-specific function-word habits"

**Potentially unfair test of:** "There were multiple ancient authors" (broader claim)

Because translation and editorial harmonization can erase the very cues the method relies on.

---

## Diagnostic Value of Our Null Result

Per GPT:

> "A null result is evidence, but not a clean yes/no test for underlying authorship because:
> - Function words are excellent when surface language is authors' own
> - They are much less diagnostic with translation/register imposition
> - Null ≠ falsification
> - But repeated, robust nulls still update expectations"

**Bottom line:** Our result tells us the English layer does not behave like a clean multi-author corpus partitioned by narrator for function-word stylometry. It does NOT uniquely identify why.

---

## Additional Tests Needed

To strengthen conclusions, we should run:

1. **Unmasking analysis** - Does separation collapse quickly (shallow) or gradually (deep)?
2. **Pairwise comparisons** - Mormon vs Nephi only (larger samples)
3. **Control corpora** - Does method work on known multi-author texts (KJV)?
4. **Same-author verification** - Are cross-narrator pairs as different as expected?
5. **Genre-controlled analysis** - Compare sermon vs sermon, narrative vs narrative

---

## References

- Mosteller & Wallace (1964). Inference and Disputed Authorship: The Federalist
- Burrows, J. (1987, 2002). Delta method for authorship attribution
- Stamatatos, E. (2009). Survey of modern authorship attribution methods
- Koppel, M. et al. (2009). Computational methods in authorship attribution
- Koppel, M. & Ordan, N. (2011). Translationese and its dialects
- Baker, M. (2000). Towards a methodology for investigating the style of a literary translator
