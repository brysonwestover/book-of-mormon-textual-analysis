# Voice Segmentation Module Specification

**Module ID:** M01-VOICE
**Version:** 0.2.0 (Revised after GPT review)
**Status:** Pre-registration pending
**Date:** 2026-02-01
**Last Updated:** 2026-02-01 (incorporated GPT-5.2 Pro feedback)

---

## 1. Research Question

**Primary Question:** Does the Book of Mormon text exhibit statistically distinguishable stylistic variation consistent with multiple authorial voices, or does it display the homogeneity expected from single authorship?

**Core Deliverable:** A quantified measure of inter-segment stylistic distance, calibrated against known single-author and multi-author control corpora.

**Secondary Questions:**
- Do stylistic clusters align with claimed narrator boundaries (Nephi, Mormon, Moroni, etc.)?
- Is within-narrator variation lower than between-narrator variation?
- Are there stylistic discontinuities at narrator transition points?

---

## 2. Operational Definitions

### 2.1 "Distinct Voice"

A **distinct voice** is operationally defined as a text segment that:
- Exhibits JSD from other segments **above the 95th percentile of within-author JSD** as calibrated on single-author control corpora
- Maintains this distinctiveness across **≥ 2 independent feature types** (function words, character n-grams)
- Shows **internal consistency** (within-narrator JSD < between-narrator JSD)

**Note:** Absolute JSD thresholds (e.g., 0.15) are NOT pre-specified. They will be derived empirically from control corpora run through the identical pipeline.

### 2.2 "Stylistic Homogeneity"

A text exhibits **stylistic homogeneity** if:
- Mean pairwise JSD between segments falls **within the distribution observed in single-author controls**
- Between-narrator JSD is **not significantly greater** than within-narrator JSD (PERMANOVA p > 0.05)

**Note:** Absolute thresholds will be calibrated, not assumed.

### 2.3 "Claimed Narrator"

The 1830 text explicitly identifies the following narrators:
| Narrator | Primary Sections | Approximate Word Count |
|----------|------------------|------------------------|
| Nephi (1st person) | 1 Nephi, 2 Nephi 1-33 | ~47,000 |
| Jacob | Jacob 1-7, 2 Nephi 6-10 | ~9,000 |
| Enos, Jarom, Omni | Enos, Jarom, Omni | ~3,000 |
| Mormon (narrator) | Mosiah-4 Nephi (abridgment) | ~150,000 |
| Mormon (1st person) | Words of Mormon, Mormon 1-7 | ~8,000 |
| Moroni | Mormon 8-9, Ether, Moroni | ~30,000 |
| Quoted speakers | Various embedded quotes | ~20,000 |

### 2.4 Segment Boundaries

**Primary segmentation:** By claimed narrator (6-8 macro-segments)
**Secondary segmentation:** By 1830 chapter (239 segments)
**Tertiary segmentation:** Sliding windows (1000-word windows, 500-word stride)

---

## 3. Hypothesis Predictions

### H1: Ancient Multiple Authors (Historicity)
- **Prediction:** Distinct stylistic clusters corresponding to claimed narrators
- **Expected JSD:** Mean between-narrator JSD > 0.15; within-narrator JSD < 0.10
- **Pattern:** Nephi ≠ Mormon ≠ Moroni stylistically

### H2: Single 19th-Century Author
- **Prediction:** Uniform stylistic signature across all segments
- **Expected JSD:** Mean pairwise JSD < 0.10 throughout
- **Pattern:** No significant clustering by claimed narrator

### H3: 19th-Century Collaboration
- **Prediction:** 2-3 distinct clusters NOT aligned with claimed narrators
- **Expected JSD:** Clusters at JSD > 0.15, but boundaries don't match narrative claims
- **Pattern:** Stylistic shifts at non-narratively-motivated points

### H4: Ancient Sources via Single Translator
- **Prediction:** Detectable source variation ATTENUATED by translation layer
- **Expected JSD:** Moderate variation (0.08-0.12) with some narrator alignment
- **Pattern:** Signal present but weaker than H1 predicts

### H5: 19th-Century Pseudepigraphy
- **Prediction:** Deliberate style-shifting at narrator boundaries
- **Expected JSD:** Variation present but artificial patterns (e.g., too regular, stereotype-based)
- **Pattern:** Between-narrator > within-narrator, but with anachronistic features

### H0/UX: Indeterminate
- **Condition:** Results don't clearly discriminate between hypotheses
- **Threshold:** Effect sizes too small or confidence intervals overlap

---

## 4. Segmentation Rules

### 4.1 Unit of Analysis

**Primary unit:** 1000-word non-overlapping blocks within narrator sections
**Rationale:** Balances statistical stability with granularity; standard in stylometry

### 4.2 Boundary Identification

Narrator transitions identified by:
1. Explicit statements ("I, Nephi...", "I, Mormon...", "And now I, Moroni...")
2. Book divisions in 1830 text
3. Colophon markers ("Thus ends the record of...")

### 4.3 Exclusions

| Excluded Content | Rationale |
|------------------|-----------|
| Chapter summaries | Uncertain authorship, different register |
| Direct scripture quotes (Isaiah, etc.) | External source, not narrator voice |
| Quoted speech < 100 words | Too short for reliable features |
| Title page, preface, witnesses | Non-narrative paratext |

### 4.4 Minimum Segment Size

- Minimum for analysis: **500 words**
- Segments < 500 words merged with adjacent same-narrator segment
- If still < 500 words, flagged and analyzed separately

### 4.5 Quote Handling

| Quote Type | Handling |
|------------|----------|
| Direct speech (< 100 words) | Include in narrator segment |
| Extended speech (≥ 100 words) | Separate segment, labeled as quoted speaker |
| Scripture quotation | Exclude entirely |

---

## 5. Features

### 5.1 Primary Feature Set: Function Words

**50 most common English function words** (the, and, of, to, a, in, that, it, is, was, for, on, with, as, at, by, this, from, be, or, an, were, which, been, have, but, not, are, they, his, had, her, she, he, we, you, all, would, there, their, will, when, who, more, if, no, out, so, up, into)

**Representation:** Relative frequency per 1000-word segment
**Normalization:** None (raw frequencies preserve signal)

### 5.2 Secondary Feature Sets

| Feature Type | Specification | Dimensionality |
|--------------|---------------|----------------|
| Character 3-grams | Top 200 by corpus frequency | 200 |
| Character 4-grams | Top 200 by corpus frequency | 200 |
| POS bigrams | Using NLTK universal tagset | ~150 |
| Sentence length | Mean, SD, distribution | 3 |
| Vocabulary richness | TTR, hapax ratio, Yule's K | 3 |

### 5.3 Feature Extraction Notes

- All features computed **after** de-hyphenation and normalization
- Punctuation retained for character n-grams
- Case-insensitive for word-level features
- Case-preserved for character n-grams

---

## 6. Method

### 6.1 Overview

```
[Segmented Text] → [Feature Extraction] → [Distance Matrix] → [Clustering] → [Validation]
```

### 6.2 Distance Computation

**Primary metric:** Jensen-Shannon Divergence (JSD)
- Symmetric, bounded [0, 1]
- Information-theoretic interpretation
- Standard in stylometry

**Secondary metric:** Cosine distance (for robustness check)

### 6.3 Clustering

**Primary method:** Hierarchical agglomerative clustering (Ward linkage)
**Secondary method:** K-means with k = number of claimed narrators

**Cluster evaluation:**
- Adjusted Rand Index (ARI) comparing clusters to claimed narrators
- Silhouette score for cluster quality
- Gap statistic to determine optimal k

### 6.4 Statistical Testing

**Primary test: PERMANOVA** (Permutational Multivariate Analysis of Variance)
- Tests whether narrator identity explains significant variance in stylistic distance
- Accounts for multivariate nature of distance matrix
- Implementation: `adonis2()` in R vegan package or Python `skbio.stats.distance.permanova`

**Model specification:**
```
Distance ~ Narrator + Book + Genre + Position + (1|Narrator:Book)
```
- **Narrator:** Primary effect of interest
- **Book:** Control for book-level confounding
- **Genre:** Control for discourse type (narrative, sermon, epistle, prophecy)
- **Position:** Control for temporal/sequential drift

**Permutation strategy:** Blocked permutations
- Permute narrator labels **within book** to respect dependence structure
- 10,000 permutations
- This prevents inflated significance from autocorrelation

**Secondary test:** Simple permutation (for comparison)
- Shuffle narrator labels freely (10,000 times)
- Report alongside PERMANOVA to show effect of blocking

### 6.5 LLM-Assisted Components

**LLM role:** Feature extraction validation only
- Verify POS tagging accuracy on sample
- Identify potential scripture quotations for exclusion
- NOT used for holistic authorship judgments

---

## 7. Silver Label Plan (Phase 1)

### 7.1 Annotator

Single annotator (project maintainer) for Phase 1

### 7.2 Annotation Tasks

| Task | Description | Output |
|------|-------------|--------|
| Narrator boundaries | Mark start/end of each narrator section | JSON with line ranges |
| Quote identification | Mark direct speech ≥ 100 words | JSON with line ranges + speaker |
| Scripture exclusions | Mark Isaiah and other quoted scripture | JSON with line ranges + source |

### 7.3 Annotation Protocol

1. First pass: Mark obvious narrator statements
2. Second pass: Identify extended quotes
3. Third pass: Mark scripture quotations
4. Review: Check boundary consistency

### 7.4 Intra-Annotator Reliability

- Re-annotate 10% of text after 1 week
- Report agreement rate on boundary placement (±5 lines tolerance)

### 7.5 Phase 2 Requirements (Future)

- 3+ independent annotators
- Inter-annotator agreement (Krippendorff's α ≥ 0.8)
- Adjudication protocol for disagreements

---

## 8. Control Corpora

### 8.1 Required Corpora

| Corpus Type | Purpose | Candidate Sources |
|-------------|---------|-------------------|
| **Single-author 19th-c religious** | H2 baseline | Sidney Rigdon sermons, Parley Pratt writings |
| **Multi-author compilation** | H1/H3 calibration | KJV Bible (multiple books), collected sermons |
| **Pseudo-archaic 19th-c** | H5 calibration | Late of Ancient Scripture, other period imitations |
| **Translated multi-author** | H4 calibration | Single-translator Bible versions |

### 8.2 Corpus Instrument Specifications

Each control corpus must document:
- Selection rationale and matching variables
- Known confounds
- Word count and genre comparability
- Copyright/licensing status

### 8.3 Calibration Protocol

1. Apply identical segmentation to control corpora
2. Compute feature distributions
3. Establish JSD baselines:
   - Single-author within-text variation
   - Multi-author between-author variation
   - Translation attenuation factor

---

## 8.5 Confound Analysis

### Known Confounds

| Confound | Issue | Mitigation |
|----------|-------|------------|
| **Book boundaries** | Narrator changes correlate with book changes | Include Book as covariate in PERMANOVA |
| **Genre/discourse type** | Sermons vs narrative vs epistles differ stylistically | Code genre, include as covariate |
| **Sequential position** | Early vs late text may differ (drift) | Include position as covariate |
| **Quoted speech** | Direct quotes have different register | Code quotation density per segment |
| **Topic/content** | Different narrators discuss different topics | Character n-grams may leak content; use function words as primary |

### Variance Partitioning

Before interpreting narrator effect, report:
- % variance explained by Book alone
- % variance explained by Genre alone
- % variance explained by Narrator **after controlling for** Book + Genre
- Partial R² for Narrator effect

### Autocorrelation Handling

- Contiguous 1000-word blocks are not independent
- Use **blocked permutations** (within-book shuffles only)
- Report **effective sample size** adjusted for autocorrelation
- Sensitivity analysis: non-overlapping vs overlapping segments

---

## 9. Negative Controls

### 9.1 Perturbation Tests

| Test | Procedure | Expected Result |
|------|-----------|-----------------|
| **Random shuffle** | Randomize segment-narrator assignment | ARI ≈ 0, no clustering |
| **Synthetic homogenization** | Blend all segments via averaging | JSD → 0, no clustering |
| **Feature ablation** | Remove each feature type | Identify feature contribution |

### 9.2 Leakage Checks

- Verify no narrator-identifying content words in function word list
- Check that proper nouns (Nephi, Mormon, etc.) excluded from features
- Confirm segment boundaries don't leak into features

### 9.3 Synthetic Tests

- Generate synthetic multi-author text with known parameters
- Verify method recovers known authorship structure
- Establish detection sensitivity threshold

---

## 10. Primary Outcomes

### 10.1 Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Mean between-narrator JSD** | Average JSD for segment pairs from different narrators | Report with 95% CI |
| **Mean within-narrator JSD** | Average JSD for segment pairs from same narrator | Report with 95% CI |
| **JSD ratio** | Between/within ratio | > 1.5 suggests distinct voices |
| **ARI** | Clustering alignment with claimed narrators | > 0.3 = moderate; > 0.5 = strong |
| **Permutation p-value** | Significance of narrator effect | α = 0.05 |

### 10.2 Pre-Registered Decision Rules

**Note:** Absolute JSD thresholds are NOT pre-registered. Instead, we pre-register the **decision procedure**:

| Finding | Criteria |
|---------|----------|
| **Strong voice distinction** | PERMANOVA narrator effect p < 0.01 AND JSD ratio exceeds multi-author control baseline |
| **Moderate voice distinction** | PERMANOVA narrator effect p < 0.05 AND JSD ratio exceeds single-author control baseline |
| **Weak/no distinction** | PERMANOVA narrator effect p > 0.05 OR JSD ratio within single-author control range |

**Calibration requirement:** Before applying to Book of Mormon, establish:
1. Single-author control: mean and 95th percentile of within-text JSD
2. Multi-author control: mean between-author JSD and ARI with known labels
3. Single-translator control: JSD patterns for translated multi-source texts

### 10.3 Reporting

All metrics reported with:
- Point estimate
- 95% bootstrap confidence interval (10,000 resamples)
- Effect size interpretation
- Comparison to control corpora baselines

---

## 11. Uncertainty Handling

### 11.1 Bootstrap Confidence Intervals

- Resample segments with replacement (10,000 iterations)
- Compute metric distribution
- Report 2.5th and 97.5th percentiles

### 11.2 Multiple Comparisons

- Report both uncorrected and FDR-corrected p-values
- Use Benjamini-Hochberg procedure for pairwise comparisons

### 11.3 Sensitivity Analyses

| Analysis | Purpose |
|----------|---------|
| Vary segment size (500, 1000, 2000 words) | Test granularity sensitivity |
| Vary feature set | Test feature robustness |
| Include/exclude chapter summaries | Test paratext sensitivity |
| Different distance metrics | Test metric robustness |

### 11.4 Pre-Registered Features

Lock the following before analysis:
- Function word list
- Character n-gram vocabulary
- Segment boundaries
- All threshold values

---

## 12. Translation Layer Considerations

### 12.1 What Survives Translation

| Feature Type | Translation Survival | Rationale |
|--------------|---------------------|-----------|
| Function words | Low-Medium | Translator's habits dominate |
| Syntax patterns | Low | Heavily shaped by target language |
| Discourse markers | Medium | Some source influence possible |
| Vocabulary richness | Medium | May partially reflect source |
| Sentence rhythm | Low-Medium | Mixed influence |

### 12.2 Implications

- If H4 (single translator), expect **attenuated** but non-zero signal
- Function words may primarily reflect translator, not source authors
- Combine multiple feature types to disentangle layers

### 12.3 Calibration Approach

- Analyze single-translator multi-source texts (e.g., Bible translations)
- Quantify expected attenuation factor
- Adjust interpretation accordingly

---

## 13. Falsification Criteria

### Per Hypothesis

| Hypothesis | Falsified If |
|------------|--------------|
| **H1** (Ancient multiple) | JSD ratio < 1.2 AND clusters don't match narrators (ARI < 0.2) |
| **H2** (Single author) | JSD ratio > 2.0 AND clusters match narrators (ARI > 0.5) |
| **H3** (19th-c collab) | Only 1 cluster OR clusters perfectly match ancient narrator claims |
| **H4** (Single translator) | Variation exceeds single-translator Bible baseline by > 2 SD |
| **H5** (Pseudepigraphy) | No style shifting at narrator boundaries OR authentic archaic patterns |

### Module-Level Falsification

This module's approach is falsified if:
- Control corpora don't separate as expected (method invalid)
- Features show no discriminative power on known multi-author texts
- Results vary wildly across sensitivity analyses (unstable)

---

## 14. Limitations

### 14.1 Module-Specific Limitations

1. **OCR errors:** ~1% token error rate may introduce noise
2. **Segmentation subjectivity:** Boundary placement involves interpretation
3. **Feature selection:** Function word list is a methodological choice
4. **Translation confound:** Cannot fully separate source from translator

### 14.2 Interpretive Limitations

1. **Voice ≠ Author:** Distinct voices could be deliberate stylistic choices
2. **Homogeneity ≠ Single author:** Skilled imitator could maintain consistency
3. **Calibration dependency:** Results meaningful only relative to controls

### 14.3 Scope Limitations

1. Does not address content/theological claims
2. Does not establish historical identity of authors
3. Exploratory (Phase 1) - not confirmatory

---

## 15. Update Logic

### How Results Update H1-H5

| Result Pattern | Most Supported | Least Supported |
|----------------|----------------|-----------------|
| Strong narrator-aligned clusters | H1, H5 | H2 |
| No clustering | H2, H4 | H1, H3 |
| Non-narrator clusters | H3 | H1 |
| Moderate narrator-aligned | H4, H1 | H2 |
| Artificial-seeming patterns | H5 | H1 |

### Posterior Adjustment

Results feed into Bayesian-style reasoning:
- Strong evidence: Shift posterior by ~30%
- Moderate evidence: Shift posterior by ~15%
- Weak evidence: Shift posterior by ~5%
- Null result: Minimal shift, increase uncertainty

### Integration with Other Modules

Voice Segmentation provides:
- Segment definitions for other modules
- Baseline stylistic variation estimates
- Candidate author groupings to test

---

## Appendices

### A. Function Word List (50 words)

```
the, and, of, to, a, in, that, it, is, was,
for, on, with, as, at, by, this, from, be, or,
an, were, which, been, have, but, not, are, they, his,
had, her, she, he, we, you, all, would, there, their,
will, when, who, more, if, no, out, so, up, into
```

### B. File Outputs

| File | Description |
|------|-------------|
| `segments.json` | Segment definitions with boundaries |
| `features.csv` | Feature matrix (segments × features) |
| `distances.csv` | Pairwise JSD matrix |
| `clusters.json` | Cluster assignments and metrics |
| `results.json` | All metrics and statistics |
| `report.md` | Human-readable analysis report |

### C. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-01 | Initial draft for review |
| 0.2.0 | 2026-02-01 | Incorporated GPT-5.2 Pro feedback: calibrated thresholds, PERMANOVA, confound analysis, blocked permutations |

---

## GPT-5.2 Pro Review Summary (2026-02-01)

**Key changes made based on review:**
1. Replaced fixed JSD thresholds with empirically-derived thresholds from controls
2. Added PERMANOVA as primary statistical test with covariates for confounds
3. Added blocked permutation strategy to handle autocorrelation
4. Added explicit confound analysis section (book, genre, position, quotation)
5. Added variance partitioning requirement

**Remaining limitations acknowledged:**
- H4 vs H2 discrimination remains weak without translator benchmarks
- Method detects heterogeneity but cannot identify mechanism

---

## Sign-Off

**Status:** Revised based on GPT-5.2 Pro review. Ready for implementation planning.
