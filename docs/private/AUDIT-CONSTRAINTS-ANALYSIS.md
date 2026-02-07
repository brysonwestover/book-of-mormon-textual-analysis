# Comprehensive Audit: Constraints Analysis

**Created:** 2026-02-06
**Purpose:** Private analysis of methodological constraints identified in three-reviewer audit
**Status:** WORKING DOCUMENT - Not for GitHub

---

## Overview

Three expert perspectives audited the run-aggregated analysis script:
1. **Statistician** - Statistical validity, assumptions, inference
2. **Computational Stylometrist** - Field standards, best practices
3. **Adversarial Reviewer** - Attack vectors, fatal flaws

This document analyzes each identified constraint to determine:
- Is it truly fundamental (cannot be addressed)?
- Can it be mitigated (partially addressed)?
- Is it already addressed by existing work?

---

## Constraint Classification Framework

| Category | Definition | Action |
|----------|------------|--------|
| **FUNDAMENTAL** | Inherent to data/question, cannot be changed | Document clearly, bound claims |
| **MITIGABLE** | Can be partially addressed with additional analysis | Implement mitigation |
| **ADDRESSABLE** | Can be fully resolved with code/methodology changes | Fix it |
| **ALREADY ADDRESSED** | Existing work handles this | Reference it |

---

## Detailed Constraint Analysis

### CONSTRAINT 1: Small Sample Size (N=14 runs)

**Source:** All three reviewers flagged this as critical

**The Issue:**
- Only 14 independent runs exist in the data
- This limits statistical power to detect effects
- Creates high variance in all estimates
- Makes p-values less informative

**Is it Fundamental?**

| Aspect | Assessment |
|--------|------------|
| Can we get more runs? | NO - The Book of Mormon has a fixed structure. The 14 runs represent all contiguous same-narrator passages of sufficient length. |
| Can we redefine "run"? | PARTIALLY - Could use smaller runs, but this increases dependence issues |
| Can we use blocks instead? | YES, but creates pseudoreplication (the original problem) |

**Potential Mitigations:**
1. Use block-level analysis WITH proper clustering (mixed-effects models)
2. Use Bayesian inference with informative priors (borrow strength)
3. Frame as "constrained inference" with explicit power statements
4. Compare to similarly-constrained studies in the literature

**Current Status:** Partially mitigated
- Block-level analysis exists (N=244) with run-level permutation
- Run-aggregated analysis is the conservative version
- Power/sensitivity statement exists in REPRODUCIBILITY.md

**Verdict:** FUNDAMENTAL but partially MITIGABLE

---

### CONSTRAINT 2: High Dimensionality (p >> n: 169 features, 14 samples)

**Source:** Statistician, Adversarial reviewer

**The Issue:**
- More features than samples = underdetermined system
- Model depends heavily on regularization
- Coefficients are unstable
- Risk of overfitting even with CV

**Is it Fundamental?**

| Aspect | Assessment |
|--------|------------|
| Can we reduce features? | YES - Use fewer function words |
| Can we use dimensionality reduction? | YES - PCA, feature selection |
| Is 169 FW standard in field? | YES - But robustness across sizes expected |

**Potential Mitigations:**
1. **Feature reduction:** Test with 50, 100, 150 FW subsets
2. **PCA:** Reduce to top k components before classification
3. **Stronger regularization:** Use smaller C values
4. **Different models:** Use models designed for p >> n (e.g., penalized LDA)

**Current Status:** Partially mitigated
- C sensitivity analysis tests regularization strength
- L2 regularization is applied
- NOT mitigated: No feature reduction sensitivity

**Verdict:** MITIGABLE - Should add feature set size sensitivity

---

### CONSTRAINT 3: Contiguity / Temporal Dependence

**Source:** Adversarial reviewer, Stylometrist

**The Issue:**
- Runs are contiguous text segments
- Adjacent blocks share topic, register, narrative context
- Violates independence assumption
- May conflate narrator with section/topic

**Is it Fundamental?**

| Aspect | Assessment |
|--------|------------|
| Can we randomize block order? | NO - Text structure is fixed |
| Can we control for topic? | PARTIALLY - Could add topic covariates |
| Can we use non-contiguous sampling? | NO - Would destroy the narrator signal we're testing |

**Potential Mitigations:**
1. **Run-level aggregation:** Treat each run as single observation (DONE)
2. **Blocked permutation:** Permute at run level, not block level (DONE)
3. **Genre/register control:** Stratify by discourse type
4. **Topic modeling:** Add topic as covariate or confounder

**Current Status:** ADDRESSED
- Run-aggregated analysis treats runs as unit
- Permutation is at run level (count-preserving shuffle)
- Quotations excluded

**Verdict:** ADDRESSED by current methodology

---

### CONSTRAINT 4: Class Imbalance (MORONI = 2 runs)

**Source:** All three reviewers

**The Issue:**
- MORMON: 4 runs, NEPHI: 5 runs, MORONI: 2 runs, JACOB: 3 runs
- MORONI with only 2 runs means:
  - In LOO, training has only 1 MORONI example
  - Essentially "few-shot" learning for MORONI
  - Per-class metrics for MORONI are unreliable

**Is it Fundamental?**

| Aspect | Assessment |
|--------|------------|
| Can we get more MORONI text? | NO - The Book of Mormon has fixed MORONI content |
| Can we exclude MORONI? | YES - But loses information |
| Can we merge classes? | PARTIALLY - Could merge MORONI+MORMON (both "editorial") |

**Potential Mitigations:**
1. **Exclude MORONI:** Run 3-class analysis (MORMON, NEPHI, JACOB)
2. **Merge classes:** Combine into 2 classes (e.g., "editorial" vs "autobiographical")
3. **Report separately:** Flag MORONI results as exploratory
4. **Use hierarchical model:** Borrow strength across classes

**Current Status:** Partially documented
- Limitations mention MORONI=2
- Per-class metrics are computed but not prominently displayed

**Verdict:** FUNDAMENTAL but MITIGABLE via class restructuring or explicit exclusion

---

### CONSTRAINT 5: Narrator ≠ Author (Construct Validity)

**Source:** Adversarial reviewer, Stylometrist

**The Issue:**
- "Narrator" is a textual label, not verified authorship
- Single author CAN produce distinct narrator voices
- Multiple authors CAN be homogenized by editing/translation
- The mapping from "stylistic differentiation" to "number of authors" is not 1:1

**Is it Fundamental?**

| Aspect | Assessment |
|--------|------------|
| Can we verify authorship externally? | NO - That's the research question |
| Can we test voice consistency? | YES - Compare to known single/multi-author texts |
| Can we avoid overclaiming? | YES - Use careful language |

**Potential Mitigations:**
1. **Calibration studies:** Compare to known single-author multi-voice texts (DONE - Garnett)
2. **Language precision:** Say "stylistic differentiation" not "authorship"
3. **Multi-hypothesis framework:** Present H1-H5 explicitly (DONE)
4. **Explicit underdetermination statement:** (DONE in HYPOTHESIS-PREDICTIONS.md)

**Current Status:** ADDRESSED
- Garnett calibration shows method CAN detect authorial signal through translator
- Multi-hypothesis framework avoids binary claims
- Documentation explicitly states what can/cannot be concluded

**Verdict:** ADDRESSED by existing framing and calibration

---

### CONSTRAINT 6: Translation Layer Effects

**Source:** Stylometrist, Adversarial reviewer

**The Issue:**
- Book of Mormon is a claimed translation
- Function words may reflect translator, not source authors
- Translation could homogenize or introduce stylistic variation
- This creates ambiguity in interpretation

**Is it Fundamental?**

| Aspect | Assessment |
|--------|------------|
| Can we remove translation effects? | NO - We only have the translated text |
| Can we model translation effects? | PARTIALLY - Via calibration corpora |
| Can we compare to other translations? | YES - Garnett study does this |

**Potential Mitigations:**
1. **Garnett calibration:** Shows authorial signal CAN survive translation (DONE)
2. **KJV comparison:** Compare to KJV (single translator, multiple source authors)
3. **Explicit acknowledgment:** Document translation as confounder (DONE)

**Current Status:** ADDRESSED
- Garnett calibration (58% accuracy through Constance Garnett's translations)
- Explicit discussion in methodology

**Verdict:** ADDRESSED by Garnett calibration

---

### CONSTRAINT 7: Absence of Evidence ≠ Evidence of Absence

**Source:** Adversarial reviewer (FATAL FLAW designation)

**The Issue:**
- p = 0.177 means "not significant," not "no effect"
- Without power analysis, null result is uninformative
- Without equivalence test, can't claim "no difference"

**Is it Fundamental?**

| Aspect | Assessment |
|--------|------------|
| Can we do power analysis? | YES |
| Can we do equivalence testing? | YES - TOST |
| Can we report effect sizes with CI? | YES |

**Potential Mitigations:**
1. **TOST equivalence test:** (DONE in primary analysis, p=0.06)
2. **Bayes Factor:** (DONE, BF01=2.85 weak evidence for null)
3. **Power/sensitivity statement:** (DONE in REPRODUCIBILITY.md)
4. **Careful language:** "No detectable effect" not "no effect"

**Current Status:** ADDRESSED
- TOST p=0.06 (near-equivalence)
- Bayes Factor BF01=2.85 (weak evidence for null)
- Sensitivity statement in documentation

**Verdict:** ADDRESSED by existing analyses

---

### CONSTRAINT 8: Missing Field-Standard Baselines (Delta)

**Source:** Stylometrist

**The Issue:**
- Burrows' Delta is the canonical stylometry baseline
- Logistic regression alone looks "ML-ish" not "stylometry-grounded"
- Field reviewers expect Delta-family comparisons

**Is it Fundamental?**

| Aspect | Assessment |
|--------|------------|
| Can we add Delta? | YES - Straightforward to implement |
| Is it required? | EXPECTED by stylometry reviewers |

**Potential Mitigations:**
1. **Add Burrows' Delta:** Compute z-scored MFW, use Delta distance
2. **Add Cosine Delta:** More robust variant
3. **Nearest centroid classifier:** Delta-native classification

**Current Status:** NOT ADDRESSED
- No Delta baseline in current scripts
- Only logistic regression tested

**Verdict:** ADDRESSABLE - Should add Delta baseline

---

### CONSTRAINT 9: Missing Visualization

**Source:** Stylometrist

**The Issue:**
- No PCA/MDS plots showing cluster structure
- No dendrograms showing hierarchical relationships
- Field expects unsupervised exploration

**Is it Fundamental?**

| Aspect | Assessment |
|--------|------------|
| Can we add visualization? | YES |
| Is it required? | EXPECTED by stylometry reviewers |

**Potential Mitigations:**
1. **PCA plot:** 2D projection colored by narrator
2. **MDS/t-SNE:** Alternative dimensionality reduction
3. **Dendrogram:** Hierarchical clustering of runs
4. **Bootstrap consensus tree:** Show clustering stability

**Current Status:** NOT ADDRESSED
- No visualization in run-aggregated script

**Verdict:** ADDRESSABLE - Should add visualization

---

### CONSTRAINT 10: Wilson CI Inappropriate for CV

**Source:** Statistician

**The Issue:**
- Wilson interval assumes iid Bernoulli trials
- LOO predictions are NOT independent (overlapping training sets)
- CI may be misleadingly narrow or wrong

**Is it Fundamental?**

| Aspect | Assessment |
|--------|------------|
| Can we use different CI? | YES |
| What's appropriate? | Permutation-based CI, bootstrap with blocking |

**Potential Mitigations:**
1. **Permutation-based CI:** Use null distribution percentiles
2. **Document limitation:** Note Wilson is approximate
3. **Remove Wilson:** Rely only on permutation distribution

**Current Status:** PARTIALLY ADDRESSED
- Docstring notes Wilson is for raw accuracy only
- Bootstrap CI included (with caveats)
- Permutation null distribution reported

**Verdict:** MITIGABLE - Should document more clearly or replace

---

### CONSTRAINT 11: Feature List Justification

**Source:** Adversarial reviewer, Stylometrist

**The Issue:**
- Why 169 function words?
- Is the list pre-specified or data-driven?
- Arbitrary choices can affect results

**Is it Fundamental?**

| Aspect | Assessment |
|--------|------------|
| Can we justify the list? | YES - Document source |
| Can we test sensitivity? | YES - Vary list size |

**Potential Mitigations:**
1. **Document source:** State where 169 FW list comes from
2. **Sensitivity analysis:** Test 50, 100, 150, 200 FW
3. **MFW approach:** Use corpus-derived most frequent words

**Current Status:** PARTIALLY ADDRESSED
- List is in code but source not documented
- No sensitivity across list sizes

**Verdict:** MITIGABLE - Should document source and add sensitivity

---

## Summary Table

| Constraint | Classification | Current Status | Action Needed |
|------------|---------------|----------------|---------------|
| N=14 small sample | FUNDAMENTAL | Documented | Strengthen power statement |
| p >> n (169/14) | MITIGABLE | Partially addressed | Add feature size sensitivity |
| Contiguity | ADDRESSED | Run-aggregation | None |
| MORONI=2 | FUNDAMENTAL | Documented | Consider 3-class analysis |
| Narrator ≠ Author | ADDRESSED | Multi-hypothesis framework | None |
| Translation effects | ADDRESSED | Garnett calibration | None |
| Absence ≠ Evidence | ADDRESSED | TOST, BF | None |
| Missing Delta | ADDRESSABLE | Not done | Add Delta baseline |
| Missing visualization | ADDRESSABLE | Not done | Add PCA plot |
| Wilson CI | MITIGABLE | Partially documented | Clarify or replace |
| Feature list | MITIGABLE | Partially documented | Document source, add sensitivity |

---

## Questions for Further Analysis

1. **Can the fundamental N=14 constraint be addressed by Bayesian methods?**
   - Hierarchical models might borrow strength
   - Informative priors from calibration studies

2. **Should we run a 3-class analysis excluding MORONI?**
   - Would increase power for remaining classes
   - Loses information about one narrator

3. **Is Delta baseline essential for publication?**
   - Stylometry reviewers expect it
   - May not change conclusions but adds credibility

4. **What feature set sizes should we test?**
   - 50, 100, 150, 169 FW?
   - Or MFW approach (top N from corpus)?

---

## GPT Consultation: Creative Mitigations

### For N=14 Constraint

**Real mitigations (not cosmetic):**

1. **Hierarchical/mixed-effects models** - Treat windows as nested within runs
   - Use many 1000-word windows but model with run random effects
   - Inference stays at run level; windows stabilize estimation
   - Implementation: Bayesian multinomial regression with run random intercepts

2. **Simulation-based sensitivity analysis** - Show what effects ARE detectable
   - Fit generative model to observed feature distributions
   - Simulate corpora with varying effect sizes
   - Report "minimum detectable effect" (MDE)
   - Example: "With N=14, we can detect BA > 40% but not 30-35%"

3. **Permutation tests at run level** - Valid small-sample inference
   - Already implemented in current script
   - This is the RIGHT approach for N=14

4. **Bayesian methods with calibration priors** - Borrow strength from Garnett
   - Use Garnett corpus to set informative priors on effect sizes
   - Must show sensitivity to prior strength
   - Treat as "regularization + calibration" not magic

### For MORONI=2 Constraint

**Real mitigations:**

1. **Exclude from confirmatory analysis** - Run 3-class (MORMON, NEPHI, JACOB)
   - Cleaner inference for adequately-sampled classes
   - Report MORONI exploratorily only

2. **Open-set / anomaly framing** - Don't try to learn MORONI
   - Train model on non-MORONI narrators
   - Ask: "Is MORONI atypical relative to others?"
   - Doesn't require stable MORONI class estimate

3. **Posterior predictive placement** (Bayesian)
   - Fit hierarchical model on well-sampled narrators
   - Compute predictive likelihood of MORONI under each narrator model
   - Report with honest uncertainty

**NOT real mitigations:**
- Data augmentation (synthetic samples) - Circular and misleading
- Few-shot learning techniques - Doesn't solve identifiability
- Merging with another class - Label engineering without justification

### For Narrator ≠ Author Constraint

**Real mitigations:**

1. **Better framing** - Essential but cosmetic
   - "Stylistic differentiation" not "authorship"
   - Explicit underdetermination statement
   - Already done in HYPOTHESIS-PREDICTIONS.md

2. **Benchmark comparisons** - REAL calibration
   - Compare to single-author multi-voice texts
   - Compare to multi-author single-translator texts
   - Garnett calibration partially does this

3. **Discriminating tests** - REAL additional evidence
   - Content/genre controls: if differences are topic artifacts, controlling for topic should collapse separation
   - Quotation sensitivity: remove biblical quotations and retest
   - Within-narrator heterogeneity vs between-narrator patterns

4. **Accept underdetermination** - Honest conclusion
   - "Data support stylistic differentiation; authorship remains underdetermined"
   - This is philosophically and methodologically valid

---

## DSH Editor Perspective: What's Needed for Publication

### MUST ADD (Required for acceptance)

| Item | Current Status | Action Needed |
|------|----------------|---------------|
| CI on balanced accuracy | Not in run-aggregated | Add run-level bootstrap CI |
| Sensitivity/MDE analysis | Mentioned but not formal | Add explicit "detectable effect" statement |
| Alternative feature sets | Only 169 FW | Add character n-grams test |
| Alternative models | Only LogReg | Add SVM or Delta baseline |
| Confusion matrix plot | In report but basic | Enhance visualization |
| Permutation null distribution plot | Not visualized | Add plot |
| Matched 4-class calibration | Garnett is 4-class but different structure | Consider if Garnett is sufficient |

### SHOULD ADD (Strong revision request)

| Item | Current Status | Action Needed |
|------|----------------|---------------|
| Per-class performance with CI | Computed but not prominent | Add caterpillar/forest plot |
| Feature stability visualization | Not present | Add coefficient CI across folds |
| Segment-length sensitivity | Not tested | Add test with fixed-length windows |
| Confound probes | Quotations excluded | Add book/section prediction test |

### NICE TO HAVE (Minor revision suggestions)

| Item | Current Status | Action Needed |
|------|----------------|---------------|
| PCA/UMAP exploration | Not present | Add exploratory plot |
| Comparison to prior BoM studies | Not discussed | Add literature comparison |
| Repeated CV estimate | LOO only | Consider repeated k-fold |

---

## Synthesis: What's Actually Unaddressable?

After consulting GPT from multiple angles, here's the honest assessment:

### TRULY UNADDRESSABLE (Accept and Document)

1. **N=14 is fixed** - Cannot create more independent runs
   - MITIGATION: Simulation-based MDE, permutation tests, hierarchical models
   - HONEST FRAMING: "Constrained inference with limited power"

2. **Narrator ≠ Author mapping is not 1:1** - Fundamental identifiability
   - MITIGATION: Calibration benchmarks, discriminating tests
   - HONEST FRAMING: "Stylistic differentiation, authorship underdetermined"

### ADDRESSABLE WITH EFFORT

1. **p >> n** - Can reduce features or use dimensionality reduction
   - ACTION: Add feature set sensitivity (50, 100, 150, 169 FW)

2. **MORONI=2** - Can exclude or treat as exploratory
   - ACTION: Run 3-class analysis; report MORONI separately

3. **Missing Delta baseline** - Straightforward to implement
   - ACTION: Add Burrows' Delta comparison

4. **Missing visualization** - Code addition
   - ACTION: Add permutation null plot, PCA, confusion matrix enhancement

5. **Wilson CI issues** - Can replace or document
   - ACTION: Use permutation-based CI or document limitation clearly

---

## Recommended Action Plan

### Tier 1: Essential (Required for any publication)
1. Add run-level bootstrap or permutation-based CI
2. Add explicit MDE/sensitivity statement
3. Add Delta baseline OR character n-gram comparison
4. Add permutation null distribution visualization
5. Enhance confusion matrix and per-class reporting

### Tier 2: Strong (Highly recommended)
1. Run 3-class analysis excluding MORONI
2. Add feature set sensitivity (vary FW count)
3. Add segment-length sensitivity test
4. Add book/section prediction as confound probe

### Tier 3: Desirable (Would strengthen paper)
1. Add PCA/UMAP exploratory visualization
2. Add Bayesian analysis with Garnett-informed priors
3. Add comparison to prior BoM stylometry literature
4. Add coefficient stability visualization

---

## Decision Points for User

1. **Scope question:** Is this a methods paper (showing the pipeline works) or a substantive paper (claiming something about BoM)?

2. **MORONI question:** Should we exclude MORONI from confirmatory analysis?

3. **Delta question:** Is adding Burrows' Delta worth the implementation effort?

4. **Visualization question:** How much visualization is appropriate for a supplementary analysis?

5. **Publication target:** Is DSH/LLC the target, or is this for the project documentation?

