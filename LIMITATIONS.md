# Limitations

This document explicitly states what this project cannot determine, what assumptions it makes, where its methods may fail, and what constraints shape Phase 1 findings. Honest acknowledgment of limitations is essential to intellectual integrity.

**Version 2.0** — Updated February 2026 based on Claude/GPT-5.2 Pro methodology dialogue.

---

## What This Analysis Cannot Determine

### Historical Truth

This analysis **cannot determine** whether:
- The events described in the Book of Mormon actually occurred
- Nephi, Mormon, Moroni, or other figures existed as historical persons
- The Nephite/Lamanite civilizations existed
- Joseph Smith had a genuine religious experience
- Any supernatural claims are true or false

**Why:** Textual analysis examines text characteristics, not historical reality. A text with "ancient" features could still describe fictional events. A text with "modern" features could theoretically still derive from an ancient source.

### Subjective Experience

This analysis **cannot determine**:
- What Joseph Smith believed or experienced
- Whether the composition process was conscious or unconscious
- The psychological state of any person involved
- Whether any form of inspiration occurred

**Why:** Subjective experience is not accessible through textual analysis. The same text could result from deliberate composition, unconscious production, or various other mechanisms.

### Divine Involvement

This analysis **cannot determine**:
- Whether God exists
- Whether God was involved in the Book of Mormon's production
- Whether the book contains divine truth
- Whether reading the book can produce genuine spiritual experiences

**Why:** These are metaphysical and theological questions outside the scope of textual analysis. Methodological naturalism is a scoping choice, not a metaphysical conclusion.

### Definitive Authorship

This analysis **cannot definitively determine**:
- Whether the text has one human author or multiple
- The identity of the author(s)
- The exact process of composition

**Why:** Textual evidence provides probabilistic indicators, not certainties. Skilled single authors can simulate multiple voices; multi-author texts can be homogenized by editors or translators.

---

## Structural Limitations

### Underdetermination Is the Default Condition

Multiple hypotheses will often fit the same textual evidence equally well. This is not a failure of the methodology—it is a fundamental constraint on what textual analysis can achieve.

**Implication:** We do not force conclusions. When evidence is ambiguous, we report "indeterminate" (H0/UX) rather than selecting the hypothesis we prefer.

### Translation Layer Confounding

The Book of Mormon as we have it is a claimed translation. This creates interpretive problems:

- A single 19th-century translator could homogenize style across multiple ancient authors
- A single modern author could simulate stylistic variation
- Micro-stylistic features (function words, character n-grams) may reflect translator voice rather than underlying authors
- Macro-structural features may survive translation better than surface style

**Implication:** Under hypothesis H4 (ancient compilation + single translator), we expect weak micro-stylistic separability. This is not evidence against H4—it is what H4 predicts.

### Genre and Topic Confounding

Genre shifts and topic changes can masquerade as "authorial voices" even after statistical controls:

- Sermons vs. narrative sections have different stylistic profiles regardless of author
- War chapters vs. doctrinal chapters show vocabulary differences
- Quoted material (Isaiah blocks) has different features than original composition

**Implication:** "Voice detection" results must be interpreted cautiously. Topic-control regression reduces but does not eliminate this confound.

### LLM Outputs Are Not Independent Observations

When Claude and GPT-5.2 Pro agree on something, this does not establish truth:

- Both models are trained on overlapping corpora
- Both have seen apologetic and critical scholarship about the Book of Mormon
- Agreement may reflect shared training bias rather than independent confirmation
- Correlated errors across prompts and models are expected

**Implication:** Multi-model agreement is a robustness check (reducing single-model idiosyncrasy), not validation. Independent ground truth is required for validation.

### Anachronism Claims Are Conditional

Any claim that something is "anachronistic" is conditional on:

- The reference corpus used for dating
- The dating proxy employed
- Scholarly consensus about when concepts/terms emerged
- Coverage gaps in historical records

**Implication:** Anachronism findings should specify their reference frame explicitly and acknowledge that historical knowledge is incomplete.

---

## Resource Limitations (Phase 1)

Phase 1 is a single-maintainer project. This creates specific constraints:

### What Phase 1 Cannot Claim

| Claim Type | Why Not Credible in Phase 1 |
|------------|---------------------------|
| "Gold labels" | Requires multiple independent annotators + agreement process |
| Inter-annotator agreement (IAA) | Requires ≥2 annotators by definition |
| Strong benchmarking | "This is the benchmark for X" requires community validation |
| Generalizable model rankings | Sample sizes and single-annotator labels don't support this |
| Definitive conclusions | Phase 1 is exploratory only |

### What Phase 1 Can Claim

| Claim Type | With Appropriate Framing |
|------------|-------------------------|
| Pilot benchmark | "Seed dataset for future expansion" |
| Silver labels | "Single-annotator expert-coded labels" |
| Exploratory findings | "Suggestive patterns requiring confirmation" |
| Protocol contribution | "Annotation manual + tooling for scaling" |
| Infrastructure | "Reproducible pipeline for others to use" |

### Path to Stronger Claims (Phase 2 Requirements)

For each module, Phase 2 would require:
- 3+ trained annotators
- Pre-specified IAA thresholds (e.g., Krippendorff's α ≥ 0.67)
- Adjudication workflow for disagreements
- Larger stratified sampling
- Expanded control corpora with documented matching criteria

---

## LLM-Specific Limitations

### Training Data Contamination

LLMs have been trained on text that includes:
- Apologetic scholarship defending Book of Mormon historicity
- Critical scholarship arguing for 19th-century composition
- Popular discussions of both positions
- Existing stylometric studies
- Lists of claimed "anachronisms" and "Hebraisms"

**Implication:** LLM outputs may reflect regurgitated arguments rather than fresh textual analysis. We cannot fully separate text-derived signal from training-derived recall.

### Prompt Sensitivity

Small changes in prompt wording can significantly alter LLM outputs:
- Different framings produce different "conclusions"
- Order of presentation matters
- Specific word choices trigger different associations

**Implication:** Prompt development is documented. Robustness across prompt families is tested. Results that flip with minor prompt changes are flagged as unstable.

### Non-Determinism

LLM outputs vary across runs even with identical prompts:
- Temperature settings affect variability
- API models may change without notice (model drift)
- Results may not reproduce exactly over time

**Implication:** We pin model versions where possible, archive outputs, run multiple replications, and treat instability as data about reliability.

### Citation Hallucination

LLMs may:
- Fabricate scholarly sources that don't exist
- Misattribute claims to scholars
- Present invented "facts" with confident framing
- Produce rhetorically sophisticated but evidentially shallow analysis

**Implication:** LLM outputs are not trusted as sources of external facts. Any factual claims must be independently verified.

### Identifiability Problem

The core skeptical objection:

> "Your primary instrument (LLMs) is not a valid or interpretable measurement device because it is trained on the very discourse you're analyzing, and you cannot disentangle text-derived signal from training-derived regurgitation; therefore your outputs are not evidential, only rhetorical."

**Our response:**
- Constrain LLMs to validated annotation tasks, not holistic judgments
- Show stability across different model families
- Include strong negative controls
- Demonstrate calibration on texts where ground truth is known
- Treat LLM outputs as "model-mediated commentary," not evidence

This response mitigates but does not eliminate the problem.

---

## Methodological Limitations

### Stylometric Limitations

**Problem:** Stylometry works best with large samples and clear authorial boundaries. The Book of Mormon's claimed authorial sections are of varying lengths, and editorial/translation processes may have altered underlying stylistic signatures.

**Implication:** Stylometric findings should be treated with appropriate uncertainty. Calibration against control texts is required.

### Anachronism Assessment Limitations

**Problem:** Assessing theological or conceptual anachronism requires knowing what ideas were available when. Historical scholarship on both pre-exilic Israelite religion and 19th-century American religion continues to evolve.

**Implication:** Anachronism claims are only as strong as the underlying historical scholarship, which we do not produce ourselves.

### Linguistic Expertise Limitations

**Problem:** Proper linguistic analysis (especially claims about Hebraisms, Early Modern English, etc.) requires specialized expertise that LLMs may not reliably have.

**Implication:** Linguistic findings should be flagged for verification by qualified experts. We document competence boundaries.

### Comparison Text Limitations

**Problem:** Ideal analysis would compare the Book of Mormon to perfect comparison texts (e.g., confirmed ancient American records, confirmed 19th-century religious fictions with known composition processes). Such perfect comparisons don't exist.

**Implication:** Our baselines are imperfect. We document what each control corpus can and cannot tell us.

### Researcher Degrees of Freedom

Even with pre-commitment, choices remain:
- Which features to prioritize
- How to segment the text
- Which thresholds to use
- How to weight different evidence types
- How to synthesize across modules

**Implication:** We document choices transparently, run sensitivity analyses, and acknowledge that different reasonable choices might yield different conclusions.

---

## Specific Claims We Do Not Make

### Claims Outside Our Scope

| Claim | Why Outside Scope |
|-------|-------------------|
| "The Book of Mormon is true/false" | Truth in religious sense is outside analytical scope |
| "Joseph Smith was a prophet/fraud" | Prophetic status and intent are not textually determinable |
| "The witnesses saw/didn't see plates" | Witness testimony is outside our evidence scope |
| "The book is/isn't spiritually valuable" | Spiritual value is outside analytical scope |

### Claims We Investigate (With Uncertainty)

| Claim | How We Approach It | Confidence Possible |
|-------|-------------------|---------------------|
| "The text shows multiple authorial voices" | Stylometric analysis with calibration | Exploratory (Phase 1) |
| "Theological concepts fit/don't fit claimed timeframe" | Historical comparison with reference corpora | Conditional on dating framework |
| "The text contains genuine Hebraisms" | Linguistic pattern detection | Requires expert validation |
| "The narrative structure is ancient/modern" | Comparative structural analysis | Dependent on control corpus quality |
| "The text is internally consistent" | Systematic consistency checking | Exploratory |

---

## What Would Change Our Conclusions

We commit to intellectual honesty by specifying what would cause revision:

### If Analysis Seems to Favor Ancient Origin (H4)

We would revise toward greater skepticism if:
- Calibration tests show our methods cannot distinguish known single-author from multi-author texts
- Scholars identify methodological flaws in our approach
- The pattern we attribute to "multiple ancient voices" appears in known single-translator texts
- Replication with different models produces contradictory results
- Negative controls show false positives

### If Analysis Seems to Favor Modern Origin (H1/H2/H3/H5)

We would revise toward greater skepticism if:
- Calibration tests show our methods cannot reliably detect single authorship
- Scholars identify methodological flaws in our approach
- Features we call "19th-century markers" appear in ancient translated texts
- Replication with different models produces contradictory results
- Negative controls show false negatives

### If Results Are Indeterminate (H0/UX)

This is the expected default outcome for many modules. Indeterminate results mean:
- The method cannot discriminate among hypotheses
- Evidence genuinely supports multiple interpretations
- More data or better methods are needed

Indeterminate is a valid, informative outcome—not a failure.

---

## A Note on Humility

The question of the Book of Mormon's origins has been debated for nearly two centuries by intelligent people who disagree. We do not expect that LLM-assisted textual analysis will definitively resolve this question.

Our goal is more modest:
1. Demonstrate transparent, reproducible methodology
2. Generate exploratory findings that can inform further research
3. Model how contested questions can be approached with intellectual honesty
4. Build infrastructure that others can extend and improve
5. Contribute to constructive dialogue across perspectives

We may be wrong. We invite correction.

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-29 | 1.0 | Initial limitations document |
| 2026-02-01 | 2.0 | Major revision based on Claude/GPT-5.2 Pro methodology dialogue. Added: structural limitations, resource limitations, LLM-specific limitations, identifiability problem, Phase 1 constraints, H0/UX framing. |
