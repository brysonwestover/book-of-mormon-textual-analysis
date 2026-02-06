# Methodology

This document details the analytical framework, epistemological commitments, and procedural standards for this project.

**Version 2.0** — This methodology was developed through structured dialogue between Claude (Anthropic) and GPT-5.2 Pro (OpenAI) in January 2026. The full dialogue is preserved in `/docs/methodology-dialogue.md`.

## Table of Contents

1. [Core Principles](#core-principles)
2. [Hypothesis Framework](#hypothesis-framework)
3. [Evidence Scope](#evidence-scope)
4. [Module Template](#module-template)
5. [LLM Use Constraints](#llm-use-constraints)
6. [Phase Structure](#phase-structure)
7. [Interpretation Guidelines](#interpretation-guidelines)
8. [Explicit Limitations](#explicit-limitations)
9. [Open Design Choices](#open-design-choices)

---

## Core Principles

### 1. Text-Primary Analysis

The Book of Mormon text is the primary object of analysis. We reason from internal textual features. However, "text-primary" does not mean "text-only":

- **Reference corpora are measurement instruments**, not external authorities
- We do not cite scholars (Sperry, Bushman, Skousen, etc.) as authorities for our conclusions
- We argue from reproducible measurements against specified reference datasets
- Dating and anachronism claims are impossible without external corpora—this is acknowledged, not hidden

### 2. Acknowledged Limitations

We do not claim to be "unbiased" or to operate from "first principles" in any pure sense:

- LLMs (Claude, GPT-5.2 Pro) are trained on existing scholarship—apologetic, critical, and academic
- This training inevitably shapes their analysis
- Our methodology embeds design choices that reflect assumptions
- These limitations are documented openly, not hidden

### 3. Transparent Methodology

Every analytical choice is documented and justified:

- All prompts, outputs, and reasoning chains are versioned in Git
- Anyone can audit exactly how we reached our conclusions
- Prompt development process is documented (iterations, changes, rationale)

### 4. Falsifiable Claims

Each analytical module specifies IN ADVANCE:

- What specific textual features we're examining
- What findings would support each hypothesis
- What findings would be ambiguous or inconclusive
- What would count AGAINST our conclusions

We commit to these criteria before running the analysis, not after.

### 5. Multi-Model Validation

Claude and GPT-5.2 Pro critique each other's reasoning:

- Disagreements are documented and explored, not suppressed
- Consensus is noted but not forced
- **Important:** Multi-model agreement is a robustness check, not validation. Agreement does not establish truth—it reduces single-model idiosyncrasy.

### 6. Reproducible by Skeptics

Anyone can clone the repository, run the same prompts against the same text, and verify our outputs:

- Source text is explicitly versioned (1830 edition)
- Model versions, parameters, and seeds are documented
- Dependencies are pinned; environment is containerized
- Corpus hashes are recorded for integrity verification

### 7. No Predetermined Conclusion

We are not attempting to prove or disprove the Book of Mormon's historical claims:

- We are not attempting to validate or attack Mormonism
- Mixed or ambiguous results are presented honestly as mixed or ambiguous
- The goal is to surface what the text reveals, not to deliver a verdict

### 8. Living Project

This analysis is designed to be re-run as LLMs improve:

- Results are versioned by model and date
- Future analyses can be compared against earlier ones
- Model drift triggers documented re-runs

### 9. Operational Definitions

Every analytical claim must specify measurable features, extraction methods, thresholds, and decision rules before analysis begins:

- "Distinct voices" means nothing without defining what features constitute voice
- "Anachronism" requires specifying what counts and how it's dated
- Thresholds are calibrated on control corpora, not the target text

### 10. Comparison Framework

Control corpora are required measurement instruments:

- Known single-author 19th-century religious texts
- Known multi-author compiled texts
- Pseudo-archaic imitations from 18th-19th centuries
- Diachronic English corpora (COHA, EEBO/TCP)
- Translated ancient multi-author texts with single translator

Each corpus requires an "instrument spec": selection criteria, matching variables, known confounds, and intended inferential target.

### 11. Translation Layer Modeling

The Book of Mormon as we have it is a claimed translation. Any stylometric analysis must explicitly model:

- What stylistic signals should survive translation
- What signals are likely translator artifacts
- Whether translator produces consistent vs. variable English

This affects expected baselines and must be specified before analysis.

### 12. Multi-Hypothesis Framework

We do not use a binary "ancient vs. modern" framing. Instead, we assess evidence against multiple competing hypotheses (see below).

---

## Hypothesis Framework

Rather than forcing evidence into a binary dichotomy, we assess findings against five distinct hypotheses:

### H1: Modern Single-Author Composition

The text was composed by a single 19th-century author (Joseph Smith or another individual), drawing on personal knowledge, available sources, and imagination.

**Textual predictions:**
- Strong global stylistic coherence
- Detectable intertextual dependence on 18th-19th century corpora
- Theological concepts matching 19th-century Protestant formulations
- Voice variation consistent with character differentiation by a single author

### H2: Modern Multi-Source Composition

A single 19th-century author composed the text while drawing heavily on multiple sources (KJV, contemporary sermons, oral traditions, etc.).

**Textual predictions:**
- One dominant "authorial voice" with blocky source insertions
- Seams correlating with source-like shifts
- High between-segment divergence in source sections
- Unified theological perspective with imported material

### H3: Modern Collaborative Composition

Multiple 19th-century individuals contributed to composition (Joseph Smith + scribes/collaborators).

**Textual predictions:**
- Multiple recurring stylistic signatures
- Voices interleaving across topics/genres
- Boundaries appearing even within same discourse mode
- Contributor-like signatures persisting across topics

### H4: Ancient Multi-Author Compilation with Single Translator

The text derives from ancient sources compiled over centuries, rendered into English by a single 19th-century translator whose voice may dominate surface features.

**Textual predictions:**
- Lower micro-stylistic separability (translator smoothing)
- Stronger signal in discourse organization than surface style
- Change-points aligning with structural markers (colophons, record switches)
- Pattern resembling translated multi-author texts with single translator

### H5: Deliberate Ancient-Style Pseudepigraphy

A modern author deliberately imitated ancient compositional conventions and claimed ancient authorship (similar to ancient pseudepigraphical works).

**Textual predictions:**
- Artificial uniformity across purported voices
- Boundaries following narrative framing claims ("and now I, X...")
- Surface archaism + modern conceptual clustering
- Strong KJV dependence + conscious ancient literary imitation

### H0/UX: Indeterminate

Evidence is insufficiently discriminative to favor any hypothesis. This is the default condition—we do not force selection among H1-H5 when evidence doesn't warrant it.

---

## Evidence Scope

### Included: Internal Textual Evidence

| Category | Examples |
|----------|----------|
| Stylometric features | Function words, syntactic patterns, character n-grams, sentence structure |
| Discourse structure | Mode shifts, rhetorical moves, narrative framing |
| Theological content | Doctrinal concepts, religious terminology, soteriological frameworks |
| Linguistic features | Grammar, syntax, archaic forms, KJV dependency patterns |
| Narrative structure | Plot patterns, framing devices, genre conventions |
| Internal references | Cross-references, chronology, geography, genealogy |

### Required: Reference Corpora (Measurement Instruments)

| Corpus Type | Purpose |
|-------------|---------|
| Single-author 19th-c religious texts | Baseline for H1 |
| Multi-author compiled texts | Calibration for voice detection |
| Pseudo-archaic imitations | Baseline for H5 |
| Translated multi-author with single translator | Baseline for H4 |
| Diachronic corpora (COHA, EEBO) | Dating linguistic features |

### Excluded: External Evidence

| Category | Reason for Exclusion |
|----------|---------------------|
| Archaeological evidence | Outside textual analysis scope |
| DNA evidence | Outside textual analysis scope |
| Witness testimony | Cannot be textually analyzed |
| Spiritual experiences | Not intersubjectively verifiable |

---

## Module Template

Every analytical module must include these 12 elements:

### 1. Operational Definitions
- Specific features to measure
- What constitutes the phenomenon being studied
- Decision rules for classification

### 2. Method Specification
- How features will be extracted
- What tools/algorithms will be used
- How LLMs will be employed (annotation only, not judgment)

### 3. Threshold Pre-Commitment
- Quantitative thresholds for each outcome
- Calibrated on control corpora before touching target text

### 4. Calibration Plan
- Which control corpora will be used
- What benchmarks must be met before proceeding
- Error rate estimation (false positive/negative)

### 5. Falsification Criteria
- What would count AGAINST each hypothesis
- Explicit, testable statements

### 6. Translation-Layer Considerations
- What signals should survive translation
- What signals are likely translator artifacts
- How this affects interpretation

### 7. Segmentation Specification
- Unit of analysis (verse, chapter, pericope, etc.)
- Boundary rules
- Handling of embedded quotations
- Exclusion rules

### 8. Primary Outcomes + Multiplicity Plan
- 1-3 key metrics per module
- Pre-specified thresholds
- Multiple comparisons strategy (FDR control, etc.)

### 9. Uncertainty Reporting
- Confidence intervals / credible intervals
- Effect sizes, not just direction
- Bootstrap stability where applicable

### 10. Robustness Checks
- Feature ablation tests
- Prompt-family stability tests
- Cross-model replication

### 11. Negative Controls / Adversarial Tests
- Segment shuffle (should destroy signal)
- Decoy corpus insertion (should not find false positives)
- LLM leakage checks (canary strings, external fact detection)

### 12. Data Leakage / Contamination Assessment
- Document whether models likely saw relevant scholarship
- Treat contamination as bias term, not footnote

---

## LLM Use Constraints

### What LLMs Can Do (Credibly)

- **Structured annotation/extraction**: Speaker identification, discourse markers, span classification
- **Feature extraction**: Converting text to computable annotations with validation
- **Hypothesis generation**: Proposing candidate patterns to test
- **Cross-critique**: Identifying weaknesses in each other's reasoning (robustness check)

### What LLMs Cannot Do (Credibly)

- **Holistic authorship judgments**: "This feels like multiple authors" is not evidence
- **Unvalidated classification**: Labels require validation against gold/silver labels
- **Definitive dating**: "This is anachronistic" requires external corpus support
- **Truth determination**: LLM agreement does not establish ground truth

### Required Safeguards

1. **Structured outputs**: JSON with schema validation; no freeform judgments as evidence
2. **Local context only**: Limit context window to reduce global narrative priors
3. **Multiple model families**: At least two different vendors/architectures
4. **Closed justification codes**: Force selection from pre-defined categories
5. **Leakage detection**: Flag any mention of scholars, historical claims, or external facts
6. **Validation requirement**: All extraction must be validated against labeled test sets

### Framing

LLM outputs are **model-mediated commentary**, not analysis. They are useful for:
- Generating hypotheses to test
- Surfacing candidate passages/features
- Producing interpretive readings (explicitly labeled as such)

They are not useful as independent evidence for authorship/historicity claims.

---

## Phase Structure

### Phase 1: Protocol + Pilot + Infrastructure (Solo)

**Scope:** What a single maintainer can credibly produce.

**Deliverables:**
- Task specifications for each module
- Annotation guidelines v1
- Pilot dataset with **silver labels** (single-annotator, expert-coded)
- Evaluation harness (standardized input/output, metrics, splits)
- At least one non-LLM baseline per module
- Intra-annotator reliability check (re-label subset after time gap)
- Control corpora identification and documentation
- Contribution-ready infrastructure (CONTRIBUTING.md, issue templates, data schemas)

**Constraints:**
- All findings labeled as **exploratory**
- No "gold label" language
- No strong generalization claims
- No definitive model rankings

**Labels:** "Pilot benchmark," "seed dataset," "single-annotator silver labels," "exploratory findings"

### Phase 2: Gold Labels + Scale (Collaborative)

**Scope:** Requires additional contributors.

**Requirements:**
- 3+ annotators with training and calibration
- Inter-annotator agreement (IAA) reporting with pre-specified thresholds
- Adjudication workflow for disagreements
- Larger stratified sampling
- Expanded control corpora
- Dataset governance (versioning, change logs, contributor roles)

**Unlocks:**
- "Gold label" language
- Confirmatory analysis
- Generalizable claims (with appropriate caveats)

### Path to Gold (Documented)

For each module, we document:
- Number of annotators needed
- Training/calibration plan
- Adjudication method
- Target IAA thresholds
- Estimated labeling hours

This allows potential collaborators to see exactly what's needed.

---

## Interpretation Guidelines

### Confidence Levels

| Level | Meaning | Criteria |
|-------|---------|----------|
| Strong | Clear evidence, counter-explanations require special pleading | Multiple independent indicators; calibration supports reliability |
| Moderate | Good evidence, plausible counter-explanations | Clear pattern, but alternatives remain reasonable |
| Weak | Suggestive evidence, strong counter-explanations | Some indication, but easily explained under multiple hypotheses |
| Indeterminate | Evidence does not discriminate among hypotheses | Genuinely ambiguous, insufficient data, or method unreliable |

### Aggregation Protocol

**Pre-commit to one of:**
- (A) Explicit Bayesian model comparison (priors + likelihoods from calibration)
- (B) Pre-registered scoring rubric with ordinal support levels

**Do not mix approaches midstream.**

**Rules:**
- No simple vote counting across modules
- Independence matters—correlated evidence counted once
- "Indeterminate" is always an available outcome
- Contradictions reported as contradictions, not harmonized

### Reporting Requirements

Every analysis report must include:
1. Module specification (from template)
2. Findings with uncertainty quantification
3. Interpretation under each hypothesis (H1-H5)
4. Assessment of which hypotheses are supported/undermined
5. Confidence level with justification
6. Limitations specific to this analysis
7. Robustness check results
8. Negative control results

---

## Explicit Limitations

### What This Analysis Cannot Determine

- **Historicity**: Whether described events occurred
- **Authorship identity**: Who specifically wrote the text
- **Divine involvement**: Whether God was involved
- **Subjective experience**: What Joseph Smith believed or experienced
- **Religious truth**: Whether the book is spiritually valuable

### Structural Limitations

- **Underdetermination is the default condition**: Multiple hypotheses will often fit equally well
- **LLM outputs are not independent observations**: Correlated errors across prompts/models
- **Genre/topic confounds will masquerade as voices**: Even after controls
- **Translation layer may dominate micro-stylistic signals**: Under H4, surface features may not reflect underlying authors
- **Any anachronism claim is conditional**: On reference corpus coverage and dating proxy used

### Resource Limitations (Phase 1)

- **Single-annotator labels**: Cannot claim gold-standard reliability
- **Limited IAA**: Cannot calculate inter-annotator agreement
- **Exploratory only**: Cannot make confirmatory claims
- **Scale constraints**: Pilot-level sampling

### LLM-Specific Limitations

- **Training data contamination**: Models have seen apologetic/critical arguments
- **Prompt sensitivity**: Small changes can alter outputs significantly
- **Non-determinism**: Results vary across runs unless controlled
- **Citation hallucination**: Models may fabricate scholarly framing
- **Identifiability problem**: Cannot fully separate text-derived signal from training-derived regurgitation

---

## Open Design Choices

These decisions must be made and documented before Phase 1 analysis begins:

### 1. Base Text / Edition Policy
- Which edition(s) are primary
- How variants/editions are handled
- Normalization rules (spelling, punctuation)

### 2. Control-Corpus Matching Strategy
- Required matching variables (time, genre, register, theology)
- Handling of imperfect matches
- Minimum corpus requirements

### 3. Segmentation Granularity
- Global vs. module-specific segmentation
- Prevention of "segmentation shopping"
- Minimum segment length rules

### 4. Module-to-Hypothesis Aggregation
- Bayesian updating vs. scoring rubric
- How contradictory module results are handled
- Weight given to different evidence types

### 5. Translation-Layer Operationalization
- Specific tests for translation effects
- Comparable translated corpora selection
- Translator-specific baseline criteria

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-29 | 1.0 | Initial methodology document |
| 2026-02-01 | 2.0 | Complete revision based on Claude/GPT-5.2 Pro methodology dialogue. Added: multi-hypothesis framework (H1-H5), 12 core principles, module template, LLM use constraints, phase structure, explicit limitations, open design choices. |

---

## References

### Methodology Development

The methodology in this document was developed through structured adversarial dialogue between:
- Claude (Anthropic, claude-opus-4-5-20250101)
- GPT-5.2 Pro (OpenAI)

The full dialogue is preserved in `/docs/methodology-dialogue.md`.

### Relevant Literature

*Note: Full scholarly references will be compiled in the Phase 2.E final report. Key areas to be cited include:*
- Computational stylometry methods (Burrows, Eder, Stamatatos)
- Function-word authorship attribution
- Translation effects on stylometric signal
- Documentary hypothesis methodology
- Book of Mormon scholarship (multiple perspectives)

See `docs/PROJECT-WIKI.md` for current status and findings.
