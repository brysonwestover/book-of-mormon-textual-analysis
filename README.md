# LLM-Assisted Textual Analysis of the Book of Mormon

An open-source framework for applying large language models to the textual analysis of disputed historical documents, beginning with the Book of Mormon.

**Status:** Phase 1 (Protocol + Pilot + Infrastructure)
**Methodology Version:** 2.0 (February 2026)
**Maintainer:** Single-maintainer project; contributions welcome

## Project Philosophy

This project examines the Book of Mormon's claim to be an ancient historical document compiled over approximately 1000 years (600 BC - 400 AD) versus the alternative hypothesis that it was composed in the 19th century.

**This project is ONE TOOL in the broader research landscape—not a decisive verdict.** The goal is insight, not conclusion. Ambiguous results will be presented as ambiguous.

### Core Commitments

1. **Text-primary analysis** — The text is the primary evidence; reference corpora are measurement instruments, not authorities
2. **Multi-hypothesis framework** — We test against five distinct hypotheses, not a binary dichotomy
3. **Transparent methodology** — Every prompt, output, and decision is versioned in Git
4. **Falsifiable claims** — We specify in advance what would support or undermine each hypothesis
5. **Acknowledged limitations** — LLMs are trained on existing scholarship; we document this openly
6. **No predetermined conclusion** — Mixed results are reported as mixed

## Methodology Development

The methodology for this project was developed through structured adversarial dialogue between:
- **Claude** (Anthropic)
- **GPT-5.2 Pro** (OpenAI)

Both models critiqued each other's reasoning until consensus was reached. The full dialogue is preserved in `/docs/methodology-dialogue.md`. This process itself demonstrates the project's commitment to transparency and multi-perspective validation.

## Hypothesis Framework

Rather than forcing evidence into a binary "ancient vs. modern" dichotomy, we assess findings against five competing hypotheses:

| Hypothesis | Description |
|------------|-------------|
| **H1** | Modern single-author composition |
| **H2** | Modern multi-source composition (single author + sources) |
| **H3** | Modern collaborative composition (multiple 19th-c contributors) |
| **H4** | Ancient multi-author compilation with single translator |
| **H5** | Deliberate ancient-style pseudepigraphy |
| **H0/UX** | Indeterminate (evidence insufficient to discriminate) |

See [METHODOLOGY.md](METHODOLOGY.md) for detailed predictions under each hypothesis.

## What This Project Is

- A **methodological framework** for transparent, reproducible textual analysis
- A **pilot benchmark** with single-annotator (silver) labels
- A **protocol-first project** releasing annotation manuals and tooling for future expansion
- An **exercise in epistemic humility** about what LLM-assisted analysis can determine

## What This Project Is Not

- An attempt to "prove" or "disprove" the Book of Mormon
- A religious or anti-religious project
- A claim that LLMs can definitively resolve historical questions
- A substitute for traditional scholarly methods
- A "gold standard" benchmark (Phase 1 produces silver labels only)

## Phase Structure

### Phase 1: Protocol + Pilot + Infrastructure (Current — Solo)

What a single maintainer can credibly produce:

- [ ] Task specifications for each analytical module
- [ ] Annotation guidelines v1
- [ ] Pilot dataset with silver labels
- [ ] Evaluation harness
- [ ] Non-LLM baselines
- [ ] Control corpora identification
- [ ] Source text acquisition and documentation

**Constraints:** All findings are exploratory. No "gold label" claims. No strong generalizations.

### Phase 2: Gold Labels + Scale (Future — Collaborative)

Requires additional contributors:

- 3+ annotators with training and calibration
- Inter-annotator agreement reporting
- Larger stratified sampling
- Expanded control corpora
- Dataset governance

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help.

## Analysis Modules

Each module follows the 12-element template specified in [METHODOLOGY.md](METHODOLOGY.md):

### 1. Voice Variation Analysis
Do claimed authors exhibit distinct, consistent stylistic signatures? Operationalized via function words, syntactic patterns, and discourse markers with calibration against known single-author and multi-author texts.

### 2. Anachronism Analysis
Does the text contain concepts inappropriate to its claimed timeframe? Requires explicit dating criteria and reference corpora.

### 3. Linguistic Marker Analysis
KJV dependency patterns, Early Modern English features, 19th-century markers. Calibrated against diachronic corpora.

### 4. Narrative Structure Analysis
Ancient chronicle conventions vs. 19th-century literary conventions. Operationalized structural features with comparison texts.

### 5. Internal Consistency Analysis
Geographic, chronological, and genealogical consistency. Knowledge graph construction with explicit contradiction criteria.

### 6. Theological Development Analysis
Does theology evolve across the claimed timeframe? Topic modeling and doctrinal phrase tracking.

## Repository Structure

```
├── README.md                 # This file
├── METHODOLOGY.md            # Detailed analytical framework (v2.0)
├── LIMITATIONS.md            # Explicit statement of constraints
├── CONTRIBUTING.md           # How to contribute
├── docs/
│   └── methodology-dialogue.md  # Full Claude/GPT-5.2 Pro dialogue
├── prompts/
│   ├── system/               # System prompts
│   └── modules/              # Module-specific prompts
├── analysis/
│   └── modules/              # Module implementations
├── runs/
│   └── {model}-{date}/       # Outputs by model and date
├── results/
│   ├── exploratory/          # Phase 1 exploratory findings
│   └── summaries/            # Synthesized findings
├── data/
│   ├── text/                 # Book of Mormon text (1830 edition)
│   ├── reference/            # Control corpora
│   └── labels/               # Silver labels (Phase 1)
└── scripts/
    ├── run_analysis.py
    └── evaluate.py
```

## Source Text

We use the **1830 first edition** from Project Gutenberg (public domain, freely available).

We document known differences between this edition and:
- Skousen's Earliest Text
- Modern LDS editions
- Original/printer's manuscripts (where relevant)

If textual variants affect analysis, this is noted explicitly.

## LLM Use Constraints

LLMs in this project are used for:
- **Structured annotation** (validated against labeled data)
- **Hypothesis generation** (not evidence)
- **Cross-model robustness checks** (not validation)

LLMs are NOT used for:
- Holistic authorship judgments
- Unvalidated classification
- Truth determination

See [METHODOLOGY.md](METHODOLOGY.md#llm-use-constraints) for details.

## Epistemic Standards

### What Would Count as Evidence?

Each hypothesis makes specific, testable predictions. For example:

**Supporting H4 (ancient compilation + translator):**
- Weak micro-stylistic separability (translator smoothing)
- Strong discourse-structure variation
- Pattern matching translated multi-author controls

**Supporting H1 (modern single author):**
- Strong global stylistic coherence
- Voice variation matching single-author character differentiation
- Intertextual dependence on 19th-c corpora

**Indeterminate:**
- Evidence fits multiple hypotheses equally well
- Calibration shows method cannot discriminate
- Results unstable across robustness checks

### What Cannot Be Determined

- Historical truth of described events
- Divine involvement or lack thereof
- Author's subjective experience or intent
- "Authenticity" in any religious sense

See [LIMITATIONS.md](LIMITATIONS.md) for full details.

## Contributing

This project welcomes contributors from all perspectives:
- Believing Latter-day Saints
- Former members
- Religious scholars
- Secular historians
- Computational linguists
- LLM researchers

The only requirement is commitment to methodological rigor and intellectual honesty.

### How to Help

**Phase 1 (Current):**
- Review methodology for weaknesses
- Propose prompt improvements
- Identify control corpora
- Test reproducibility

**Phase 2 (Future):**
- Become an annotator
- Contribute to adjudication
- Expand control corpora
- Validate baselines

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Transparency Commitment

This repository contains:
- All prompts used
- All model outputs (archived)
- All analytical decisions and rationale
- The full methodology development dialogue
- Version history of all documents

Anyone can audit exactly how conclusions were reached.

## License

MIT License — Use freely, attribute appropriately.

## Acknowledgments

### Methodology Development
- Claude (Anthropic, claude-opus-4-5-20250101)
- GPT-5.2 Pro (OpenAI)

### Methodological Inspiration
- Computational stylometry research
- Documentary hypothesis methodology
- Digital humanities best practices
- Adversarial collaboration frameworks
- Open science pre-registration practices

## Citation

If you use this framework or methodology, please cite:

```
Book of Mormon Textual Analysis Project. (2026).
LLM-Assisted Textual Analysis of the Book of Mormon:
A Multi-Hypothesis Framework.
https://github.com/brysonwestover/book-of-mormon-textual-analysis
```

## Contact

For questions, critiques, or collaboration inquiries, open an issue or see [CONTRIBUTING.md](CONTRIBUTING.md).
