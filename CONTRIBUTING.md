# Contributing

We welcome contributions from all perspectives. This document outlines how to contribute effectively to each phase of the project.

**Version 2.0** — Updated February 2026 to reflect phase structure.

## Project Phases

### Phase 1 (Current — Solo Maintainer)

Phase 1 produces exploratory findings with silver labels. Contributions focus on:
- Methodology review and critique
- Control corpora identification
- Prompt improvement
- Infrastructure testing
- Documentation improvements

### Phase 2 (Future — Collaborative)

Phase 2 produces confirmatory findings with gold labels. Contributions include:
- Annotation (requires training)
- Adjudication
- Expanded control corpora
- Validation studies

---

## Phase 1 Contribution Opportunities

### 1. Methodology Critique (High Value)

The methodology was developed through Claude/GPT-5.2 Pro dialogue but benefits from human review.

**We especially need:**
- Identification of blind spots or biases
- Suggestions for additional hypotheses
- Critique of falsification criteria
- Identification of missing negative controls

**How to contribute:**
- Open an issue with label `methodology`
- Reference specific sections of METHODOLOGY.md
- Propose concrete improvements

### 2. Control Corpora Identification

We need comparison texts for calibration. See METHODOLOGY.md for requirements.

**Needed corpora:**
- Single-author 19th-century religious texts (sermons, conversion narratives)
- Multi-author compiled texts with known authorship
- Pseudo-archaic biblical imitations (18th-19th century)
- Translated multi-author ancient texts (single translator)
- Diachronic English corpora (or pointers to existing ones)

**How to contribute:**
- Open an issue with label `control-corpus`
- Include: title, author(s), date, availability, licensing status
- Explain why this corpus is useful for calibration

### 3. Prompt Improvement

Prompts must follow constraints in METHODOLOGY.md (structured outputs, no holistic judgments).

**Before submitting:**
- Ensure prompt produces structured (JSON) output
- Check for leading language
- Verify both hypothesis directions are considered
- Include justification code categories (closed set)

**How to contribute:**
- Fork repository
- Add/modify prompts in `prompts/` directory
- Open PR with rationale

### 4. Infrastructure Testing

Help verify reproducibility.

**Needed:**
- Test that scripts run on different environments
- Verify output format compliance
- Check dependency pinning
- Test evaluation harness

**How to contribute:**
- Clone repo and attempt to reproduce
- Open issues for any failures
- Submit PRs for fixes

### 5. Documentation Improvements

Clarity and accuracy matter.

**Valued contributions:**
- Typo fixes
- Clarification of ambiguous sections
- Additional examples
- FAQ development

---

## Phase 2 Contribution Opportunities (Future)

### Becoming an Annotator

Phase 2 requires 3+ trained annotators per module.

**Requirements:**
- Complete training module (to be developed)
- Pass calibration test
- Commit to labeling quota
- Follow annotation guidelines exactly

**What annotators do:**
- Label segments according to annotation manual
- Flag ambiguous cases
- Participate in adjudication discussions

**Interest registration:**
Open an issue with label `annotator-interest` including:
- Your background/expertise
- Which modules interest you
- Time availability

### Adjudication

Experienced annotators may become adjudicators.

**Requirements:**
- Complete annotation for ≥2 modules
- High agreement with gold labels
- Understanding of edge cases

### Control Corpus Curation

Phase 2 expands control corpora.

**Contributions:**
- Acquire and prepare texts
- Document provenance
- Create instrument specs

---

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** with descriptive name (e.g., `add-corpus-spec-revivalist-sermons`)
3. **Make changes** with clear commit messages
4. **Test** your changes where applicable
5. **Open PR** with:
   - Clear description of changes
   - Rationale referencing METHODOLOGY.md where relevant
   - Any relevant discussion

### PR Review Criteria

- Consistency with methodology principles
- No introduction of bias
- Clear documentation
- Reproducibility

---

## Code of Conduct

### Core Principles

1. **Intellectual honesty above all.** We value being correct over being comfortable.

2. **Steelman, don't strawman.** Represent opposing views as their best proponents would.

3. **Evidence over assertion.** Support claims with specific textual citations.

4. **Humility about limitations.** Acknowledge what we cannot determine.

5. **Respectful disagreement.** Critique arguments, not people.

6. **No predetermined conclusions.** Do not contribute with intent to bias results.

### Unacceptable Behavior

- Ad hominem attacks
- Dismissing perspectives without engagement
- Misrepresenting others' positions
- Introducing deliberately biased prompts
- Editing model outputs to alter conclusions
- Claiming certainty the methodology doesn't support

### Enforcement

Violations will result in comment removal, PR rejection, or contributor ban depending on severity.

---

## Governance

### Current (Phase 1)

Single maintainer with full editorial control. Contributions are reviewed and integrated at maintainer discretion.

### Future (Phase 2+)

If the project grows, governance will be established to:
- Prevent capture by partisan interests (apologetic or critical)
- Ensure methodological consistency
- Manage contributor roles
- Handle disputes

Governance guidelines will be developed transparently before Phase 2 begins.

---

## Issue Labels

| Label | Use For |
|-------|---------|
| `methodology` | Methodology critique or improvement |
| `control-corpus` | Control corpus suggestions |
| `prompt` | Prompt improvements |
| `bug` | Technical issues |
| `documentation` | Documentation improvements |
| `question` | General questions |
| `annotator-interest` | Interest in Phase 2 annotation |
| `help-wanted` | Issues where help is especially needed |

---

## Recognition

All contributors will be acknowledged in CONTRIBUTORS.md. Major contributions will be noted in relevant documents.

Types of recognition:
- **Methodology contributors:** Listed in METHODOLOGY.md revision history
- **Annotators (Phase 2):** Listed in dataset documentation
- **Code contributors:** Listed in CONTRIBUTORS.md
- **Corpus contributors:** Listed in corpus documentation

---

## Questions?

Open an issue with the `question` label, or see the existing issues for common questions.
