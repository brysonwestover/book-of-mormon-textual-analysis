# GPT-5.2 Pro Consultation Protocol

This document establishes the standard procedure for consulting GPT-5.2 Pro during the Book of Mormon Textual Analysis project.

**Version:** 1.0.0
**Date:** 2026-02-01

---

## Purpose

GPT-5.2 Pro serves as an adversarial collaborator to:
1. Identify methodological gaps and biases
2. Stress-test analytical decisions
3. Provide independent critique before conclusions are finalized
4. Ensure rigor through structured disagreement

**Key Principle:** Multi-model agreement is a **robustness check**, not validation. GPT critique strengthens the work by identifying weaknesses, not by conferring truth.

---

## Consultation Triggers

### Mandatory Consultations

Consult GPT-5.2 Pro **before proceeding** when:

| Trigger | Consultation Type |
|---------|-------------------|
| New methodology or analytical approach | Full methodology review |
| Data acquisition/preprocessing decisions | Data integrity audit |
| Completing an analysis module | Results critique |
| Before publishing/sharing findings | Final adversarial review |
| Significant code changes to analysis pipeline | Pipeline audit |
| Interpretation of ambiguous results | Interpretation challenge |

### Optional Consultations

Consider consulting when:
- Uncertain about a technical decision
- Need calibration on claim strength
- Want to explore alternative interpretations
- Debugging unexpected results

---

## Consultation Types

### 1. Methodology Review

**When:** Before implementing new analytical approaches

**Prompt Template:**
```
Review the following methodology for [MODULE NAME]. Critique for:
1. Operational definition clarity
2. Falsifiability of predictions
3. Potential confounds
4. Missing controls
5. Claim-to-evidence alignment

[METHODOLOGY DETAILS]

Questions:
- What's missing?
- What biases might emerge?
- What would make this credible to skeptical academics?
```

**Model Settings:** `reasoning_effort: high`

### 2. Data Integrity Audit

**When:** After data acquisition or preprocessing

**Prompt Template:**
```
Review the following data acquisition/preprocessing work. Critique for:
1. Source selection defensibility
2. Transformation appropriateness
3. Audit trail completeness
4. Reproducibility
5. Potential artifacts introduced

[DATA DETAILS]

Questions:
- Is the process methodologically sound?
- What gaps exist in the audit trail?
- What limitations should be documented?
```

**Model Settings:** `reasoning_effort: high`

### 3. Results Critique

**When:** After completing analysis, before drawing conclusions

**Prompt Template:**
```
Critique the following analysis results. Challenge:
1. Whether the evidence supports the claims
2. Alternative explanations
3. Effect size interpretation
4. Generalizability
5. What the results do NOT show

[RESULTS AND INTERPRETATION]

Questions:
- What's the strongest objection to these conclusions?
- What additional evidence would strengthen/weaken the claims?
- How should uncertainty be communicated?
```

**Model Settings:** `reasoning_effort: xhigh`

### 4. Pipeline Audit

**When:** After significant code changes

**Prompt Template:**
```
Audit the following analysis pipeline code for:
1. Correctness of implementation
2. Edge cases and failure modes
3. Reproducibility concerns
4. Potential bugs affecting results

[CODE OR CODE SUMMARY]

Questions:
- Are there logic errors?
- What edge cases are unhandled?
- What tests should be added?
```

**Model Settings:** `reasoning_effort: high`

### 5. Interpretation Challenge

**When:** Interpreting ambiguous or unexpected results

**Prompt Template:**
```
Challenge the following interpretation:

Observation: [WHAT WAS FOUND]
Current interpretation: [PROPOSED MEANING]
Context: [RELEVANT BACKGROUND]

Questions:
- What alternative explanations exist?
- What assumptions does this interpretation rely on?
- What would falsify this interpretation?
```

**Model Settings:** `reasoning_effort: high`

---

## Documentation Requirements

### For Every Consultation

Record in `docs/consultations/` with filename `YYYY-MM-DD-[topic].md`:

1. **Date and time**
2. **Consultation type**
3. **Full prompt sent**
4. **Full response received**
5. **Action items identified**
6. **Decisions made in response**
7. **Any disagreements and resolution**

### Consultation Log

Maintain a summary in `docs/consultations/log.md`:

```markdown
| Date | Type | Topic | Key Findings | Actions Taken |
|------|------|-------|--------------|---------------|
| 2026-02-01 | Methodology | Initial framework | 6 major gaps | Revised to v2.0 |
| 2026-02-01 | Data Audit | Preprocessing | 5 critical gaps | Addressing now |
```

---

## Response Handling

### When GPT Identifies Gaps

1. **Acknowledge** - Don't dismiss critique defensively
2. **Categorize** - Critical vs. important vs. nice-to-have
3. **Prioritize** - Address critical gaps before proceeding
4. **Document** - Record gaps even if not immediately addressed
5. **Justify** - If not addressing, document explicit rationale

### When GPT and Claude Disagree

1. Document both positions clearly
2. Identify the crux of disagreement
3. Use `debate_methodology` tool if needed
4. Present both views in final documentation
5. Let evidence and user judgment resolve

### When GPT Endorses

Endorsement means the approach is **defensible given stated constraints**, not that it's optimal or that conclusions are correct. Document:
- What specifically was endorsed
- What caveats accompanied endorsement
- What assumptions underlie the endorsement

---

## Model Configuration

### Standard Settings

| Parameter | Default | High-stakes |
|-----------|---------|-------------|
| model | gpt-5.2-pro | gpt-5.2-pro |
| reasoning_effort | high | xhigh |
| max_tokens | 4000 | 8000 |

### When to Use `xhigh` Reasoning

- Final methodology review before analysis
- Results interpretation for publication
- Resolving substantive disagreements
- Complex methodological debates

---

## Anti-Patterns to Avoid

1. **Confirmation seeking** - Don't frame prompts to get desired answers
2. **Cherry-picking** - Don't ignore inconvenient critiques
3. **Over-reliance** - GPT agreement doesn't validate conclusions
4. **Under-consultation** - Don't skip consultations to save time
5. **Prompt manipulation** - Don't adjust prompts until you get agreement

---

## Integration with Workflow

```
[Decision Point]
    → Is this a consultation trigger?
    → YES → Select consultation type
         → Prepare prompt from template
         → Run consultation
         → Document response
         → Address gaps or document rationale
         → Proceed
    → NO → Proceed
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-01 | Initial protocol |
