# Results

This directory contains synthesized findings and cross-model comparisons.

## Directory Structure

```
results/
├── summaries/      # Synthesized findings from individual runs
└── comparisons/    # Cross-model and cross-run comparisons
```

## Summaries

The `summaries/` directory contains aggregated findings:

- Module-level summaries (e.g., `voice_analysis_summary.md`)
- Cross-module synthesis documents
- Overall findings reports

### Summary Document Format

```markdown
# [Module] Analysis Summary

## Overview
[High-level summary of findings]

## Key Findings

### Finding 1: [Title]
- Evidence: [specific citations]
- Confidence: [level]
- Hypothesis A interpretation: [...]
- Hypothesis B interpretation: [...]

[Additional findings...]

## Patterns Across Runs
[What findings are consistent across different model runs?]

## Areas of Disagreement
[Where do different runs produce different results?]

## Confidence Assessment
[Overall confidence in findings]

## Limitations
[What this analysis cannot determine]

## Recommendations for Further Analysis
[Suggested next steps]
```

## Comparisons

The `comparisons/` directory contains analyses comparing:

- Different model outputs on the same prompts
- Different prompt versions on the same model
- Temporal stability (re-runs of the same analysis)

### Comparison Document Format

```markdown
# Comparison: [Run A] vs. [Run B]

## Methodology
[How the comparison was conducted]

## Areas of Agreement
[Where runs produced similar findings]

## Areas of Disagreement
[Where runs produced different findings]

## Analysis of Disagreements
[Why might these disagreements occur?]

## Implications
[What do agreements/disagreements tell us about reliability?]
```

## Important Notes

- Results are interpretations of model outputs, not ground truth
- Always link findings to specific runs and prompts
- Acknowledge uncertainty appropriately
- Do not overstate conclusions
