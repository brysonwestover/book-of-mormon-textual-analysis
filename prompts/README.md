# Prompts

This directory contains all prompts used for analysis, organized by module.

## Directory Structure

```
prompts/
├── system/         # System prompts establishing analytical stance
├── voice/          # Voice variation analysis prompts
├── theology/       # Theological anachronism prompts
├── linguistics/    # Linguistic marker prompts
├── narrative/      # Narrative structure prompts
└── consistency/    # Internal consistency prompts
```

## Prompt Design Principles

All prompts in this repository must:

1. **State questions neutrally** — No leading language
2. **Consider both hypotheses** — Explicitly request analysis under both ancient-origin and 19th-century-origin assumptions
3. **Request evidence** — Require citation of specific textual passages
4. **Request confidence levels** — Ask for uncertainty quantification
5. **Acknowledge ambiguity** — Request identification of ambiguous findings

## Prompt File Format

Each prompt file should include:

```markdown
# [Prompt Name]

## Purpose
[What this prompt is designed to analyze]

## Module
[Which analysis module this belongs to]

## Version
[Version number and date]

## Prompt Text
[The actual prompt]

## Expected Output Format
[How results should be structured]

## Notes
[Any special considerations]
```

## Version Control

All prompts are version-controlled. When modifying a prompt:
1. Update the version number
2. Document changes in commit message
3. Consider keeping old versions for comparison

## Review Process

New or modified prompts should be reviewed for:
- Balance between hypotheses
- Absence of leading language
- Clarity of instructions
- Appropriate scope
