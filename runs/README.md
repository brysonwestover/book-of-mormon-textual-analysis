# Model Runs

This directory stores outputs from analysis runs, organized by model and date.

## Directory Naming Convention

```
runs/
├── {model-name}-{date}/
│   ├── metadata.json
│   ├── voice/
│   ├── theology/
│   ├── linguistics/
│   ├── narrative/
│   └── consistency/
```

Example: `claude-opus-4-2025-01-29/`

## Metadata File Format

Each run directory must contain a `metadata.json` file:

```json
{
  "model": {
    "name": "claude-opus-4",
    "version": "claude-opus-4-5-20251101",
    "provider": "Anthropic"
  },
  "run": {
    "date": "2025-01-29",
    "runner": "username",
    "purpose": "Initial analysis run"
  },
  "parameters": {
    "temperature": 0,
    "max_tokens": 4096,
    "system_prompt_version": "1.0"
  },
  "prompts_used": [
    "system/base_analytical_stance.md",
    "voice/01_author_identification.md"
  ],
  "notes": "Any relevant notes about this run"
}
```

## Output File Format

Each analysis output should include:

1. **Header section**
   - Prompt used (with version)
   - Input text/passages analyzed
   - Timestamp

2. **Raw output**
   - Complete model response

3. **Extracted findings**
   - Structured summary of key findings
   - Confidence assessments
   - Hypothesis implications

## Reproducibility Requirements

For a run to be considered valid:

- [ ] All prompts used are documented and versioned
- [ ] Model and parameters are fully specified
- [ ] Input texts are clearly identified
- [ ] Raw outputs are preserved without modification
- [ ] Metadata file is complete and accurate

## Contributing Runs

To contribute a run from a different model:

1. Create a directory following the naming convention
2. Include complete metadata
3. Follow the output file format
4. Submit via pull request with description of the run
