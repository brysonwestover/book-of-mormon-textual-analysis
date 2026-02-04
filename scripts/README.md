# Scripts

Utility scripts for running analyses and processing results.

## Available Scripts

### `run_analysis.py`
Runs analysis prompts against configured LLM APIs.

```bash
python scripts/run_analysis.py --module voice --prompt 01_author_identification
```

### `compare_runs.py`
Compares outputs across different runs.

```bash
python scripts/compare_runs.py --run1 claude-opus-4-2025-01-29 --run2 gpt-4-2025-01-29
```

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the repository root:

```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
# Add other providers as needed
```

**Important:** The `.env` file is gitignored and should never be committed.

## Adding New Scripts

When adding scripts:

1. Include docstrings explaining purpose and usage
2. Add command-line argument documentation
3. Handle errors gracefully with informative messages
4. Log all API calls and responses
5. Update this README with usage instructions

## Script Design Principles

- **Reproducibility:** Scripts should produce identical outputs given identical inputs
- **Logging:** All API interactions should be logged
- **Metadata:** Output files should include full metadata about the run
- **Error handling:** Failures should be clear and informative
- **Modularity:** Scripts should be composable for different workflows
