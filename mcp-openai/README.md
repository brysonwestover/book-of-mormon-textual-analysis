# MCP OpenAI Server

Enables Claude to send prompts to OpenAI models for multi-model analysis.

## Setup

### 1. Install uv (if not installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies
```bash
cd mcp-openai
uv pip install -e .
```

### 3. Set your OpenAI API key
```bash
export OPENAI_API_KEY="your-key-here"
```

### 4. Add to Claude Code

Copy the example configuration file and add your API key:
```bash
cp .mcp.json.example .mcp.json
# Edit .mcp.json and replace "your-openai-api-key-here" with your actual key
```

Or use the CLI:
```bash
claude mcp add openai -s project -- uv run --directory /path/to/book-of-mormon-textual-analysis/mcp-openai python server.py
```

Or manually add to `.mcp.json`:
```json
{
  "mcpServers": {
    "openai": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/book-of-mormon-textual-analysis/mcp-openai", "python", "server.py"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

## Available Tools

### `ask_gpt`
Send any prompt to a GPT model.

Parameters:
- `prompt` (required): The user message to send
- `system_prompt`: System instructions (default: "You are a helpful assistant.")
- `model`: Model to use (default: `gpt-5.2-pro`)
- `temperature`: Sampling temperature 0-2 (only for Chat Completions API models)
- `max_tokens`: Maximum response length (default: 4096)
- `reasoning_effort`: For GPT-5.2 Pro - none, low, medium, high, xhigh (default: medium)

### `debate_methodology`
Ask GPT to evaluate a methodological debate between two positions.

Parameters:
- `topic` (required): The methodological question being debated
- `position_a` (required): First position/argument
- `position_b` (required): Second position/argument
- `model`: Model to use (default: `gpt-5.2-pro`)
- `reasoning_effort`: For GPT-5.2 Pro (default: high)

## Models Available

### Responses API Models (GPT-5.2 Pro)
- `gpt-5.2-pro` (default) - Research-grade model with adjustable reasoning effort
- `gpt-5.2` - Standard GPT-5.2
- `gpt-5-pro` - GPT-5 Pro variant
- `gpt-5` - Standard GPT-5

### Chat Completions API Models
- `gpt-4o` - Fast, capable model
- `gpt-4-turbo` - Extended context
- `gpt-4o-mini` - Lightweight option
- `o1-preview` - Reasoning model
- `o1-mini` - Lightweight reasoning

## Reasoning Effort (GPT-5.2 Pro only)

The `reasoning_effort` parameter controls how much computation GPT-5.2 Pro uses:
- `none` - No extended reasoning
- `low` - Light reasoning
- `medium` - Balanced (default for ask_gpt)
- `high` - Deep analysis (default for debate_methodology)
- `xhigh` - Maximum reasoning depth
