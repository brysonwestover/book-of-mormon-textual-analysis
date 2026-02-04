#!/usr/bin/env python3
"""
Run analysis prompts against LLM APIs.

This script loads prompts from the prompts/ directory and runs them against
configured LLM APIs, saving results to the runs/ directory.

Usage:
    python run_analysis.py --module voice --prompt 01_author_identification
    python run_analysis.py --module theology --all
    python run_analysis.py --list-prompts
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# API clients (install as needed)
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# Configuration
REPO_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
RUNS_DIR = REPO_ROOT / "runs"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system" / "base_analytical_stance.md"


def load_prompt(module: str, prompt_name: str) -> dict:
    """Load a prompt file and extract the prompt text."""
    prompt_path = PROMPTS_DIR / module / f"{prompt_name}.md"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    content = prompt_path.read_text()

    # Extract prompt text from markdown code block
    prompt_match = re.search(r"## Prompt Text\s*```\s*(.*?)```", content, re.DOTALL)
    if not prompt_match:
        raise ValueError(f"Could not extract prompt text from {prompt_path}")

    prompt_text = prompt_match.group(1).strip()

    # Extract metadata
    version_match = re.search(r"## Version\s*(.+)", content)
    purpose_match = re.search(r"## Purpose\s*(.+)", content)

    return {
        "path": str(prompt_path),
        "module": module,
        "name": prompt_name,
        "version": version_match.group(1).strip() if version_match else "unknown",
        "purpose": purpose_match.group(1).strip() if purpose_match else "",
        "prompt_text": prompt_text,
    }


def load_system_prompt() -> str:
    """Load the base system prompt."""
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"System prompt not found: {SYSTEM_PROMPT_PATH}")

    content = SYSTEM_PROMPT_PATH.read_text()

    # Extract prompt text from markdown code block
    prompt_match = re.search(r"## Prompt Text\s*```\s*(.*?)```", content, re.DOTALL)
    if not prompt_match:
        raise ValueError("Could not extract system prompt text")

    return prompt_match.group(1).strip()


def list_prompts() -> dict[str, list[str]]:
    """List all available prompts by module."""
    prompts = {}

    for module_dir in PROMPTS_DIR.iterdir():
        if module_dir.is_dir() and module_dir.name != "system":
            module_prompts = []
            for prompt_file in module_dir.glob("*.md"):
                if prompt_file.name != "README.md":
                    module_prompts.append(prompt_file.stem)
            if module_prompts:
                prompts[module_dir.name] = sorted(module_prompts)

    return prompts


def run_with_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-opus-4-5-20251101",
    max_tokens: int = 4096,
) -> dict:
    """Run analysis using Anthropic API."""
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return {
        "model": model,
        "provider": "anthropic",
        "response": response.content[0].text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


def run_with_openai(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4-turbo-preview",
    max_tokens: int = 4096,
) -> dict:
    """Run analysis using OpenAI API."""
    if not HAS_OPENAI:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "model": model,
        "provider": "openai",
        "response": response.choices[0].message.content,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    }


def save_run(
    prompt_info: dict,
    result: dict,
    input_text: Optional[str],
    run_dir: Path,
) -> Path:
    """Save run output to the runs directory."""
    # Create module subdirectory
    module_dir = run_dir / prompt_info["module"]
    module_dir.mkdir(parents=True, exist_ok=True)

    # Create output file
    timestamp = datetime.now().isoformat()
    output_file = module_dir / f"{prompt_info['name']}_{timestamp.replace(':', '-')}.json"

    output = {
        "metadata": {
            "timestamp": timestamp,
            "prompt": prompt_info,
            "model": result["model"],
            "provider": result["provider"],
            "usage": result["usage"],
        },
        "input_text": input_text,
        "raw_response": result["response"],
    }

    output_file.write_text(json.dumps(output, indent=2))

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Run Book of Mormon textual analysis prompts"
    )

    parser.add_argument(
        "--module",
        choices=["voice", "theology", "linguistics", "narrative", "consistency"],
        help="Analysis module to run",
    )
    parser.add_argument(
        "--prompt",
        help="Specific prompt to run (without .md extension)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all prompts in the specified module",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List all available prompts",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        help="Specific model to use (defaults to provider's default)",
    )
    parser.add_argument(
        "--input-text",
        help="Text to analyze (can also read from stdin)",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="File containing text to analyze",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without making API calls",
    )

    args = parser.parse_args()

    # Handle --list-prompts
    if args.list_prompts:
        prompts = list_prompts()
        print("Available prompts by module:\n")
        for module, prompt_list in sorted(prompts.items()):
            print(f"  {module}/")
            for prompt in prompt_list:
                print(f"    - {prompt}")
        return

    # Validate arguments
    if not args.module:
        parser.error("--module is required unless using --list-prompts")

    if not args.prompt and not args.all:
        parser.error("Either --prompt or --all is required")

    # Load input text
    input_text = args.input_text
    if args.input_file:
        input_text = args.input_file.read_text()
    elif not sys.stdin.isatty():
        input_text = sys.stdin.read()

    # Load system prompt
    try:
        system_prompt = load_system_prompt()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading system prompt: {e}", file=sys.stderr)
        return 1

    # Determine which prompts to run
    if args.all:
        prompts_to_run = list_prompts().get(args.module, [])
    else:
        prompts_to_run = [args.prompt]

    if not prompts_to_run:
        print(f"No prompts found for module: {args.module}", file=sys.stderr)
        return 1

    # Create run directory
    model_name = args.model or ("claude-opus-4" if args.provider == "anthropic" else "gpt-4")
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_dir = RUNS_DIR / f"{model_name}-{date_str}"

    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    # Run prompts
    for prompt_name in prompts_to_run:
        try:
            prompt_info = load_prompt(args.module, prompt_name)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading prompt {prompt_name}: {e}", file=sys.stderr)
            continue

        # Substitute input text into prompt if provided
        user_prompt = prompt_info["prompt_text"]
        if input_text:
            user_prompt = user_prompt.replace("[INSERT PASSAGE]", input_text)

        print(f"Running: {args.module}/{prompt_name}")

        if args.dry_run:
            print(f"  System prompt: {len(system_prompt)} chars")
            print(f"  User prompt: {len(user_prompt)} chars")
            print(f"  Provider: {args.provider}")
            print(f"  Output would be saved to: {run_dir}")
            continue

        # Run the analysis
        try:
            if args.provider == "anthropic":
                result = run_with_anthropic(
                    system_prompt,
                    user_prompt,
                    model=args.model or "claude-opus-4-5-20251101",
                )
            else:
                result = run_with_openai(
                    system_prompt,
                    user_prompt,
                    model=args.model or "gpt-4-turbo-preview",
                )

            # Save results
            output_file = save_run(prompt_info, result, input_text, run_dir)
            print(f"  Saved to: {output_file}")
            print(f"  Tokens: {result['usage']}")

        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            continue

    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    sys.exit(main() or 0)
