#!/usr/bin/env python3
"""
De-hyphenate line-break hyphenation in the 1830 Book of Mormon text.

This script handles end-of-line hyphenation artifacts from the original
typesetting that were preserved in the OCR process.

Pattern detected: "word-\nfragment" where the hyphen is a line-break artifact
Example: "respec-\nting" -> "respecting"

This transformation is logged with full audit trail for reproducibility.
"""

import re
import hashlib
import json
import sys
import platform
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
import subprocess

SCRIPT_VERSION = "1.0.0"

# Input/Output configuration
INPUT_FILE = Path(__file__).parent.parent / "data/text/processed/bom-1830-clean.txt"
OUTPUT_FILE = Path(__file__).parent.parent / "data/text/processed/bom-1830-dehyphenated.txt"
LOG_FILE = Path(__file__).parent.parent / "data/text/processed/dehyphenation_log.json"


def compute_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit_hash() -> Optional[str]:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_environment_info() -> dict:
    """Capture environment information for reproducibility."""
    import locale
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": get_git_commit_hash(),
    }


def dehyphenate_text(text: str) -> tuple[str, list]:
    """
    Remove line-break hyphenation, joining split words.

    Rules:
    1. Pattern: hyphen at end of line followed by newline and word fragment
    2. Only join if the continuation starts with lowercase (indicating word continuation)
    3. Special case: ALL CAPS words (headers) are joined if both parts are caps
    4. Handle blank lines between hyphen and continuation (page breaks)

    Returns:
        Tuple of (processed_text, list of changes made)
    """
    lines = text.split('\n')
    changes = []
    result_lines = []
    skip_until = -1  # Track lines to skip

    i = 0
    while i < len(lines):
        if i < skip_until:
            i += 1
            continue

        line = lines[i]

        # Check if line ends with hyphen (with optional trailing whitespace)
        if re.search(r'-\s*$', line):
            # Get the word fragment before the hyphen
            match = re.search(r'(\S+)-\s*$', line)
            if match:
                prefix = match.group(1)

                # Look ahead for continuation, skipping blank lines
                j = i + 1
                blank_lines_skipped = 0
                while j < len(lines) and lines[j].strip() == '':
                    blank_lines_skipped += 1
                    j += 1

                if j < len(lines):
                    next_content_line = lines[j]

                    # Get the first word of the continuation line
                    next_match = re.match(r'^(\S+)', next_content_line)
                    if next_match:
                        suffix = next_match.group(1)

                        # Determine if this is a line-break hyphenation
                        should_join = False

                        # Case 1: lowercase continuation (most common)
                        if suffix[0].islower():
                            should_join = True

                        # Case 2: ALL CAPS continuation (headers like "MOR-MON")
                        elif prefix.isupper() and suffix.isupper():
                            should_join = True

                        if should_join:
                            # Remove the hyphen and join
                            new_line = re.sub(r'-\s*$', '', line) + suffix

                            # Remove the joined word from the continuation line
                            remaining = next_content_line[len(suffix):].lstrip()

                            # Record the change
                            joined_word = prefix + suffix
                            changes.append({
                                "line": i + 1,
                                "original_prefix": prefix + "-",
                                "original_suffix": suffix,
                                "joined_word": joined_word,
                                "blank_lines_skipped": blank_lines_skipped,
                            })

                            # Add the joined line
                            result_lines.append(new_line)

                            # Skip the blank lines
                            # Don't add them to result (they were page breaks)

                            # Handle the continuation line
                            if remaining:
                                lines[j] = remaining
                                skip_until = j  # Will process remaining line next
                            else:
                                skip_until = j + 1  # Skip the now-empty continuation

                            i += 1
                            continue

        result_lines.append(line)
        i += 1

    return '\n'.join(result_lines), changes


def main():
    """Run de-hyphenation with full audit trail."""
    print("=" * 60)
    print("De-hyphenation of Line-Break Artifacts")
    print(f"Script Version: {SCRIPT_VERSION}")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return 1

    # Capture environment
    env_info = get_environment_info()
    print(f"\nEnvironment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Git commit: {env_info['git_commit'] or 'not available'}")

    # Read input
    input_hash = compute_hash(INPUT_FILE)
    print(f"\nInput file: {INPUT_FILE}")
    print(f"Input hash: {input_hash}")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    input_lines = text.count('\n') + 1
    input_words = len(text.split())

    # Count hyphenated lines before processing
    hyphen_count_before = len(re.findall(r'-\s*$', text, re.MULTILINE))
    print(f"Lines ending with hyphen: {hyphen_count_before}")

    # Apply de-hyphenation
    processed_text, changes = dehyphenate_text(text)

    # Write output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(processed_text)

    output_hash = compute_hash(OUTPUT_FILE)
    output_lines = processed_text.count('\n') + 1
    output_words = len(processed_text.split())

    # Count remaining hyphenated lines
    hyphen_count_after = len(re.findall(r'-\s*$', processed_text, re.MULTILINE))

    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Output hash: {output_hash}")
    print(f"\nStatistics:")
    print(f"  Input lines: {input_lines:,}")
    print(f"  Output lines: {output_lines:,}")
    print(f"  Lines removed (joined): {input_lines - output_lines:,}")
    print(f"  Words joined: {len(changes):,}")
    print(f"  Remaining hyphenated lines: {hyphen_count_after}")

    # Sample of changes for review
    print(f"\nSample changes (first 20):")
    for change in changes[:20]:
        print(f"  Line {change['line']}: {change['original_prefix']}{change['original_suffix']} -> {change['joined_word']}")

    if len(changes) > 20:
        print(f"  ... and {len(changes) - 20} more")

    # Save log
    log_data = {
        "input": {
            "file": str(INPUT_FILE),
            "hash_sha256": input_hash,
            "lines": input_lines,
            "words": input_words,
            "hyphenated_lines": hyphen_count_before,
        },
        "output": {
            "file": str(OUTPUT_FILE),
            "hash_sha256": output_hash,
            "lines": output_lines,
            "words": output_words,
            "hyphenated_lines_remaining": hyphen_count_after,
        },
        "transformations": {
            "words_joined": len(changes),
            "lines_removed": input_lines - output_lines,
            "changes": changes,  # Full list for audit
        },
        "environment": env_info,
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

    log_hash = compute_hash(LOG_FILE)
    print(f"\nLog saved: {LOG_FILE}")
    print(f"Log hash: {log_hash}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"De-hyphenated {len(changes):,} words")
    print(f"Remaining hyphenated lines: {hyphen_count_after} (likely legitimate compounds)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
