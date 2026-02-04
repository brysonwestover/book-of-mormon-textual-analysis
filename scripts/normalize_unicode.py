#!/usr/bin/env python3
"""
Apply Unicode normalization to the Book of Mormon text.

Creates a normalized version using NFC (Canonical Decomposition, followed by
Canonical Composition) which is the standard for text analysis.

Also standardizes:
- Curly quotes → straight quotes
- Em/en dashes → hyphens (for consistency)
- Non-breaking spaces → regular spaces
"""

import unicodedata
import hashlib
import json
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

SCRIPT_VERSION = "1.0.0"

INPUT_FILE = Path(__file__).parent.parent / "data/text/processed/bom-1830-dehyphenated.txt"
OUTPUT_FILE = Path(__file__).parent.parent / "data/text/processed/bom-1830-normalized.txt"
LOG_FILE = Path(__file__).parent.parent / "data/text/processed/normalization_log.json"

# Character replacements for standardization
REPLACEMENTS = {
    # Curly quotes → straight
    '\u2018': "'",  # Left single quotation mark
    '\u2019': "'",  # Right single quotation mark
    '\u201C': '"',  # Left double quotation mark
    '\u201D': '"',  # Right double quotation mark
    # Dashes → hyphen
    '\u2013': '-',  # En dash
    '\u2014': '-',  # Em dash
    '\u2015': '-',  # Horizontal bar
    # Spaces → regular space
    '\u00A0': ' ',  # Non-breaking space
    '\u2003': ' ',  # Em space
    '\u2002': ' ',  # En space
}


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


def normalize_text(text: str) -> tuple[str, dict]:
    """
    Apply Unicode NFC normalization and character replacements.

    Returns:
        Tuple of (normalized_text, stats_dict)
    """
    stats = {
        "nfc_changes": 0,
        "replacements": {},
    }

    # Count characters that will change with NFC
    nfc_text = unicodedata.normalize('NFC', text)
    if nfc_text != text:
        stats["nfc_changes"] = sum(1 for a, b in zip(text, nfc_text) if a != b)

    # Apply character replacements
    result = nfc_text
    for old_char, new_char in REPLACEMENTS.items():
        count = result.count(old_char)
        if count > 0:
            stats["replacements"][f"U+{ord(old_char):04X}"] = {
                "name": unicodedata.name(old_char, "UNKNOWN"),
                "replaced_with": new_char,
                "count": count
            }
            result = result.replace(old_char, new_char)

    return result, stats


def main():
    """Run Unicode normalization."""
    print("=" * 60)
    print("Unicode Normalization")
    print(f"Script Version: {SCRIPT_VERSION}")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return 1

    # Read input
    input_hash = compute_hash(INPUT_FILE)
    print(f"\nInput file: {INPUT_FILE}")
    print(f"Input hash: {input_hash}")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    input_chars = len(text)

    # Apply normalization
    normalized_text, stats = normalize_text(text)

    # Write output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(normalized_text)

    output_hash = compute_hash(OUTPUT_FILE)
    output_chars = len(normalized_text)

    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Output hash: {output_hash}")

    print(f"\nStatistics:")
    print(f"  Input characters: {input_chars:,}")
    print(f"  Output characters: {output_chars:,}")
    print(f"  NFC normalization changes: {stats['nfc_changes']}")
    print(f"  Character replacements: {sum(r['count'] for r in stats['replacements'].values())}")

    if stats['replacements']:
        print("\nReplacements made:")
        for code, info in stats['replacements'].items():
            print(f"  {code} ({info['name']}): {info['count']} occurrences")

    # Save log
    log_data = {
        "input": {
            "file": str(INPUT_FILE),
            "hash_sha256": input_hash,
            "characters": input_chars,
        },
        "output": {
            "file": str(OUTPUT_FILE),
            "hash_sha256": output_hash,
            "characters": output_chars,
        },
        "transformations": {
            "normalization_form": "NFC",
            "nfc_changes": stats["nfc_changes"],
            "character_replacements": stats["replacements"],
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_commit": get_git_commit_hash(),
        },
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

    log_hash = compute_hash(LOG_FILE)
    print(f"\nLog saved: {LOG_FILE}")
    print(f"Log hash: {log_hash}")

    print("\n" + "=" * 60)
    print("Normalization complete.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
