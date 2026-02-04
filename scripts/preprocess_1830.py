#!/usr/bin/env python3
"""
Preprocess the 1830 Book of Mormon text from Archive.org (Jenson replica).

This script:
1. Removes modern preface (lines 1-43)
2. Removes garbled Table of Contents (lines 44-350)
3. Keeps authentic 1830 content starting from title page
4. Identifies and optionally removes/marks page headers
5. Applies known OCR corrections with full audit trail
6. Preserves page markers for traceability

All transformations are logged for reproducibility.
"""

import re
import hashlib
import json
import sys
import platform
import locale
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional

# Configuration
INPUT_FILE = Path(__file__).parent.parent / "data/text/book-of-mormon-1830-replica.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "data/text/processed"
LOG_FILE = OUTPUT_DIR / "preprocessing_log.json"

# Line ranges to remove (0-indexed internally, 1-indexed in documentation)
MODERN_PREFACE_END = 43  # Lines 1-43 (Jenson's modern preface)
TOC_END = 350  # Lines 44-350 (garbled table of contents)
CONTENT_START = 351  # Line 351 (actual 1830 title page begins)

# Known OCR corrections with evidence
# Format: (wrong, correct, context_hint, line_number_approx)
KNOWN_CORRECTIONS = [
    ("Neput", "Nephi", "1 Nephi 1:1 opening", 530),
    ("day 5", "days", "1 Nephi 1:1 'in the course of my days'", 532),
    ("utttered", "uttered", "double t error", None),
]

# Page header patterns (to mark or remove)
# These appear as "PAGE_NUM BOOK_NAME" or "BOOK_NAME PAGE_NUM"
PAGE_HEADER_PATTERNS = [
    r'^\d+\s+(?:FIRST |SECOND |THIRD |FOURTH )?BOOK OF (?:NEPHI|MORMON|ALMA|MOSIAH|JACOB|ENOS|JAROM|OMNI|HELAMAN|ETHER|MORONI)\.*\s*$',
    r'^(?:FIRST |SECOND |THIRD |FOURTH )?BOOK OF (?:NEPHI|MORMON|ALMA|MOSIAH|JACOB|ENOS|JAROM|OMNI|HELAMAN|ETHER|MORONI)\.*\s+\d+\s*$',
    r'^\d+\s+WORDS OF MORMON\.*\s*$',
    r'^WORDS OF MORMON\.*\s+\d+\s*$',
]

# OCR-corrupted page headers that don't match numeric patterns
# These have letters substituted for digits (3→B, 4→A, 7→T)
OCR_CORRUPTED_HEADERS = [
    "BOOK OF ALMA. B47",      # Should be 347
    "BOOK OF HELAMAN. 4A7",   # Should be 447
    "47A BOOK OF NEPHI.",     # Should be 474
    "BOOK OF ETHER. 5AT",     # Should be 547
]


SCRIPT_VERSION = "1.2.0"  # Added OCR-corrupted header overrides


def get_git_commit_hash() -> Optional[str]:
    """Get the current git commit hash of this script."""
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
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "locale": locale.getlocale(),
        "encoding": sys.getdefaultencoding(),
        "script_path": str(Path(__file__).resolve()),
        "git_commit": get_git_commit_hash(),
    }


@dataclass
class PreprocessingStats:
    """Statistics from preprocessing run."""
    input_file: str
    input_hash_sha256: str
    input_lines: int
    input_words: int
    output_file: str
    output_hash_sha256: str
    output_lines: int
    output_words: int
    lines_removed_preface: int
    lines_removed_toc: int
    page_headers_found: int
    page_headers_matched: list
    corrections_applied: int
    corrections_detail: list
    timestamp: str
    script_version: str = SCRIPT_VERSION


def compute_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def is_page_header(line: str) -> bool:
    """Check if line is a page header."""
    line_stripped = line.strip()
    # Check regex patterns
    for pattern in PAGE_HEADER_PATTERNS:
        if re.match(pattern, line_stripped, re.IGNORECASE):
            return True
    # Check OCR-corrupted headers (exact match)
    if line_stripped in OCR_CORRUPTED_HEADERS:
        return True
    return False


def apply_corrections(text: str, corrections: list) -> tuple[str, list]:
    """Apply known OCR corrections and return corrected text + log."""
    applied = []
    for wrong, correct, context, line_approx in corrections:
        count = text.count(wrong)
        if count > 0:
            text = text.replace(wrong, correct)
            applied.append({
                "original": wrong,
                "corrected": correct,
                "context": context,
                "occurrences": count,
                "approximate_line": line_approx
            })
    return text, applied


def preprocess(
    input_path: Path,
    output_path: Path,
    remove_page_headers: bool = False,
    mark_page_headers: bool = True,
    apply_ocr_corrections: bool = True,
) -> PreprocessingStats:
    """
    Preprocess the 1830 text.

    Args:
        input_path: Path to raw input file
        output_path: Path for processed output
        remove_page_headers: If True, remove page headers entirely
        mark_page_headers: If True (and not removing), mark headers with <!-- PAGE_HEADER -->
        apply_ocr_corrections: If True, apply known OCR corrections

    Returns:
        PreprocessingStats with details of transformations
    """
    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    input_hash = compute_hash(input_path)
    input_lines = len(lines)
    input_words = count_words(''.join(lines))

    # Remove modern preface and TOC
    # Keep from CONTENT_START onwards (0-indexed: line 351 = index 350)
    content_lines = lines[CONTENT_START - 1:]

    lines_removed_preface = MODERN_PREFACE_END
    lines_removed_toc = TOC_END - MODERN_PREFACE_END

    # Process page headers
    page_headers_found = 0
    page_headers_matched = []  # Store examples for audit
    processed_lines = []

    for i, line in enumerate(content_lines):
        if is_page_header(line):
            page_headers_found += 1
            # Store first 10 and last 5 examples for audit trail
            if len(page_headers_matched) < 10 or page_headers_found > 560:
                page_headers_matched.append({
                    "line_in_content": i + 1,
                    "line_in_original": i + CONTENT_START,
                    "text": line.strip()
                })
            if remove_page_headers:
                continue  # Skip this line
            elif mark_page_headers:
                # Mark it but keep it
                processed_lines.append(f"<!-- PAGE_HEADER: {line.strip()} -->\n")
                continue
        processed_lines.append(line)

    # Join text for corrections
    text = ''.join(processed_lines)

    # Apply OCR corrections
    corrections_applied = []
    if apply_ocr_corrections:
        text, corrections_applied = apply_corrections(text, KNOWN_CORRECTIONS)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    output_lines = text.count('\n') + 1
    output_words = count_words(text)
    output_hash = compute_hash(output_path)

    return PreprocessingStats(
        input_file=str(input_path),
        input_hash_sha256=input_hash,
        input_lines=input_lines,
        input_words=input_words,
        output_file=str(output_path),
        output_hash_sha256=output_hash,
        output_lines=output_lines,
        output_words=output_words,
        lines_removed_preface=lines_removed_preface,
        lines_removed_toc=lines_removed_toc,
        page_headers_found=page_headers_found,
        page_headers_matched=page_headers_matched,
        corrections_applied=len(corrections_applied),
        corrections_detail=corrections_applied,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def save_removed_blocks(input_path: Path, output_dir: Path) -> dict:
    """Save removed blocks (preface, TOC) as separate artifacts for audit."""
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Save modern preface (lines 1-43)
    preface_path = output_dir / "removed-modern-preface.txt"
    preface_content = ''.join(lines[:MODERN_PREFACE_END])
    with open(preface_path, 'w', encoding='utf-8') as f:
        f.write(preface_content)

    # Save garbled TOC (lines 44-350)
    toc_path = output_dir / "removed-garbled-toc.txt"
    toc_content = ''.join(lines[MODERN_PREFACE_END:TOC_END])
    with open(toc_path, 'w', encoding='utf-8') as f:
        f.write(toc_content)

    return {
        "preface": {
            "file": str(preface_path),
            "lines": f"1-{MODERN_PREFACE_END}",
            "hash_sha256": compute_hash(preface_path),
        },
        "toc": {
            "file": str(toc_path),
            "lines": f"{MODERN_PREFACE_END + 1}-{TOC_END}",
            "hash_sha256": compute_hash(toc_path),
        }
    }


def main():
    """Run preprocessing with default settings."""
    print("=" * 60)
    print("1830 Book of Mormon Text Preprocessing")
    print(f"Script Version: {SCRIPT_VERSION}")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return 1

    # Capture environment
    env_info = get_environment_info()
    print(f"\nEnvironment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Git commit: {env_info['git_commit'] or 'not available'}")

    print(f"\nInput file: {INPUT_FILE}")
    print(f"Input hash: {compute_hash(INPUT_FILE)}")

    # Create multiple output versions
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Version 1: Headers marked (for traceability)
    output_marked = OUTPUT_DIR / "bom-1830-clean-headers-marked.txt"
    stats_marked = preprocess(
        INPUT_FILE,
        output_marked,
        remove_page_headers=False,
        mark_page_headers=True,
        apply_ocr_corrections=True,
    )
    print(f"\nCreated: {output_marked}")
    print(f"  Lines: {stats_marked.output_lines}, Words: {stats_marked.output_words}")
    print(f"  Hash: {stats_marked.output_hash_sha256}")

    # Version 2: Headers removed (for analysis)
    output_clean = OUTPUT_DIR / "bom-1830-clean.txt"
    stats_clean = preprocess(
        INPUT_FILE,
        output_clean,
        remove_page_headers=True,
        mark_page_headers=False,
        apply_ocr_corrections=True,
    )
    print(f"\nCreated: {output_clean}")
    print(f"  Lines: {stats_clean.output_lines}, Words: {stats_clean.output_words}")
    print(f"  Hash: {stats_clean.output_hash_sha256}")

    # Version 3: No corrections (for comparison/validation)
    output_nocorrect = OUTPUT_DIR / "bom-1830-clean-no-corrections.txt"
    stats_nocorrect = preprocess(
        INPUT_FILE,
        output_nocorrect,
        remove_page_headers=True,
        mark_page_headers=False,
        apply_ocr_corrections=False,
    )
    print(f"\nCreated: {output_nocorrect}")
    print(f"  Lines: {stats_nocorrect.output_lines}, Words: {stats_nocorrect.output_words}")
    print(f"  Hash: {stats_nocorrect.output_hash_sha256}")

    # Save removed blocks as separate artifacts
    print("\nSaving removed blocks...")
    removed_blocks = save_removed_blocks(INPUT_FILE, OUTPUT_DIR)
    print(f"  Saved: {removed_blocks['preface']['file']}")
    print(f"  Saved: {removed_blocks['toc']['file']}")

    # Save detailed log
    log_data = {
        "preprocessing_runs": [
            {"output_file": str(output_marked), "stats": asdict(stats_marked)},
            {"output_file": str(output_clean), "stats": asdict(stats_clean)},
            {"output_file": str(output_nocorrect), "stats": asdict(stats_nocorrect)},
        ],
        "removed_blocks": removed_blocks,
        "configuration": {
            "modern_preface_end_line": MODERN_PREFACE_END,
            "toc_end_line": TOC_END,
            "content_start_line": CONTENT_START,
            "known_corrections": [
                {"wrong": w, "correct": c, "context": ctx}
                for w, c, ctx, _ in KNOWN_CORRECTIONS
            ],
            "page_header_patterns": PAGE_HEADER_PATTERNS,
        },
        "environment": env_info,
        "script_version": SCRIPT_VERSION,
    }

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

    # Compute and display log hash
    log_hash = compute_hash(LOG_FILE)
    print(f"\nLog saved: {LOG_FILE}")
    print(f"Log hash: {log_hash}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input:  {stats_marked.input_lines:,} lines, {stats_marked.input_words:,} words")
    print(f"Output: {stats_clean.output_lines:,} lines, {stats_clean.output_words:,} words")
    print(f"Removed: {stats_marked.lines_removed_preface} lines (modern preface)")
    print(f"         {stats_marked.lines_removed_toc} lines (garbled TOC)")
    print(f"Page headers found: {stats_marked.page_headers_found}")
    print(f"OCR corrections applied: {stats_marked.corrections_applied}")

    if stats_marked.corrections_detail:
        print("\nCorrections detail:")
        for c in stats_marked.corrections_detail:
            print(f"  '{c['original']}' -> '{c['corrected']}' ({c['occurrences']}x) - {c['context']}")

    print("\n" + "=" * 60)
    print("Preprocessing complete. Files ready for analysis.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
