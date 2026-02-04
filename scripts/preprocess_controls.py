#!/usr/bin/env python3
"""
Preprocess control corpora for stylometric calibration.

Applies identical preprocessing to control texts as applied to Book of Mormon:
- Remove headers/footers (Gutenberg, CCEL, library stamps)
- Unicode NFC normalization
- Standardize quotes and dashes
- Segment into 1000-word blocks
- Create manifest with hashes
"""

import re
import hashlib
import json
import unicodedata
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional

SCRIPT_VERSION = "1.0.0"

BASE_DIR = Path(__file__).parent.parent
REFERENCE_DIR = BASE_DIR / "data/reference"
OUTPUT_DIR = REFERENCE_DIR / "processed"

# Unicode replacements (same as normalize_unicode.py)
REPLACEMENTS = {
    '\u2018': "'", '\u2019': "'",  # Curly single quotes
    '\u201C': '"', '\u201D': '"',  # Curly double quotes
    '\u2013': '-', '\u2014': '-', '\u2015': '-',  # Dashes
    '\u00A0': ' ', '\u2003': ' ', '\u2002': ' ',  # Spaces
}


def compute_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def normalize_text(text: str) -> str:
    """Apply Unicode NFC normalization and character replacements."""
    text = unicodedata.normalize('NFC', text)
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def clean_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    # Find start marker
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT!",
    ]
    for marker in start_markers:
        if marker in text:
            idx = text.find(marker)
            text = text[idx + len(marker):]
            # Skip past the marker line
            text = text.split('\n', 1)[-1] if '\n' in text else text
            break

    # Find end marker
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg's",
    ]
    for marker in end_markers:
        if marker in text:
            idx = text.find(marker)
            text = text[:idx]
            break

    return text.strip()


def clean_ccel(text: str) -> str:
    """Remove CCEL header metadata."""
    lines = text.split('\n')
    content_start = 0

    # Skip CCEL metadata block (starts with underscores)
    in_header = True
    for i, line in enumerate(lines):
        if in_header:
            if line.strip().startswith('___'):
                continue
            elif 'Title:' in line or 'Creator' in line or 'Rights:' in line or 'CCEL Subjects:' in line:
                continue
            elif line.strip() == '':
                continue
            else:
                # Check if we've passed the header
                if i > 10:  # Header should be in first ~20 lines
                    content_start = i
                    in_header = False

    return '\n'.join(lines[content_start:]).strip()


def clean_late_war(text: str) -> str:
    """Clean Late War OCR artifacts."""
    lines = text.split('\n')
    cleaned_lines = []

    # Skip library stamps and title page (first ~50 lines typically)
    content_started = False
    for i, line in enumerate(lines):
        # Look for "CHAP. I" or "CHAPTER I" as content start
        if not content_started:
            if re.match(r'^\s*CHAP\.?\s*I\.?\s*$', line, re.IGNORECASE):
                content_started = True
                cleaned_lines.append(line)
            continue

        # Skip page numbers and headers
        if re.match(r'^\s*\d+\s*$', line.strip()):
            continue
        if re.match(r'^\s*THE\s+LATE\s+WAR\.?\s*$', line, re.IGNORECASE):
            continue
        if re.match(r'^\s*LATE\s+WAR\.?\s*\d*\s*$', line, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Fix common OCR errors
    ocr_fixes = [
        (r'\bi\s+', 'I '),  # Lowercase I at word boundary
        (r'\s+,', ','),  # Space before comma
        (r'\s+\.', '.'),  # Space before period
        (r'\s+;', ';'),  # Space before semicolon
    ]
    for pattern, replacement in ocr_fixes:
        text = re.sub(pattern, replacement, text)

    return text.strip()


def segment_text(text: str, segment_size: int = 1000) -> list:
    """Segment text into approximately equal word blocks."""
    words = text.split()
    segments = []

    for i in range(0, len(words), segment_size):
        segment_words = words[i:i + segment_size]
        if len(segment_words) >= segment_size // 2:  # Keep if at least half size
            segments.append({
                "id": len(segments) + 1,
                "start_word": i,
                "end_word": i + len(segment_words),
                "word_count": len(segment_words),
                "text": ' '.join(segment_words)
            })

    return segments


def process_corpus(
    name: str,
    input_path: Path,
    cleaner_func,
    author: str,
    source_type: str,
    description: str,
) -> dict:
    """Process a single control corpus."""
    print(f"\nProcessing {name}...")

    # Read raw
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        raw_text = f.read()

    raw_words = len(raw_text.split())
    raw_hash = compute_text_hash(raw_text)
    print(f"  Raw: {raw_words:,} words")

    # Clean
    cleaned = cleaner_func(raw_text)
    cleaned = normalize_text(cleaned)

    # Remove excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned)

    clean_words = len(cleaned.split())
    clean_hash = compute_text_hash(cleaned)
    print(f"  Cleaned: {clean_words:,} words")

    # Save cleaned text
    output_dir = OUTPUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = output_dir / f"{name}-clean.txt"
    with open(clean_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    # Segment
    segments = segment_text(cleaned, 1000)
    print(f"  Segments: {len(segments)} (1000-word blocks)")

    # Save segments
    segments_path = output_dir / f"{name}-segments.json"
    with open(segments_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2)

    # Create corpus info
    corpus_info = {
        "name": name,
        "author": author,
        "source_type": source_type,
        "description": description,
        "input_file": str(input_path),
        "input_hash": raw_hash,
        "input_words": raw_words,
        "clean_file": str(clean_path),
        "clean_hash": compute_hash(clean_path),
        "clean_words": clean_words,
        "segments_file": str(segments_path),
        "segment_count": len(segments),
        "segment_size": 1000,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save corpus metadata
    meta_path = output_dir / f"{name}-metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(corpus_info, f, indent=2)

    return corpus_info


def extract_kjv_books(kjv_path: Path) -> dict:
    """Extract individual books from KJV Bible for multi-author control."""
    print("\nProcessing KJV Bible...")

    with open(kjv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse book-chapter:verse format
    books = {}
    current_book = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match "Book Chapter:Verse" pattern
        match = re.match(r'^(\d?\s*[A-Za-z]+)\s+\d+:\d+\s+(.+)$', line)
        if match:
            book_name = match.group(1).strip()
            verse_text = match.group(2)

            if book_name not in books:
                books[book_name] = []
            books[book_name].append(verse_text)

    # Select representative books for multi-author comparison
    # Traditional authorship attributions (for calibration purposes)
    selected_books = {
        "Genesis": {"author": "Moses (traditional)", "words": 0, "text": ""},
        "Psalms": {"author": "David/Multiple (traditional)", "words": 0, "text": ""},
        "Proverbs": {"author": "Solomon (traditional)", "words": 0, "text": ""},
        "Isaiah": {"author": "Isaiah (traditional)", "words": 0, "text": ""},
        "Matthew": {"author": "Matthew (traditional)", "words": 0, "text": ""},
        "John": {"author": "John (traditional)", "words": 0, "text": ""},
        "Romans": {"author": "Paul (traditional)", "words": 0, "text": ""},
        "Hebrews": {"author": "Unknown/Paul? (traditional)", "words": 0, "text": ""},
    }

    output_dir = OUTPUT_DIR / "kjv"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_segments = []

    for book_name, info in selected_books.items():
        if book_name in books:
            text = ' '.join(books[book_name])
            text = normalize_text(text)
            info["text"] = text
            info["words"] = len(text.split())

            # Save individual book
            book_path = output_dir / f"kjv-{book_name.lower()}-clean.txt"
            with open(book_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # Segment this book
            segments = segment_text(text, 1000)
            for seg in segments:
                seg["book"] = book_name
                seg["author"] = info["author"]
            all_segments.extend(segments)

            print(f"  {book_name}: {info['words']:,} words, {len(segments)} segments")

    # Save combined segments
    segments_path = output_dir / "kjv-segments.json"
    with open(segments_path, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, indent=2)

    # Create metadata
    total_words = sum(info["words"] for info in selected_books.values())
    meta = {
        "name": "kjv",
        "source_type": "multi-author",
        "description": "KJV Bible - selected books with different traditional authors",
        "books": {k: {"author": v["author"], "words": v["words"]}
                  for k, v in selected_books.items()},
        "total_words": total_words,
        "segment_count": len(all_segments),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = output_dir / "kjv-metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"  Total: {total_words:,} words, {len(all_segments)} segments")

    return meta


def main():
    print("=" * 60)
    print("Control Corpora Preprocessing")
    print(f"Script Version: {SCRIPT_VERSION}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. Process Finney
    finney_path = REFERENCE_DIR / "finney/finney-revivals-raw.txt"
    if finney_path.exists():
        info = process_corpus(
            name="finney",
            input_path=finney_path,
            cleaner_func=clean_ccel,
            author="Charles Finney",
            source_type="single-author",
            description="Lectures on Revivals of Religion (1835) - single 19th-c author baseline",
        )
        results.append(info)
    else:
        print(f"WARNING: {finney_path} not found")

    # 2. Process Late War
    late_war_path = REFERENCE_DIR / "late-war/late-war-raw.txt"
    if late_war_path.exists():
        info = process_corpus(
            name="late-war",
            input_path=late_war_path,
            cleaner_func=clean_late_war,
            author="Gilbert Hunt",
            source_type="pseudo-archaic",
            description="The Late War (1816) - KJV-style pseudo-biblical narrative",
        )
        results.append(info)
    else:
        print(f"WARNING: {late_war_path} not found")

    # 3. Process Josephus
    josephus_path = REFERENCE_DIR / "josephus/josephus-antiquities-raw.txt"
    if josephus_path.exists():
        info = process_corpus(
            name="josephus",
            input_path=josephus_path,
            cleaner_func=clean_gutenberg,
            author="William Whiston (translator)",
            source_type="single-translator",
            description="Antiquities of the Jews - single translator, multiple source books",
        )
        results.append(info)
    else:
        print(f"WARNING: {josephus_path} not found")

    # 4. Process KJV Bible
    kjv_path = BASE_DIR / "data/text/lds-scriptures-2020.12.08/text/kjv-scriptures.txt"
    if not kjv_path.exists():
        # Try alternate location
        kjv_path = BASE_DIR / "data/reference/kjv/kjv-scriptures.txt"

    if kjv_path.exists():
        kjv_info = extract_kjv_books(kjv_path)
        results.append(kjv_info)
    else:
        print(f"WARNING: KJV not found, will need to acquire separately")

    # Create master manifest
    manifest = {
        "manifest_version": "1.0.0",
        "generated": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "git_commit": get_git_commit(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "corpora": results,
        "summary": {
            "total_corpora": len(results),
            "total_segments": sum(c.get("segment_count", 0) for c in results),
            "by_type": {},
        }
    }

    # Summarize by type
    for corpus in results:
        stype = corpus.get("source_type", "unknown")
        if stype not in manifest["summary"]["by_type"]:
            manifest["summary"]["by_type"][stype] = {"count": 0, "segments": 0}
        manifest["summary"]["by_type"][stype]["count"] += 1
        manifest["summary"]["by_type"][stype]["segments"] += corpus.get("segment_count", 0)

    manifest_path = OUTPUT_DIR / "control-corpora-manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for corpus in results:
        print(f"  {corpus['name']}: {corpus.get('clean_words', corpus.get('total_words', 'N/A')):,} words, "
              f"{corpus.get('segment_count', 'N/A')} segments ({corpus.get('source_type', 'N/A')})")
    print(f"\nManifest saved: {manifest_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
