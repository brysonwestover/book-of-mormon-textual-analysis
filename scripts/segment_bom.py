#!/usr/bin/env python3
"""
Segment Book of Mormon text and pre-annotate with claimed narrators.

Creates 1000-word segments with narrator labels based on book boundaries.
Outputs JSON for review and potential manual correction.
"""

import json
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_VERSION = "1.0.0"

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data/text/processed/bom-1830-normalized.txt"
OUTPUT_DIR = BASE_DIR / "data/text/processed"

# Book boundaries (line numbers from grep) and their primary narrators
# Based on internal claims in the text
BOOK_BOUNDARIES = [
    # (start_line, book_name, primary_narrator, narrator_type)
    (154, "1 Nephi", "Nephi", "first-person"),
    (2875, "2 Nephi", "Nephi", "first-person"),
    (5952, "Jacob", "Jacob", "first-person"),
    (6947, "Enos", "Enos", "first-person"),
    (7072, "Jarom", "Jarom", "first-person"),
    (7154, "Omni", "Multiple", "first-person"),  # Omni, Amaron, Chemish, Abinadom, Amaleki
    (7317, "Words of Mormon", "Mormon", "first-person"),
    (7421, "Mosiah", "Mormon", "third-person-abridgment"),
    (10692, "Alma", "Mormon", "third-person-abridgment"),
    (19662, "Helaman", "Mormon", "third-person-abridgment"),
    (21774, "3 Nephi", "Mormon", "third-person-abridgment"),
    (24749, "4 Nephi", "Mormon", "third-person-abridgment"),
    (24967, "Mormon", "Mormon", "mixed"),  # Ch 1-7 Mormon, Ch 8-9 Moroni
    (25926, "Ether", "Moroni", "third-person-abridgment"),
    (27590, "Moroni", "Moroni", "first-person"),
]

# Paratext exclusion (from paratext-policy.md)
NARRATIVE_START_LINE = 180  # After title page, copyright, preface
NARRATIVE_END_LINE = 28560  # Before witness statements


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_narrator_for_line(line_num: int) -> tuple:
    """Determine narrator based on line number."""
    current_book = None
    current_narrator = None
    current_type = None

    for start_line, book, narrator, ntype in BOOK_BOUNDARIES:
        if line_num >= start_line:
            current_book = book
            current_narrator = narrator
            current_type = ntype
        else:
            break

    return current_book, current_narrator, current_type


def segment_text(lines: list, start_line: int, end_line: int, segment_size: int = 1000) -> list:
    """
    Segment text into word blocks with narrator annotations.

    Returns list of segments with:
    - id, start_line, end_line, word_count, text
    - book, narrator, narrator_type
    - needs_review flag for edge cases
    """
    # Extract narrative content
    narrative_lines = lines[start_line-1:end_line]

    # Build word list with line tracking
    words_with_lines = []
    for i, line in enumerate(narrative_lines):
        line_num = start_line + i
        line_words = line.split()
        for word in line_words:
            words_with_lines.append((word, line_num))

    segments = []
    for i in range(0, len(words_with_lines), segment_size):
        chunk = words_with_lines[i:i + segment_size]

        if len(chunk) < segment_size // 2:
            # Skip very short final segment
            continue

        seg_start_line = chunk[0][1]
        seg_end_line = chunk[-1][1]
        seg_words = [w[0] for w in chunk]

        # Get narrator info (use midpoint of segment)
        mid_line = (seg_start_line + seg_end_line) // 2
        book, narrator, ntype = get_narrator_for_line(mid_line)

        # Check if segment spans book boundaries (needs review)
        start_book, _, _ = get_narrator_for_line(seg_start_line)
        end_book, _, _ = get_narrator_for_line(seg_end_line)
        spans_boundary = start_book != end_book

        # Flag special cases for manual review
        needs_review = False
        review_reason = None

        if spans_boundary:
            needs_review = True
            review_reason = f"Spans {start_book}/{end_book} boundary"
        elif book == "Omni":
            needs_review = True
            review_reason = "Multiple mini-narrators in Omni"
        elif book == "Mormon":
            needs_review = True
            review_reason = "Mormon/Moroni transition unclear without chapter markers"

        segment = {
            "id": len(segments) + 1,
            "start_line": seg_start_line,
            "end_line": seg_end_line,
            "word_count": len(seg_words),
            "book": book,
            "narrator": narrator,
            "narrator_type": ntype,
            "needs_review": needs_review,
            "review_reason": review_reason,
            "text": " ".join(seg_words)
        }
        segments.append(segment)

    return segments


def main():
    print("=" * 60)
    print("Book of Mormon Segmentation")
    print(f"Script Version: {SCRIPT_VERSION}")
    print("=" * 60)

    # Read input
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"\nInput: {INPUT_FILE.name}")
    print(f"Total lines: {total_lines:,}")
    print(f"Narrative range: lines {NARRATIVE_START_LINE}-{NARRATIVE_END_LINE}")

    # Segment
    segments = segment_text(lines, NARRATIVE_START_LINE, NARRATIVE_END_LINE, 1000)

    print(f"\nGenerated {len(segments)} segments (1000-word blocks)")

    # Summary by narrator
    narrator_counts = {}
    review_count = 0
    for seg in segments:
        n = seg["narrator"]
        narrator_counts[n] = narrator_counts.get(n, 0) + 1
        if seg["needs_review"]:
            review_count += 1

    print("\nSegments by claimed narrator:")
    for narrator, count in sorted(narrator_counts.items(), key=lambda x: -x[1]):
        print(f"  {narrator}: {count} segments")

    print(f"\nSegments flagged for review: {review_count}")

    # Save segments
    output_path = OUTPUT_DIR / "bom-1830-segments.json"
    output_data = {
        "metadata": {
            "source_file": str(INPUT_FILE),
            "source_hash": compute_text_hash(open(INPUT_FILE).read()),
            "narrative_start_line": NARRATIVE_START_LINE,
            "narrative_end_line": NARRATIVE_END_LINE,
            "segment_size": 1000,
            "total_segments": len(segments),
            "segments_needing_review": review_count,
            "script_version": SCRIPT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "narrator_summary": narrator_counts,
        "segments": segments
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nOutput: {output_path}")

    # Also save a CSV for easier manual review
    csv_path = OUTPUT_DIR / "bom-1830-segments-review.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("id,book,narrator,narrator_type,needs_review,review_reason,start_line,end_line,word_count,first_20_words\n")
        for seg in segments:
            first_words = " ".join(seg["text"].split()[:20])
            # Escape quotes for CSV
            first_words = first_words.replace('"', '""')
            review_reason = seg["review_reason"] or ""
            f.write(f'{seg["id"]},{seg["book"]},{seg["narrator"]},{seg["narrator_type"]},'
                    f'{seg["needs_review"]},{review_reason},{seg["start_line"]},{seg["end_line"]},'
                    f'{seg["word_count"]},"{first_words}"\n')

    print(f"Review CSV: {csv_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
