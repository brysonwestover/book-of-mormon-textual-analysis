#!/usr/bin/env python3
"""
Verse-based Book of Mormon annotation with embedded discourse detection.

Implements GPT-5.2 Pro recommended MVA-3 approach:
1. Extract verses with stable IDs
2. Assign frame narrator labels (with Mormon ABRIDGER/AUTHOR split)
3. Detect embedded discourse using regex patterns
4. Flag low-confidence spans for human review

Output: JSON with verse-level annotations ready for stylometric analysis.
"""

import re
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

SCRIPT_VERSION = "1.0.0"

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data/text/lds-scriptures-2020.12.08/text/lds-scriptures.txt"
OUTPUT_DIR = BASE_DIR / "data/text/processed"

# =============================================================================
# NARRATOR ASSIGNMENTS (per GPT consensus)
# =============================================================================

# Books and their frame narrators
# Format: (book_name, narrator_label, narrator_type)
BOOK_NARRATORS = {
    "1 Nephi": ("FRAME_NEPHI", "first-person"),
    "2 Nephi": ("FRAME_NEPHI", "first-person"),
    "Jacob": ("FRAME_JACOB", "first-person"),
    "Enos": ("FRAME_ENOS", "first-person"),
    "Jarom": ("FRAME_JAROM", "first-person"),
    "Omni": ("FRAME_OMNI_MULTIPLE", "first-person"),  # Multiple mini-narrators
    "Words of Mormon": ("FRAME_MORMON_AUTHOR", "first-person"),
    "Mosiah": ("FRAME_MORMON_ABRIDGER", "third-person"),
    "Alma": ("FRAME_MORMON_ABRIDGER", "third-person"),
    "Helaman": ("FRAME_MORMON_ABRIDGER", "third-person"),
    "3 Nephi": ("FRAME_MORMON_ABRIDGER", "third-person"),
    "4 Nephi": ("FRAME_MORMON_ABRIDGER", "third-person"),
    "Mormon": ("FRAME_MORMON_AUTHOR", "first-person"),  # Default, will split below
    "Ether": ("FRAME_MORONI_ABRIDGER", "third-person"),
    "Moroni": ("FRAME_MORONI_AUTHOR", "first-person"),
}

# Mormon book chapter split: chapters 1-7 = Mormon as author, 8-9 = Moroni
MORMON_MORONI_TRANSITION = 8  # Chapter 8 onward is Moroni

# Isaiah blocks to exclude (2 Nephi chapters with Isaiah quotations)
ISAIAH_CHAPTERS = {
    "2 Nephi": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
}

# =============================================================================
# EMBEDDED DISCOURSE DETECTION PATTERNS (from GPT protocol)
# =============================================================================

# Speech opening patterns
SPEECH_OPEN_PATTERNS = [
    # "And X said/spake... saying"
    r'\b(and|now)\s+\w+\s+(said|saith|spake|cried|answered|proclaimed|testified).*?\bsaying\b',
    # "began to teach/preach/say"
    r'\b(began|did)\s+to\s+(say|teach|preach|speak)\b',
    # "these are the words of X"
    r'\bthese are the words (of|which)\b',
    # "he opened his mouth and taught"
    r'\bopened his mouth and (taught|said|spake)\b',
]

# Speech closing patterns
SPEECH_CLOSE_PATTERNS = [
    r'\b(and )?(thus|now)\s+ended the (words|sayings|preaching|speech) of\b',
    r'\bhad (made an end|finished) (of )?(speaking|preaching|teaching)\b',
    r'\bhad ended (his|their) words\b',
    r'\bamen\b',  # Often ends prayers/sermons
]

# Document/epistle patterns
DOCUMENT_OPEN_PATTERNS = [
    r'\ban epistle\b',
    r'\bthis is the (epistle|letter|record|decree|proclamation)\b',
    r'\bthe words (of|which).*wrote\b',
]

# Scripture/prophecy patterns
SCRIPTURE_PATTERNS = [
    r'\bthus saith the Lord\b',
    r'\bthe words of Isaiah\b',
    r'\baccording to the words of Isaiah\b',
    r'\bIsaiah (said|spake|wrote|prophesied)\b',
]

# Narrator self-identification (high confidence frame marker)
NARRATOR_SELF_ID = [
    r'\bI,?\s+(Nephi|Mormon|Moroni|Jacob|Enos|Jarom)\b',
    r'\b(Nephi|Mormon|Moroni)\s+do\s+(write|make|finish)\b',
]

# Editorial aside patterns
EDITORIAL_ASIDE_PATTERNS = [
    r'\band thus we see\b',
    r'\band now I,?\s+\w+\b',
    r'\bbehold,?\s+I\s+(say|write|speak)\b',
]

# =============================================================================
# SPEAKER EXTRACTION
# =============================================================================

KNOWN_SPEAKERS = [
    "Lehi", "Nephi", "Jacob", "Enos", "Benjamin", "Mosiah", "Alma", "Amulek",
    "Abinadi", "Ammon", "Aaron", "Helaman", "Nephi", "Samuel", "Mormon", "Moroni",
    "Jesus", "Christ", "God", "Lord", "Angel", "Lamoni", "Zeezrom", "Korihor",
    "Gideon", "Limhi", "Zeniff", "Noah", "Pahoran", "Moriantum", "Teancum",
]

def extract_speaker(text: str, window: int = 100) -> Optional[str]:
    """Extract speaker name from nearby text."""
    text_lower = text.lower()
    for speaker in KNOWN_SPEAKERS:
        if speaker.lower() in text_lower[:window]:
            return speaker
    return None


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_verse_reference(line: str) -> Optional[Dict]:
    """Parse 'Book Chapter:Verse' format."""
    # Match patterns like "1 Nephi 1:1", "Alma 5:10", "Words of Mormon 1:1"
    match = re.match(r'^([\w\s]+?)\s+(\d+):(\d+)\s+(.+)$', line.strip())
    if match:
        book = match.group(1).strip()
        chapter = int(match.group(2))
        verse = int(match.group(3))
        text = match.group(4).strip()
        return {
            "book": book,
            "chapter": chapter,
            "verse": verse,
            "text": text,
            "reference": f"{book} {chapter}:{verse}"
        }
    return None


def is_bom_book(book: str) -> bool:
    """Check if book is part of Book of Mormon."""
    bom_books = [
        "1 Nephi", "2 Nephi", "Jacob", "Enos", "Jarom", "Omni",
        "Words of Mormon", "Mosiah", "Alma", "Helaman",
        "3 Nephi", "4 Nephi", "Mormon", "Ether", "Moroni"
    ]
    return book in bom_books


def get_frame_narrator(book: str, chapter: int) -> tuple:
    """Get frame narrator for a verse, handling Mormon/Moroni split."""
    if book == "Mormon":
        if chapter >= MORMON_MORONI_TRANSITION:
            return ("FRAME_MORONI_AUTHOR", "first-person")
        else:
            return ("FRAME_MORMON_AUTHOR", "first-person")

    return BOOK_NARRATORS.get(book, ("UNKNOWN", "unknown"))


def is_isaiah_block(book: str, chapter: int) -> bool:
    """Check if verse is in Isaiah quotation block."""
    if book in ISAIAH_CHAPTERS:
        return chapter in ISAIAH_CHAPTERS[book]
    return False


def detect_embedded_discourse(text: str) -> Dict:
    """
    Detect embedded discourse markers in verse text.
    Returns detection results with confidence levels.
    """
    text_lower = text.lower()
    results = {
        "has_speech_open": False,
        "has_speech_close": False,
        "has_document": False,
        "has_scripture": False,
        "has_narrator_self_id": False,
        "has_editorial_aside": False,
        "speaker": None,
        "embed_type": "NONE",
        "confidence": "HIGH",
        "markers_found": []
    }

    # Check speech patterns
    for pattern in SPEECH_OPEN_PATTERNS:
        if re.search(pattern, text_lower):
            results["has_speech_open"] = True
            results["markers_found"].append(f"speech_open:{pattern[:30]}")
            break

    for pattern in SPEECH_CLOSE_PATTERNS:
        if re.search(pattern, text_lower):
            results["has_speech_close"] = True
            results["markers_found"].append(f"speech_close:{pattern[:30]}")
            break

    # Check document patterns
    for pattern in DOCUMENT_OPEN_PATTERNS:
        if re.search(pattern, text_lower):
            results["has_document"] = True
            results["markers_found"].append(f"document:{pattern[:30]}")
            results["embed_type"] = "EMBED_DOCUMENT"
            break

    # Check scripture patterns
    for pattern in SCRIPTURE_PATTERNS:
        if re.search(pattern, text_lower):
            results["has_scripture"] = True
            results["markers_found"].append(f"scripture:{pattern[:30]}")
            results["embed_type"] = "EMBED_SCRIPTURE"
            break

    # Check narrator self-ID (indicates frame, not embedded)
    for pattern in NARRATOR_SELF_ID:
        if re.search(pattern, text, re.IGNORECASE):
            results["has_narrator_self_id"] = True
            results["markers_found"].append(f"narrator_id:{pattern[:30]}")
            break

    # Check editorial asides
    for pattern in EDITORIAL_ASIDE_PATTERNS:
        if re.search(pattern, text_lower):
            results["has_editorial_aside"] = True
            results["markers_found"].append(f"editorial:{pattern[:30]}")
            break

    # Determine embed type if speech detected
    if results["has_speech_open"] and not results["has_narrator_self_id"]:
        results["embed_type"] = "EMBED_SPEECH"
        results["speaker"] = extract_speaker(text)

    # Set confidence
    if results["has_narrator_self_id"]:
        results["confidence"] = "HIGH"  # Clear frame marker
    elif results["has_speech_open"] and results["speaker"]:
        results["confidence"] = "HIGH"  # Clear speech with speaker
    elif results["has_speech_open"] or results["has_document"]:
        results["confidence"] = "MEDIUM"  # Detected but no speaker
    elif results["has_speech_close"] and not results["has_speech_open"]:
        results["confidence"] = "LOW"  # Close without open (mid-speech verse)

    return results


def annotate_verse(verse_data: Dict) -> Dict:
    """
    Fully annotate a single verse.
    """
    book = verse_data["book"]
    chapter = verse_data["chapter"]
    text = verse_data["text"]

    # Get frame narrator
    frame_narrator, narrator_type = get_frame_narrator(book, chapter)

    # Check for Isaiah exclusion
    is_isaiah = is_isaiah_block(book, chapter)

    # Detect embedded discourse
    embed_detection = detect_embedded_discourse(text)

    # Determine if verse needs review
    needs_review = False
    review_reasons = []

    if embed_detection["confidence"] == "LOW":
        needs_review = True
        review_reasons.append("Low confidence detection")

    if book == "Omni":
        needs_review = True
        review_reasons.append("Multiple mini-narrators in Omni")

    if embed_detection["has_speech_open"] and not embed_detection["speaker"]:
        needs_review = True
        review_reasons.append("Speech detected but speaker unknown")

    # Build annotation
    annotation = {
        "reference": verse_data["reference"],
        "book": book,
        "chapter": chapter,
        "verse": verse_data["verse"],
        "text": text,
        "word_count": len(text.split()),

        # Frame annotation
        "frame_narrator": frame_narrator,
        "narrator_type": narrator_type,

        # Embedded discourse
        "embed_type": embed_detection["embed_type"],
        "speaker": embed_detection["speaker"],
        "has_speech_open": embed_detection["has_speech_open"],
        "has_speech_close": embed_detection["has_speech_close"],
        "has_narrator_self_id": embed_detection["has_narrator_self_id"],
        "has_editorial_aside": embed_detection["has_editorial_aside"],

        # Exclusions
        "is_isaiah_block": is_isaiah,
        "exclude_from_analysis": is_isaiah,

        # Review flags
        "confidence": embed_detection["confidence"],
        "needs_review": needs_review,
        "review_reasons": review_reasons if review_reasons else None,
        "markers_found": embed_detection["markers_found"] if embed_detection["markers_found"] else None,
    }

    return annotation


def main():
    print("=" * 70)
    print("Book of Mormon Verse-Based Annotation")
    print(f"Script Version: {SCRIPT_VERSION}")
    print("Implements GPT-5.2 Pro MVA-3 approach")
    print("=" * 70)

    # Read input
    print(f"\nReading: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract Book of Mormon verses
    verses = []
    for line in lines:
        parsed = parse_verse_reference(line)
        if parsed and is_bom_book(parsed["book"]):
            verses.append(parsed)

    print(f"Extracted {len(verses):,} Book of Mormon verses")

    # Annotate each verse
    print("\nAnnotating verses...")
    annotated = []
    for verse in verses:
        annotation = annotate_verse(verse)
        annotated.append(annotation)

    # Compute statistics
    stats = {
        "total_verses": len(annotated),
        "total_words": sum(v["word_count"] for v in annotated),
        "by_frame_narrator": {},
        "by_embed_type": {},
        "isaiah_excluded": 0,
        "needs_review": 0,
        "by_confidence": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
    }

    for v in annotated:
        # Count by narrator
        narrator = v["frame_narrator"]
        stats["by_frame_narrator"][narrator] = stats["by_frame_narrator"].get(narrator, 0) + 1

        # Count by embed type
        embed = v["embed_type"]
        stats["by_embed_type"][embed] = stats["by_embed_type"].get(embed, 0) + 1

        # Count exclusions
        if v["is_isaiah_block"]:
            stats["isaiah_excluded"] += 1

        # Count review needed
        if v["needs_review"]:
            stats["needs_review"] += 1

        # Count confidence
        conf = v["confidence"]
        stats["by_confidence"][conf] = stats["by_confidence"].get(conf, 0) + 1

    # Print summary
    print("\n" + "-" * 50)
    print("ANNOTATION SUMMARY")
    print("-" * 50)

    print(f"\nTotal verses: {stats['total_verses']:,}")
    print(f"Total words: {stats['total_words']:,}")

    print("\nBy Frame Narrator:")
    for narrator, count in sorted(stats["by_frame_narrator"].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats["total_verses"]
        print(f"  {narrator}: {count:,} ({pct:.1f}%)")

    print("\nBy Embed Type:")
    for embed, count in sorted(stats["by_embed_type"].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats["total_verses"]
        print(f"  {embed}: {count:,} ({pct:.1f}%)")

    print(f"\nIsaiah blocks excluded: {stats['isaiah_excluded']:,}")
    print(f"Verses needing review: {stats['needs_review']:,}")

    print("\nBy Confidence:")
    for conf, count in stats["by_confidence"].items():
        pct = 100 * count / stats["total_verses"]
        print(f"  {conf}: {count:,} ({pct:.1f}%)")

    # Save output
    output_data = {
        "metadata": {
            "source_file": str(INPUT_FILE),
            "source_hash": compute_text_hash(open(INPUT_FILE).read()),
            "script_version": SCRIPT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "methodology": "GPT-5.2 Pro MVA-3 (verse-based + embedded discourse detection)",
        },
        "statistics": stats,
        "verses": annotated,
    }

    output_path = OUTPUT_DIR / "bom-verses-annotated.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Save review CSV for flagged verses
    review_path = OUTPUT_DIR / "bom-verses-review.csv"
    with open(review_path, 'w', encoding='utf-8') as f:
        f.write("reference,frame_narrator,embed_type,speaker,confidence,needs_review,review_reasons,first_50_chars\n")
        for v in annotated:
            if v["needs_review"] or v["confidence"] != "HIGH":
                reasons = "; ".join(v["review_reasons"]) if v["review_reasons"] else ""
                first_chars = v["text"][:50].replace('"', '""').replace('\n', ' ')
                f.write(f'"{v["reference"]}",{v["frame_narrator"]},{v["embed_type"]},'
                        f'{v["speaker"] or ""},"{v["confidence"]}",{v["needs_review"]},'
                        f'"{reasons}","{first_chars}"\n')
    print(f"Saved review file: {review_path}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Review flagged verses in bom-verses-review.csv")
    print("2. Run block aggregation to create multi-verse segments")
    print("3. Verify embedded speech block continuity")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
