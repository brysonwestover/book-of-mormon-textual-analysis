#!/usr/bin/env python3
"""
Verse-based Book of Mormon annotation v2.0

Implements GPT-5.2 Pro fixes:
1. Narrator overrides for Jacob in 2 Nephi 6-10
2. Moroni first-person detection in Ether
3. Omni verse-level narrator splits
4. Improved speech open/close patterns
5. Better block tracking with narrative resumption closers

Output: JSON with verse-level annotations ready for stylometric analysis.
"""

import re
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict
from collections import defaultdict

SCRIPT_VERSION = "2.0.0"

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data/text/lds-scriptures-2020.12.08/text/lds-scriptures.txt"
OUTPUT_DIR = BASE_DIR / "data/text/processed"

# =============================================================================
# NARRATOR ASSIGNMENTS - WITH OVERRIDES (GPT fix #1, #2, #3)
# =============================================================================

# Default book-level narrators
BOOK_NARRATORS = {
    "1 Nephi": ("FRAME_NEPHI", "first-person"),
    "2 Nephi": ("FRAME_NEPHI", "first-person"),  # With overrides below
    "Jacob": ("FRAME_JACOB", "first-person"),
    "Enos": ("FRAME_ENOS", "first-person"),
    "Jarom": ("FRAME_JAROM", "first-person"),
    "Omni": ("FRAME_OMNI", "first-person"),  # Will be split by verse
    "Words of Mormon": ("FRAME_MORMON_AUTHOR", "first-person"),
    "Mosiah": ("FRAME_MORMON_ABRIDGER", "third-person"),
    "Alma": ("FRAME_MORMON_ABRIDGER", "third-person"),
    "Helaman": ("FRAME_MORMON_ABRIDGER", "third-person"),
    "3 Nephi": ("FRAME_MORMON_ABRIDGER", "third-person"),
    "4 Nephi": ("FRAME_MORMON_ABRIDGER", "third-person"),
    "Mormon": ("FRAME_MORMON_AUTHOR", "first-person"),
    "Ether": ("FRAME_MORONI_ABRIDGER", "third-person"),  # With first-person detection
    "Moroni": ("FRAME_MORONI_AUTHOR", "first-person"),
}

# GPT Fix #1: Jacob speaks in 2 Nephi 6-10
# These chapters are "the words of Jacob" - not Nephi's voice
CHAPTER_OVERRIDES = {
    ("2 Nephi", 6): ("FRAME_JACOB", "first-person"),
    ("2 Nephi", 7): ("FRAME_JACOB", "first-person"),
    ("2 Nephi", 8): ("FRAME_JACOB", "first-person"),
    ("2 Nephi", 9): ("FRAME_JACOB", "first-person"),
    ("2 Nephi", 10): ("FRAME_JACOB", "first-person"),
}

# Mormon/Moroni transition in book of Mormon
MORMON_MORONI_TRANSITION = 8  # Chapter 8+ is Moroni

# GPT Fix #3: Omni narrator transitions (verse-level)
# Based on "I, [Name]" self-identifications
OMNI_NARRATOR_VERSES = {
    # Omni 1:1-3: Omni
    # Omni 1:4-8: Amaron
    # Omni 1:9: Chemish
    # Omni 1:10-11: Abinadom
    # Omni 1:12-30: Amaleki
    (1, 1): "FRAME_OMNI_OMNI",
    (1, 2): "FRAME_OMNI_OMNI",
    (1, 3): "FRAME_OMNI_OMNI",
    (1, 4): "FRAME_OMNI_AMARON",
    (1, 5): "FRAME_OMNI_AMARON",
    (1, 6): "FRAME_OMNI_AMARON",
    (1, 7): "FRAME_OMNI_AMARON",
    (1, 8): "FRAME_OMNI_AMARON",
    (1, 9): "FRAME_OMNI_CHEMISH",
    (1, 10): "FRAME_OMNI_ABINADOM",
    (1, 11): "FRAME_OMNI_ABINADOM",
    # 1:12-30 are Amaleki (default for remaining)
}

# Isaiah blocks to exclude (2 Nephi chapters with Isaiah quotations)
ISAIAH_CHAPTERS = {
    "2 Nephi": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
}

# =============================================================================
# EMBEDDED DISCOURSE DETECTION - IMPROVED PATTERNS (GPT fix #4, #5)
# =============================================================================

# GPT Fix #2: Moroni first-person detection in Ether
MORONI_FIRST_PERSON = [
    r'\bI,?\s*Moroni\b',
    r'\bI\s+Moroni\b',
    r'\band now I,?\s*Moroni\b',
    r'\bbehold,?\s*I,?\s*Moroni\b',
]

# Improved speech opening patterns (GPT feedback: add simpler patterns)
SPEECH_OPEN_PATTERNS = [
    # Original patterns
    r'\b(and|now)\s+\w+\s+(said|saith|spake|cried|answered|proclaimed|testified).*?\bsaying\b',
    r'\b(began|did)\s+to\s+(say|teach|preach|speak)\b',
    r'\bthese are the words (of|which)\b',
    r'\bopened his mouth and (taught|said|spake)\b',
    # GPT additions: simpler "X said" patterns
    r'\band\s+(\w+)\s+said\s+unto\b',
    r'\band\s+he\s+said\b',
    r'\band\s+she\s+said\b',
    r'\band\s+they\s+said\b',
    r'\bhe\s+spake\s+unto\s+them\b',
    r'\bsaying\s*[:;,]',  # "saying:" or "saying,"
    r'\bbehold,?\s+I\s+say\s+unto\s+you\b',
]

# Improved speech closing patterns (GPT feedback: add "when X had spoken")
SPEECH_CLOSE_PATTERNS = [
    # Original patterns
    r'\b(and\s+)?(thus|now)\s+ended\s+the\s+(words|sayings|preaching|speech)\s+of\b',
    r'\bhad\s+(made\s+an\s+end|finished)\s+(of\s+)?(speaking|preaching|teaching)\b',
    r'\bhad\s+ended\s+(his|their)\s+words\b',
    r'\bamen\b',
    # GPT additions: narrative resumption patterns
    r'\bwhen\s+\w+\s+had\s+(spoken|said|finished\s+speaking)\s+(these\s+)?words\b',
    r'\bafter\s+\w+\s+had\s+(spoken|said)\b',
    r'\bwhen\s+(he|she|they)\s+had\s+(spoken|said|made\s+an\s+end)\b',
    r'\bafter\s+(he|she|they)\s+had\s+(spoken|finished)\b',
    r'\bhaving\s+said\s+these\s+things\b',
]

# Document/epistle patterns
DOCUMENT_OPEN_PATTERNS = [
    r'\ban\s+epistle\b',
    r'\bthis\s+is\s+the\s+(epistle|letter|record|decree|proclamation)\b',
    r'\bthe\s+words\s+(of|which).*wrote\b',
    r'\bthe\s+copy\s+of\s+the\s+(epistle|letter)\b',
]

# Scripture/prophecy patterns
SCRIPTURE_PATTERNS = [
    r'\bthus\s+saith\s+the\s+Lord\b',
    r'\bthe\s+words\s+of\s+Isaiah\b',
    r'\baccording\s+to\s+the\s+words\s+of\s+Isaiah\b',
    r'\bIsaiah\s+(said|spake|wrote|prophesied)\b',
]

# Narrator self-identification (frame marker)
NARRATOR_SELF_ID = [
    r'\bI,?\s+(Nephi|Mormon|Moroni|Jacob|Enos|Jarom)\b',
    r'\b(Nephi|Mormon|Moroni)\s+do\s+(write|make|finish)\b',
    r'\band\s+now\s+I,?\s+(Nephi|Mormon|Moroni|Jacob)\b',
]

# Editorial aside patterns
EDITORIAL_ASIDE_PATTERNS = [
    r'\band\s+thus\s+we\s+see\b',
    r'\band\s+now\s+I,?\s+\w+\b',
    r'\bbehold,?\s+I\s+(say|write|speak)\b',
]

# =============================================================================
# KNOWN SPEAKERS
# =============================================================================

KNOWN_SPEAKERS = [
    "Lehi", "Nephi", "Jacob", "Enos", "Benjamin", "Mosiah", "Alma", "Amulek",
    "Abinadi", "Ammon", "Aaron", "Helaman", "Samuel", "Mormon", "Moroni",
    "Jesus", "Christ", "God", "Lord", "Angel", "Lamoni", "Zeezrom", "Korihor",
    "Gideon", "Limhi", "Zeniff", "Noah", "Pahoran", "Teancum", "Amalickiah",
    "Giddianhi", "Lachoneus", "Ether", "Coriantumr",
]

def extract_speaker(text: str, window: int = 80) -> Optional[str]:
    """Extract speaker name from nearby text."""
    text_sample = text[:window].lower()
    for speaker in KNOWN_SPEAKERS:
        if speaker.lower() in text_sample:
            return speaker
    return None


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_verse_reference(line: str) -> Optional[Dict]:
    """Parse 'Book Chapter:Verse' format."""
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


def get_frame_narrator(book: str, chapter: int, verse: int, text: str) -> tuple:
    """
    Get frame narrator for a verse with all overrides applied.
    GPT fixes #1, #2, #3 implemented here.
    """

    # GPT Fix #1: Chapter-level overrides (Jacob in 2 Nephi)
    if (book, chapter) in CHAPTER_OVERRIDES:
        return CHAPTER_OVERRIDES[(book, chapter)]

    # GPT Fix #3: Omni verse-level splits
    if book == "Omni":
        key = (chapter, verse)
        if key in OMNI_NARRATOR_VERSES:
            return (OMNI_NARRATOR_VERSES[key], "first-person")
        else:
            # Default to Amaleki for verses 12-30
            return ("FRAME_OMNI_AMALEKI", "first-person")

    # GPT Fix #2: Moroni first-person detection in Ether
    if book == "Ether":
        for pattern in MORONI_FIRST_PERSON:
            if re.search(pattern, text, re.IGNORECASE):
                return ("FRAME_MORONI_AUTHOR", "first-person")
        return ("FRAME_MORONI_ABRIDGER", "third-person")

    # Mormon book: split at chapter 8
    if book == "Mormon":
        if chapter >= MORMON_MORONI_TRANSITION:
            return ("FRAME_MORONI_AUTHOR", "first-person")
        else:
            return ("FRAME_MORMON_AUTHOR", "first-person")

    # Default book-level assignment
    return BOOK_NARRATORS.get(book, ("UNKNOWN", "unknown"))


def is_isaiah_block(book: str, chapter: int) -> bool:
    """Check if verse is in Isaiah quotation block."""
    if book in ISAIAH_CHAPTERS:
        return chapter in ISAIAH_CHAPTERS[book]
    return False


def detect_embedded_discourse(text: str) -> Dict:
    """
    Detect embedded discourse markers in verse text.
    Improved patterns per GPT feedback.
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
        "markers_found": [],
        "open_pattern": None,
        "close_pattern": None,
    }

    # Check speech opening patterns
    for pattern in SPEECH_OPEN_PATTERNS:
        if re.search(pattern, text_lower):
            results["has_speech_open"] = True
            results["open_pattern"] = pattern[:40]
            results["markers_found"].append(f"speech_open")
            break

    # Check speech closing patterns
    for pattern in SPEECH_CLOSE_PATTERNS:
        if re.search(pattern, text_lower):
            results["has_speech_close"] = True
            results["close_pattern"] = pattern[:40]
            results["markers_found"].append(f"speech_close")
            break

    # Check document patterns
    for pattern in DOCUMENT_OPEN_PATTERNS:
        if re.search(pattern, text_lower):
            results["has_document"] = True
            results["markers_found"].append(f"document")
            results["embed_type"] = "EMBED_DOCUMENT"
            break

    # Check scripture patterns
    for pattern in SCRIPTURE_PATTERNS:
        if re.search(pattern, text_lower):
            results["has_scripture"] = True
            results["markers_found"].append(f"scripture")
            results["embed_type"] = "EMBED_SCRIPTURE"
            break

    # Check narrator self-ID
    for pattern in NARRATOR_SELF_ID:
        if re.search(pattern, text, re.IGNORECASE):
            results["has_narrator_self_id"] = True
            results["markers_found"].append(f"narrator_id")
            break

    # Check editorial asides
    for pattern in EDITORIAL_ASIDE_PATTERNS:
        if re.search(pattern, text_lower):
            results["has_editorial_aside"] = True
            results["markers_found"].append(f"editorial")
            break

    # Determine embed type
    if results["has_speech_open"] and not results["has_narrator_self_id"]:
        results["embed_type"] = "EMBED_SPEECH"
        results["speaker"] = extract_speaker(text)

    # Set confidence
    if results["has_narrator_self_id"]:
        results["confidence"] = "HIGH"
    elif results["has_speech_open"] and results["speaker"]:
        results["confidence"] = "HIGH"
    elif results["has_speech_open"] or results["has_document"]:
        results["confidence"] = "MEDIUM"
    elif results["has_speech_close"] and not results["has_speech_open"]:
        results["confidence"] = "LOW"

    return results


def annotate_verse(verse_data: Dict) -> Dict:
    """Fully annotate a single verse with all GPT fixes applied."""
    book = verse_data["book"]
    chapter = verse_data["chapter"]
    verse_num = verse_data["verse"]
    text = verse_data["text"]

    # Get frame narrator (with overrides)
    frame_narrator, narrator_type = get_frame_narrator(book, chapter, verse_num, text)

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

    if embed_detection["has_speech_open"] and not embed_detection["speaker"]:
        needs_review = True
        review_reasons.append("Speech detected but speaker unknown")

    # Build annotation
    annotation = {
        "reference": verse_data["reference"],
        "book": book,
        "chapter": chapter,
        "verse": verse_num,
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
        "open_pattern": embed_detection["open_pattern"],
        "close_pattern": embed_detection["close_pattern"],

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
    print("Book of Mormon Verse-Based Annotation v2.0")
    print(f"Script Version: {SCRIPT_VERSION}")
    print("Implements GPT-5.2 Pro fixes")
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
    print("\nAnnotating verses with GPT fixes...")
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
        "speech_opens": 0,
        "speech_closes": 0,
    }

    for v in annotated:
        narrator = v["frame_narrator"]
        stats["by_frame_narrator"][narrator] = stats["by_frame_narrator"].get(narrator, 0) + 1

        embed = v["embed_type"]
        stats["by_embed_type"][embed] = stats["by_embed_type"].get(embed, 0) + 1

        if v["is_isaiah_block"]:
            stats["isaiah_excluded"] += 1

        if v["needs_review"]:
            stats["needs_review"] += 1

        conf = v["confidence"]
        stats["by_confidence"][conf] = stats["by_confidence"].get(conf, 0) + 1

        if v["has_speech_open"]:
            stats["speech_opens"] += 1
        if v["has_speech_close"]:
            stats["speech_closes"] += 1

    # Print summary
    print("\n" + "-" * 50)
    print("ANNOTATION SUMMARY (v2.0 with GPT fixes)")
    print("-" * 50)

    print(f"\nTotal verses: {stats['total_verses']:,}")
    print(f"Total words: {stats['total_words']:,}")

    print("\nBy Frame Narrator (with overrides):")
    for narrator, count in sorted(stats["by_frame_narrator"].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats["total_verses"]
        print(f"  {narrator}: {count:,} ({pct:.1f}%)")

    print(f"\nSpeech detection:")
    print(f"  Speech opens detected: {stats['speech_opens']:,}")
    print(f"  Speech closes detected: {stats['speech_closes']:,}")

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
            "methodology": "GPT-5.2 Pro MVA-3 v2.0 with narrator overrides",
            "fixes_applied": [
                "Jacob in 2 Nephi 6-10 override",
                "Moroni first-person detection in Ether",
                "Omni verse-level narrator splits",
                "Improved speech open/close patterns",
            ],
        },
        "statistics": stats,
        "verses": annotated,
    }

    output_path = OUTPUT_DIR / "bom-verses-annotated-v2.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Save review CSV
    review_path = OUTPUT_DIR / "bom-verses-review-v2.csv"
    with open(review_path, 'w', encoding='utf-8') as f:
        f.write("reference,frame_narrator,embed_type,speaker,confidence,needs_review,open_pattern,close_pattern,first_50_chars\n")
        for v in annotated:
            if v["needs_review"] or v["confidence"] != "HIGH" or v["has_speech_open"] or v["has_speech_close"]:
                first_chars = v["text"][:50].replace('"', '""').replace('\n', ' ')
                open_pat = v.get("open_pattern") or ""
                close_pat = v.get("close_pattern") or ""
                f.write(f'"{v["reference"]}",{v["frame_narrator"]},{v["embed_type"]},'
                        f'{v["speaker"] or ""},"{v["confidence"]}",{v["needs_review"]},'
                        f'"{open_pat}","{close_pat}","{first_chars}"\n')
    print(f"Saved review file: {review_path}")

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
