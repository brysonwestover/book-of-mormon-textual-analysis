#!/usr/bin/env python3
"""
Add voice and quote_source fields to Book of Mormon verse annotations.

This implements GPT's Option C dual-annotation recommendation:
- frame_narrator: Who compiled/wrote the plates (editorial layer)
- voice: Who is speaking in this passage (surface voice for stylometry)
- quote_source: Source of quotation if applicable (ISAIAH, MALACHI, ZENOS, etc.)

Version: 3.0.0
Date: 2026-02-01
"""

import json
from datetime import datetime, timezone
from pathlib import Path

INPUT_FILE = Path("data/text/processed/bom-verses-annotated-v2.json")
OUTPUT_FILE = Path("data/text/processed/bom-verses-annotated-v3.json")


def normalize_voice_from_frame(frame_narrator):
    """Convert frame_narrator to normalized voice value."""
    if frame_narrator is None:
        return "UNKNOWN"

    fn = frame_narrator.upper()

    if "NEPHI" in fn:
        return "NEPHI"
    if "JACOB" in fn:
        return "JACOB"
    if "ENOS" in fn:
        return "ENOS"
    if "JAROM" in fn:
        return "JAROM"
    if "OMNI" in fn:
        return "OMNI"
    if "AMARON" in fn:
        return "AMARON"
    if "CHEMISH" in fn:
        return "CHEMISH"
    if "ABINADOM" in fn:
        return "ABINADOM"
    if "AMALEKI" in fn:
        return "AMALEKI"
    if "MORMON" in fn:
        return "MORMON"
    if "MORONI" in fn:
        return "MORONI"

    return "UNKNOWN"


def get_quote_source(verse):
    """Determine quote_source based on existing flags and known ranges."""
    if verse.get("is_isaiah_block"):
        return "ISAIAH"
    return None


def is_in_range(book, chapter, verse, ranges):
    """Check if a verse is within any of the specified ranges."""
    for r in ranges:
        if r["book"] == book:
            if r.get("chapters"):
                if chapter in r["chapters"]:
                    return True
            elif r.get("chapter") and r.get("verses"):
                if chapter == r["chapter"] and verse in r["verses"]:
                    return True
            elif r.get("chapter_range"):
                start, end = r["chapter_range"]
                if start <= chapter <= end:
                    return True
    return False


# Define special cases
# 2 Nephi 6:2-10:25 = Jacob speaking (frame stays NEPHI, voice = JACOB)
JACOB_DISCOURSE = {
    "book": "2 Nephi",
    "chapter_range": (6, 10),
    "exclude_verses": [(6, 1)]  # 6:1 is Nephi's intro
}

# Moroni 7-9 = Mormon's content
MORMON_IN_MORONI = [
    {"book": "Moroni", "chapter": 7},  # Mormon's sermon
    {"book": "Moroni", "chapter": 8},  # Mormon's letter
    {"book": "Moroni", "chapter": 9},  # Mormon's letter
]

# Jacob 5 = Zenos quotation
ZENOS_ALLEGORY = {"book": "Jacob", "chapter": 5}

# 3 Nephi Sermon on the Mount
SERMON_ON_MOUNT = {"book": "3 Nephi", "chapter_range": (12, 14)}

# 3 Nephi Malachi
MALACHI_QUOTES = {"book": "3 Nephi", "chapter_range": (24, 25)}

# Isaiah blocks (supplement existing is_isaiah_block)
ISAIAH_BLOCKS = [
    {"book": "1 Nephi", "chapter_range": (20, 21)},
    {"book": "2 Nephi", "chapter_range": (12, 24)},
    {"book": "2 Nephi", "chapter": 27},  # Isaiah dependence
]


def process_verse(verse):
    """Add voice and quote_source fields to a verse."""
    book = verse["book"]
    chapter = verse["chapter"]
    verse_num = verse["verse"]

    # Start with defaults
    voice = normalize_voice_from_frame(verse["frame_narrator"])
    quote_source = get_quote_source(verse)

    # Special case: 2 Nephi 6:2-10:25 (Jacob's discourse)
    if book == "2 Nephi" and 6 <= chapter <= 10:
        if not (chapter == 6 and verse_num == 1):  # Exclude 6:1 (Nephi's intro)
            voice = "JACOB"
            # Keep frame_narrator as NEPHI (will update separately)

    # Special case: Moroni 7-9 (Mormon's content)
    if book == "Moroni" and chapter in [7, 8, 9]:
        voice = "MORMON"

    # Special case: Jacob 5 (Zenos)
    if book == "Jacob" and chapter == 5:
        voice = "ZENOS"
        quote_source = "ZENOS"

    # Special case: 3 Nephi 12-14 (Sermon on the Mount)
    if book == "3 Nephi" and 12 <= chapter <= 14:
        quote_source = "MATTHEW"
        # voice stays MORMON (Jesus speaking, but through Mormon's record)
        # Actually, this is Jesus speaking - let's mark it
        voice = "JESUS_CHRIST"

    # Special case: 3 Nephi 24-25 (Malachi)
    if book == "3 Nephi" and chapter in [24, 25]:
        voice = "MALACHI"
        quote_source = "MALACHI"

    # Isaiah blocks - ensure quote_source is set
    if book == "1 Nephi" and chapter in [20, 21]:
        voice = "ISAIAH"
        quote_source = "ISAIAH"
    if book == "2 Nephi" and 12 <= chapter <= 24:
        voice = "ISAIAH"
        quote_source = "ISAIAH"
    if book == "2 Nephi" and chapter == 27:
        # Heavy Isaiah dependence but not pure quotation
        quote_source = "ISAIAH"
        # voice stays NEPHI (Nephi paraphrasing Isaiah)

    # Add new fields
    verse["voice"] = voice
    verse["quote_source"] = quote_source

    return verse


def update_frame_narrator(verse):
    """Update frame_narrator where needed (2 Nephi 6-10)."""
    book = verse["book"]
    chapter = verse["chapter"]

    # 2 Nephi 6-10: frame should be NEPHI (he's the compiler), not JACOB
    if book == "2 Nephi" and 6 <= chapter <= 10:
        if verse["frame_narrator"] == "FRAME_JACOB":
            verse["frame_narrator"] = "FRAME_NEPHI"

    return verse


def main():
    print("Loading annotations from", INPUT_FILE)
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    print(f"Processing {len(data['verses'])} verses...")

    # Process each verse
    for verse in data["verses"]:
        verse = update_frame_narrator(verse)
        verse = process_verse(verse)

    # Update metadata
    data["metadata"]["script_version"] = "3.0.0"
    data["metadata"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    data["metadata"]["methodology"] = "GPT-5.2 Pro Option C dual-annotation (frame + voice)"
    data["metadata"]["schema_version"] = "3.0"
    data["metadata"]["new_fields"] = ["voice", "quote_source"]
    data["metadata"]["voice_annotation_notes"] = [
        "voice = surface speaking voice for stylometry",
        "frame_narrator = plate compiler/editorial layer",
        "quote_source = source of quotation (ISAIAH, ZENOS, MALACHI, MATTHEW, null)",
        "2 Nephi 6:2-10:25: frame=NEPHI, voice=JACOB",
        "Moroni 7-9: frame=MORONI, voice=MORMON",
        "Jacob 5: voice=ZENOS, quote_source=ZENOS",
        "3 Nephi 12-14: voice=JESUS_CHRIST, quote_source=MATTHEW",
        "3 Nephi 24-25: voice=MALACHI, quote_source=MALACHI",
        "Isaiah blocks: voice=ISAIAH, quote_source=ISAIAH"
    ]

    # Compute new statistics
    voice_counts = {}
    quote_source_counts = {}
    for verse in data["verses"]:
        v = verse.get("voice", "UNKNOWN")
        voice_counts[v] = voice_counts.get(v, 0) + 1
        qs = verse.get("quote_source")
        if qs:
            quote_source_counts[qs] = quote_source_counts.get(qs, 0) + 1

    data["statistics"]["by_voice"] = dict(sorted(voice_counts.items(), key=lambda x: -x[1]))
    data["statistics"]["by_quote_source"] = dict(sorted(quote_source_counts.items(), key=lambda x: -x[1]))

    print("\nVoice distribution:")
    for voice, count in sorted(voice_counts.items(), key=lambda x: -x[1]):
        print(f"  {voice}: {count}")

    print("\nQuote source distribution:")
    for qs, count in sorted(quote_source_counts.items(), key=lambda x: -x[1]):
        print(f"  {qs}: {count}")

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print("Done!")

    return data


if __name__ == "__main__":
    main()
