#!/usr/bin/env python3
"""
Aggregate verses into speech blocks v2.0

Improved block tracking with better close detection.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

SCRIPT_VERSION = "2.0.0"

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data/text/processed/bom-verses-annotated-v2.json"
OUTPUT_DIR = BASE_DIR / "data/text/processed"

# Maximum block length before forced close (safety valve)
MAX_BLOCK_VERSES = 50


def main():
    print("=" * 70)
    print("Speech Block Aggregation v2.0")
    print(f"Script Version: {SCRIPT_VERSION}")
    print("=" * 70)

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    verses = data["verses"]
    print(f"\nLoaded {len(verses):,} annotated verses")

    # Track speech blocks with improved logic
    current_block_id = 0
    in_speech = False
    current_speaker = None
    block_start_verse = None
    block_verse_count = 0

    blocks_info = []

    for i, verse in enumerate(verses):
        # Check for speech close FIRST (before checking open)
        if in_speech:
            close_block = False
            close_reason = None

            # Explicit close marker
            if verse["has_speech_close"]:
                close_block = True
                close_reason = "explicit_close"

            # Narrator self-ID returns to frame
            elif verse["has_narrator_self_id"]:
                close_block = True
                close_reason = "narrator_self_id"

            # Safety: max block length reached
            elif block_verse_count >= MAX_BLOCK_VERSES:
                close_block = True
                close_reason = f"max_length_{MAX_BLOCK_VERSES}"

            # New speech open (different speaker implied)
            elif verse["has_speech_open"] and verse["speaker"] and verse["speaker"] != current_speaker:
                close_block = True
                close_reason = "new_speaker"

            if close_block:
                verse["embed_block_id"] = current_block_id
                verse["embed_block_position"] = "END"
                verse["block_close_reason"] = close_reason

                blocks_info.append({
                    "block_id": current_block_id,
                    "start": block_start_verse,
                    "end": verse["reference"],
                    "verses": block_verse_count + 1,
                    "speaker": current_speaker,
                    "close_reason": close_reason,
                })

                in_speech = False
                current_speaker = None
                block_verse_count = 0

                # If this verse also opens a new speech, handle below
                if not (verse["has_speech_open"] and close_reason != "new_speaker"):
                    continue

        # Check for speech start
        if verse["has_speech_open"] and not in_speech:
            in_speech = True
            current_block_id += 1
            current_speaker = verse["speaker"]
            block_start_verse = verse["reference"]
            block_verse_count = 1

            verse["embed_block_id"] = current_block_id
            verse["embed_block_position"] = "START"

            if verse["embed_type"] == "NONE":
                verse["embed_type"] = "EMBED_SPEECH"

        # Continue existing speech
        elif in_speech:
            verse["embed_block_id"] = current_block_id
            verse["embed_block_position"] = "MIDDLE"
            block_verse_count += 1

            if verse["embed_type"] == "NONE":
                verse["embed_type"] = "EMBED_SPEECH_CONTINUATION"

            if not verse["speaker"] and current_speaker:
                verse["speaker"] = current_speaker

        else:
            verse["embed_block_id"] = None
            verse["embed_block_position"] = None

    # Close any unclosed block at end
    if in_speech:
        blocks_info.append({
            "block_id": current_block_id,
            "start": block_start_verse,
            "end": verses[-1]["reference"],
            "verses": block_verse_count,
            "speaker": current_speaker,
            "close_reason": "end_of_text",
        })

    # Analyze blocks
    print(f"\nIdentified {len(blocks_info)} speech/document blocks")

    verses_in_blocks = sum(b["verses"] for b in blocks_info)
    print(f"Verses in embedded blocks: {verses_in_blocks:,} ({100*verses_in_blocks/len(verses):.1f}%)")

    if blocks_info:
        block_sizes = [b["verses"] for b in blocks_info]
        print(f"\nBlock size distribution:")
        print(f"  Min: {min(block_sizes)} verses")
        print(f"  Max: {max(block_sizes)} verses")
        print(f"  Avg: {sum(block_sizes)/len(block_sizes):.1f} verses")
        print(f"  Median: {sorted(block_sizes)[len(block_sizes)//2]} verses")

        # Show close reason distribution
        close_reasons = defaultdict(int)
        for b in blocks_info:
            close_reasons[b["close_reason"]] += 1

        print(f"\nBlock close reasons:")
        for reason, count in sorted(close_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

        # Show largest blocks
        print(f"\nLargest blocks (top 10):")
        for b in sorted(blocks_info, key=lambda x: -x["verses"])[:10]:
            print(f"  {b['verses']:3d} verses: {b['start']} → {b['end']} (speaker: {b['speaker']}, close: {b['close_reason']})")

    # Create analysis dataset
    analysis_verses = []
    excluded_embed = 0
    excluded_isaiah = 0

    for verse in verses:
        if verse["is_isaiah_block"]:
            excluded_isaiah += 1
            continue
        if verse.get("embed_type") in ["EMBED_SPEECH", "EMBED_SPEECH_CONTINUATION",
                                        "EMBED_DOCUMENT", "EMBED_SCRIPTURE"]:
            excluded_embed += 1
            continue
        analysis_verses.append(verse)

    print(f"\n--- Analysis Dataset ---")
    print(f"Total verses: {len(analysis_verses):,}")
    print(f"Excluded (Isaiah): {excluded_isaiah:,}")
    print(f"Excluded (embedded): {excluded_embed:,}")

    # Count by narrator
    narrator_segments = defaultdict(list)
    for verse in analysis_verses:
        narrator = verse["frame_narrator"]
        narrator_segments[narrator].append(verse)

    print(f"\nAnalysis verses by narrator (frame text only):")
    for narrator, vlist in sorted(narrator_segments.items(), key=lambda x: -len(x[1])):
        words = sum(v["word_count"] for v in vlist)
        print(f"  {narrator}: {len(vlist):,} verses, {words:,} words")

    # Threshold check
    MIN_VERSES = 100
    valid_narrators = {n: v for n, v in narrator_segments.items() if len(v) >= MIN_VERSES}
    small_narrators = {n: v for n, v in narrator_segments.items() if len(v) < MIN_VERSES}

    print(f"\nNarrators meeting threshold (≥{MIN_VERSES} verses): {len(valid_narrators)}")
    for n in valid_narrators:
        print(f"  ✓ {n}")

    if small_narrators:
        print(f"\nNarrators below threshold:")
        for n, v in small_narrators.items():
            print(f"  ✗ {n}: {len(v)} verses")

    # Update and save
    data["metadata"]["block_aggregation_version"] = SCRIPT_VERSION
    data["metadata"]["block_aggregation_at"] = datetime.now(timezone.utc).isoformat()
    data["metadata"]["max_block_verses"] = MAX_BLOCK_VERSES

    data["blocks"] = blocks_info
    data["analysis_summary"] = {
        "total_verses": len(verses),
        "verses_in_speech_blocks": verses_in_blocks,
        "speech_blocks_identified": len(blocks_info),
        "isaiah_excluded": excluded_isaiah,
        "embedded_excluded": excluded_embed,
        "analysis_verses": len(analysis_verses),
        "valid_narrators": list(valid_narrators.keys()),
        "small_narrators": list(small_narrators.keys()),
    }

    output_path = OUTPUT_DIR / "bom-verses-annotated-v2.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nUpdated: {output_path}")

    # Save frame-only dataset
    analysis_output = {
        "metadata": {
            "source": str(INPUT_FILE),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "description": "Frame narration only (embedded discourse and Isaiah excluded)",
            "valid_narrators": list(valid_narrators.keys()),
            "script_version": SCRIPT_VERSION,
        },
        "verses": analysis_verses,
    }

    analysis_path = OUTPUT_DIR / "bom-frame-verses-v2.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_output, f, indent=2)
    print(f"Saved: {analysis_path}")

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
