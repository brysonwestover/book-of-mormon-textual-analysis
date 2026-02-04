#!/usr/bin/env python3
"""
Aggregate verses into speech blocks with continuous tracking.

Takes verse-level annotations and:
1. Tracks speech/document blocks across verses
2. Assigns block IDs for multi-verse embedded content
3. Creates analysis-ready segments excluding embedded discourse

Per GPT-5.2 Pro MVA-2/MVA-3 approach.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

SCRIPT_VERSION = "1.0.0"

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data/text/processed/bom-verses-annotated.json"
OUTPUT_DIR = BASE_DIR / "data/text/processed"


def main():
    print("=" * 70)
    print("Speech Block Aggregation")
    print(f"Script Version: {SCRIPT_VERSION}")
    print("=" * 70)

    # Load annotated verses
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    verses = data["verses"]
    print(f"\nLoaded {len(verses):,} annotated verses")

    # Track speech blocks
    current_block_id = 0
    in_speech = False
    current_speaker = None

    for i, verse in enumerate(verses):
        # Check for speech start
        if verse["has_speech_open"] and not in_speech:
            in_speech = True
            current_block_id += 1
            current_speaker = verse["speaker"]
            verse["embed_block_id"] = current_block_id
            verse["embed_block_position"] = "START"
            if verse["embed_type"] == "NONE":
                verse["embed_type"] = "EMBED_SPEECH"

        # Continue existing speech
        elif in_speech:
            verse["embed_block_id"] = current_block_id
            verse["embed_block_position"] = "MIDDLE"
            if verse["embed_type"] == "NONE":
                verse["embed_type"] = "EMBED_SPEECH_CONTINUATION"
            if not verse["speaker"] and current_speaker:
                verse["speaker"] = current_speaker

            # Check for speech end
            if verse["has_speech_close"]:
                verse["embed_block_position"] = "END"
                in_speech = False
                current_speaker = None

            # Also end on narrator self-ID (returns to frame)
            elif verse["has_narrator_self_id"]:
                verse["embed_block_position"] = "END_IMPLICIT"
                verse["embed_type"] = "NONE"  # This verse is frame, not speech
                verse["embed_block_id"] = None
                in_speech = False
                current_speaker = None

        else:
            verse["embed_block_id"] = None
            verse["embed_block_position"] = None

    # Count blocks and verses in blocks
    block_stats = defaultdict(lambda: {"count": 0, "speakers": set()})
    verses_in_blocks = 0

    for verse in verses:
        if verse.get("embed_block_id"):
            bid = verse["embed_block_id"]
            block_stats[bid]["count"] += 1
            if verse.get("speaker"):
                block_stats[bid]["speakers"].add(verse["speaker"])
            verses_in_blocks += 1

    print(f"\nIdentified {len(block_stats)} speech/document blocks")
    print(f"Verses in embedded blocks: {verses_in_blocks:,} ({100*verses_in_blocks/len(verses):.1f}%)")

    # Show block size distribution
    block_sizes = [b["count"] for b in block_stats.values()]
    if block_sizes:
        print(f"\nBlock size distribution:")
        print(f"  Min: {min(block_sizes)} verses")
        print(f"  Max: {max(block_sizes)} verses")
        print(f"  Avg: {sum(block_sizes)/len(block_sizes):.1f} verses")

    # Create analysis dataset (excluding embedded + Isaiah)
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

    # Segment analysis verses by narrator
    narrator_segments = defaultdict(list)
    for verse in analysis_verses:
        narrator = verse["frame_narrator"]
        narrator_segments[narrator].append(verse)

    print(f"\nAnalysis verses by narrator (frame text only):")
    for narrator, vlist in sorted(narrator_segments.items(), key=lambda x: -len(x[1])):
        words = sum(v["word_count"] for v in vlist)
        print(f"  {narrator}: {len(vlist):,} verses, {words:,} words")

    # Check minimum threshold (≥10 segments = ~500-1000 verses for meaningful analysis)
    MIN_VERSES = 100  # Rough threshold for statistical validity
    valid_narrators = {n: v for n, v in narrator_segments.items() if len(v) >= MIN_VERSES}
    small_narrators = {n: v for n, v in narrator_segments.items() if len(v) < MIN_VERSES}

    print(f"\nNarrators meeting threshold (≥{MIN_VERSES} verses): {len(valid_narrators)}")
    for n in valid_narrators:
        print(f"  ✓ {n}")

    if small_narrators:
        print(f"\nNarrators below threshold (exclude from quantitative claims):")
        for n, v in small_narrators.items():
            print(f"  ✗ {n}: {len(v)} verses")

    # Update metadata
    data["metadata"]["block_aggregation_version"] = SCRIPT_VERSION
    data["metadata"]["block_aggregation_at"] = datetime.now(timezone.utc).isoformat()

    # Add analysis summary
    data["analysis_summary"] = {
        "total_verses": len(verses),
        "verses_in_speech_blocks": verses_in_blocks,
        "speech_blocks_identified": len(block_stats),
        "isaiah_excluded": excluded_isaiah,
        "embedded_excluded": excluded_embed,
        "analysis_verses": len(analysis_verses),
        "valid_narrators": list(valid_narrators.keys()),
        "small_narrators": list(small_narrators.keys()),
        "narrator_verse_counts": {n: len(v) for n, v in narrator_segments.items()},
    }

    # Save updated annotations
    output_path = OUTPUT_DIR / "bom-verses-annotated.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nUpdated: {output_path}")

    # Save analysis-ready dataset (frame text only)
    analysis_output = {
        "metadata": {
            "source": str(INPUT_FILE),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "description": "Frame narration only (embedded discourse and Isaiah excluded)",
            "valid_narrators": list(valid_narrators.keys()),
        },
        "verses": analysis_verses,
    }

    analysis_path = OUTPUT_DIR / "bom-frame-verses.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_output, f, indent=2)
    print(f"Saved analysis dataset: {analysis_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ {len(block_stats)} speech blocks identified and tracked")
    print(f"✓ {len(analysis_verses):,} verses ready for frame-narrator analysis")
    print(f"✓ {len(valid_narrators)} narrators meet statistical threshold")
    print(f"✗ {len(small_narrators)} narrators excluded (below threshold)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
