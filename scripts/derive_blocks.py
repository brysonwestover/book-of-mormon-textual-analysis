#!/usr/bin/env python3
"""
Derive fixed-length blocks from verse-level annotations for stylometric analysis.

Implements GPT-5.2 Pro approved "Option C" algorithm:
1. Create voice runs (contiguous verses with same voice + quote_source status)
2. Assign stable run_id to each run
3. Generate non-overlapping blocks within runs
4. Store full provenance for reproducibility

See: docs/decisions/block-derivation-strategy.md

Version: 1.0.0
Date: 2026-02-01
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Configuration
INPUT_FILE = Path("data/text/processed/bom-verses-annotated-v3.json")
OUTPUT_FILE = Path("data/text/processed/bom-voice-blocks.json")

# Block size targets (in words)
BLOCK_SIZES = [500, 1000, 2000]
MIN_BLOCK_WORDS = 300  # Minimum words to keep a remainder block


def load_verses(filepath: Path) -> dict:
    """Load the annotated verses JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_quote_status(verse: dict) -> str:
    """Return quote status for grouping: 'quoted' or 'original'."""
    return "quoted" if verse.get("quote_source") else "original"


def create_voice_runs(verses: list) -> list:
    """
    Create voice runs: contiguous verses with same voice + quote_source status.

    Returns list of runs, each containing:
    - run_id: Stable identifier
    - voice: The speaking voice
    - quote_status: 'quoted' or 'original'
    - quote_source: The source if quoted, else None
    - frame_narrator: Editorial layer
    - verses: List of verse objects
    - total_words: Sum of word counts
    - start_ref: First verse reference
    - end_ref: Last verse reference
    """
    runs = []
    current_run = None

    for verse in verses:
        voice = verse.get("voice", "UNKNOWN")
        quote_status = get_quote_status(verse)
        quote_source = verse.get("quote_source")
        frame_narrator = verse.get("frame_narrator", "UNKNOWN")

        # Check if we need to start a new run
        if current_run is None:
            # First verse
            current_run = {
                "voice": voice,
                "quote_status": quote_status,
                "quote_source": quote_source,
                "frame_narrator": frame_narrator,
                "verses": [verse],
                "total_words": verse.get("word_count", 0)
            }
        elif (voice == current_run["voice"] and
              quote_status == current_run["quote_status"]):
            # Same run, add verse
            current_run["verses"].append(verse)
            current_run["total_words"] += verse.get("word_count", 0)
        else:
            # New run - save current and start new
            runs.append(current_run)
            current_run = {
                "voice": voice,
                "quote_status": quote_status,
                "quote_source": quote_source,
                "frame_narrator": frame_narrator,
                "verses": [verse],
                "total_words": verse.get("word_count", 0)
            }

    # Don't forget the last run
    if current_run:
        runs.append(current_run)

    # Add run_id and reference info
    for i, run in enumerate(runs):
        run["run_id"] = f"run_{i:04d}"
        run["start_ref"] = run["verses"][0]["reference"]
        run["end_ref"] = run["verses"][-1]["reference"]
        run["verse_count"] = len(run["verses"])

    return runs


def derive_blocks_from_run(run: dict, target_size: int, min_size: int = MIN_BLOCK_WORDS) -> list:
    """
    Derive non-overlapping blocks from a single voice run.

    Args:
        run: Voice run dictionary
        target_size: Target block size in words
        min_size: Minimum words to keep a remainder block

    Returns:
        List of block dictionaries
    """
    blocks = []
    verses = run["verses"]

    if not verses:
        return blocks

    current_block_verses = []
    current_word_count = 0
    block_index = 0

    for verse in verses:
        word_count = verse.get("word_count", 0)

        # Add verse to current block
        current_block_verses.append(verse)
        current_word_count += word_count

        # Check if we've reached target size
        if current_word_count >= target_size:
            # Create block
            block = create_block(
                run=run,
                verses=current_block_verses,
                word_count=current_word_count,
                block_index=block_index,
                target_size=target_size
            )
            blocks.append(block)

            # Reset for next block
            current_block_verses = []
            current_word_count = 0
            block_index += 1

    # Handle remainder
    if current_block_verses:
        if current_word_count >= min_size:
            # Keep as separate block
            block = create_block(
                run=run,
                verses=current_block_verses,
                word_count=current_word_count,
                block_index=block_index,
                target_size=target_size,
                is_remainder=True
            )
            blocks.append(block)
        elif blocks:
            # Merge into previous block
            prev_block = blocks[-1]
            prev_block["_full_verses"].extend(current_block_verses)
            prev_block["word_count"] += current_word_count
            prev_block["end_ref"] = current_block_verses[-1]["reference"]
            prev_block["end_verse"] = current_block_verses[-1]["verse"]
            prev_block["end_chapter"] = current_block_verses[-1]["chapter"]
            prev_block["verse_count"] = len(prev_block["_full_verses"])
            prev_block["text"] = " ".join(v["text"] for v in prev_block["_full_verses"])
            prev_block["verse_refs"] = [{"reference": v["reference"], "word_count": v["word_count"]}
                                        for v in prev_block["_full_verses"]]
            prev_block["merged_remainder"] = True
        elif current_word_count > 0:
            # No previous block to merge into, keep as small block
            block = create_block(
                run=run,
                verses=current_block_verses,
                word_count=current_word_count,
                block_index=block_index,
                target_size=target_size,
                is_remainder=True,
                is_undersized=True
            )
            blocks.append(block)

    return blocks


def create_block(run: dict, verses: list, word_count: int, block_index: int,
                 target_size: int, is_remainder: bool = False,
                 is_undersized: bool = False) -> dict:
    """Create a block dictionary with full provenance."""
    return {
        "block_id": f"{run['run_id']}_blk{block_index:02d}_t{target_size}",
        "run_id": run["run_id"],
        "voice": run["voice"],
        "frame_narrator": run["frame_narrator"],
        "quote_status": run["quote_status"],
        "quote_source": run["quote_source"],
        "target_size": target_size,
        "word_count": word_count,
        "verse_count": len(verses),
        "start_ref": verses[0]["reference"],
        "end_ref": verses[-1]["reference"],
        "book": verses[0]["book"],
        "start_chapter": verses[0]["chapter"],
        "start_verse": verses[0]["verse"],
        "end_chapter": verses[-1]["chapter"],
        "end_verse": verses[-1]["verse"],
        "is_remainder": is_remainder,
        "is_undersized": is_undersized,
        "merged_remainder": False,
        "text": " ".join(v["text"] for v in verses),
        "verse_refs": [{"reference": v["reference"], "word_count": v["word_count"]}
                       for v in verses],  # Lightweight refs for output
        "_full_verses": verses  # Keep full verses temporarily for merging
    }


def compute_statistics(blocks: list, runs: list) -> dict:
    """Compute summary statistics for the derived blocks."""
    stats = {
        "total_runs": len(runs),
        "total_blocks": len(blocks),
        "by_target_size": {},
        "by_voice": {},
        "by_quote_status": {},
        "runs_by_voice": {},
        "word_coverage": {}
    }

    # Group by target size
    for target in BLOCK_SIZES:
        target_blocks = [b for b in blocks if b["target_size"] == target]
        original_blocks = [b for b in target_blocks if b["quote_status"] == "original"]

        stats["by_target_size"][target] = {
            "total_blocks": len(target_blocks),
            "original_voice_blocks": len(original_blocks),
            "total_words": sum(b["word_count"] for b in target_blocks),
            "original_voice_words": sum(b["word_count"] for b in original_blocks),
            "mean_block_size": (sum(b["word_count"] for b in target_blocks) / len(target_blocks)
                               if target_blocks else 0),
            "undersized_blocks": len([b for b in target_blocks if b.get("is_undersized")]),
            "remainder_blocks": len([b for b in target_blocks if b.get("is_remainder")])
        }

    # Group by voice (for 1000-word blocks as reference)
    ref_blocks = [b for b in blocks if b["target_size"] == 1000]
    for block in ref_blocks:
        voice = block["voice"]
        if voice not in stats["by_voice"]:
            stats["by_voice"][voice] = {
                "total_blocks": 0,
                "original_blocks": 0,
                "total_words": 0,
                "original_words": 0
            }
        stats["by_voice"][voice]["total_blocks"] += 1
        stats["by_voice"][voice]["total_words"] += block["word_count"]
        if block["quote_status"] == "original":
            stats["by_voice"][voice]["original_blocks"] += 1
            stats["by_voice"][voice]["original_words"] += block["word_count"]

    # Runs by voice
    for run in runs:
        voice = run["voice"]
        if voice not in stats["runs_by_voice"]:
            stats["runs_by_voice"][voice] = {
                "run_count": 0,
                "total_words": 0,
                "total_verses": 0
            }
        stats["runs_by_voice"][voice]["run_count"] += 1
        stats["runs_by_voice"][voice]["total_words"] += run["total_words"]
        stats["runs_by_voice"][voice]["total_verses"] += run["verse_count"]

    return stats


def main():
    print("=" * 70)
    print("BLOCK DERIVATION FOR STYLOMETRIC ANALYSIS")
    print("=" * 70)
    print()

    # Load data
    print(f"Loading verses from {INPUT_FILE}...")
    data = load_verses(INPUT_FILE)
    verses = data["verses"]
    print(f"  Loaded {len(verses)} verses")
    print()

    # Create voice runs
    print("Creating voice runs...")
    runs = create_voice_runs(verses)
    print(f"  Created {len(runs)} voice runs")

    # Show run distribution
    original_runs = [r for r in runs if r["quote_status"] == "original"]
    quoted_runs = [r for r in runs if r["quote_status"] == "quoted"]
    print(f"  Original voice runs: {len(original_runs)}")
    print(f"  Quoted runs: {len(quoted_runs)}")
    print()

    # Derive blocks at each target size
    all_blocks = []
    for target_size in BLOCK_SIZES:
        print(f"Deriving blocks at {target_size} words...")
        for run in runs:
            blocks = derive_blocks_from_run(run, target_size)
            all_blocks.extend(blocks)

        # Count for this size
        size_blocks = [b for b in all_blocks if b["target_size"] == target_size]
        original_blocks = [b for b in size_blocks if b["quote_status"] == "original"]
        print(f"  Total blocks: {len(size_blocks)}")
        print(f"  Original voice blocks: {len(original_blocks)}")
    print()

    # Clean up temporary fields before statistics/output
    for block in all_blocks:
        if "_full_verses" in block:
            del block["_full_verses"]

    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(all_blocks, runs)

    # Primary analysis summary (1000-word, original voice only)
    print()
    print("=" * 70)
    print("PRIMARY ANALYSIS DATA (1000-word blocks, original voice)")
    print("=" * 70)
    for voice, counts in sorted(stats["by_voice"].items(),
                                 key=lambda x: -x[1]["original_blocks"]):
        if counts["original_blocks"] > 0:
            print(f"  {voice}: {counts['original_blocks']} blocks, "
                  f"{counts['original_words']:,} words")
    print()

    # Prepare output
    output = {
        "metadata": {
            "source_file": str(INPUT_FILE),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script_version": "1.0.0",
            "algorithm": "GPT-5.2 Pro Option C - voice runs with non-overlapping blocks",
            "block_sizes": BLOCK_SIZES,
            "min_block_words": MIN_BLOCK_WORDS,
            "methodology_doc": "docs/decisions/block-derivation-strategy.md"
        },
        "statistics": stats,
        "runs": [
            {
                "run_id": r["run_id"],
                "voice": r["voice"],
                "quote_status": r["quote_status"],
                "quote_source": r["quote_source"],
                "frame_narrator": r["frame_narrator"],
                "start_ref": r["start_ref"],
                "end_ref": r["end_ref"],
                "verse_count": r["verse_count"],
                "total_words": r["total_words"]
            }
            for r in runs
        ],
        "blocks": all_blocks
    }

    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total runs: {len(runs)}")
    print(f"Total blocks: {len(all_blocks)}")
    print()
    print("For stylometric analysis, filter blocks with:")
    print('  quote_status == "original"')
    print('  voice in ["MORMON", "NEPHI", "MORONI", "JACOB"]')

    return output


if __name__ == "__main__":
    main()
