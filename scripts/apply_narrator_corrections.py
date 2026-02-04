#!/usr/bin/env python3
"""
Apply narrator corrections to Book of Mormon segments.

Based on ANNOTATION-GUIDE.md recommendations:
1. Fix Mormon/Moroni transition in Book of Mormon
2. Mark uncertain boundary segments
3. Generate final annotated segments
"""

import json
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).parent.parent
SEGMENTS_FILE = BASE_DIR / "data/text/processed/bom-1830-segments.json"
OUTPUT_FILE = BASE_DIR / "data/text/processed/bom-1830-segments-annotated.json"


def apply_corrections():
    print("Applying narrator corrections to BoM segments...")

    with open(SEGMENTS_FILE, 'r') as f:
        data = json.load(f)

    segments = data["segments"]
    corrections = []
    exclusions = []

    for seg in segments:
        sid = seg["id"]

        # Mormon/Moroni transition corrections
        if sid in [238, 239, 240, 241, 242]:
            if seg["narrator"] != "Mormon":
                corrections.append(f"Segment {sid}: {seg['narrator']} → Mormon (Mormon ch 1-7)")
            seg["narrator"] = "Mormon"
            seg["narrator_type"] = "first-person"
            seg["needs_review"] = False
            seg["review_reason"] = None
            seg["correction_applied"] = "Mormon ch 1-7 (before Moroni transition)"

        elif sid == 243:
            # Transition segment - mark for exclusion
            seg["narrator"] = "TRANSITION"
            seg["needs_review"] = True
            seg["review_reason"] = "Mormon/Moroni transition at line 25601 - recommend exclude"
            seg["exclude_recommended"] = True
            exclusions.append(f"Segment {sid}: marked as transition (Mormon→Moroni)")

        elif sid in [244, 245, 246]:
            if seg["narrator"] != "Moroni":
                corrections.append(f"Segment {sid}: {seg['narrator']} → Moroni (Mormon ch 8-9)")
            seg["narrator"] = "Moroni"
            seg["narrator_type"] = "first-person"
            seg["needs_review"] = False
            seg["review_reason"] = None
            seg["correction_applied"] = "Mormon ch 8-9 (after Moroni takes over)"

        # Omni segment - mark for exclusion
        elif sid == 67:
            seg["narrator"] = "MULTIPLE"
            seg["needs_review"] = True
            seg["review_reason"] = "5 mini-narrators (Omni, Amaron, Chemish, Abinadom, Amaleki)"
            seg["exclude_recommended"] = True
            exclusions.append(f"Segment {sid}: marked as multiple mini-narrators")

        # Boundary segments with same narrator on both sides - clear review flag
        elif sid in [26, 69, 100, 186, 206, 235, 237, 247, 263]:
            seg["needs_review"] = False
            seg["review_reason"] = None
            seg["correction_applied"] = "Same narrator on both sides of boundary"

        # Boundary segments needing user decision
        elif sid == 55:
            seg["review_reason"] = "Spans 2 Nephi/Jacob - mostly 2 Nephi, recommend keep as Nephi"
        elif sid == 64:
            seg["review_reason"] = "Spans Jacob/Enos - mostly Jacob, recommend keep as Jacob"
        elif sid == 66:
            seg["review_reason"] = "Spans Jarom/Omni boundary - recommend exclude"
            seg["exclude_recommended"] = True
            exclusions.append(f"Segment {sid}: Jarom/Omni boundary noise")
        elif sid == 68:
            seg["review_reason"] = "Spans Omni/Words of Mormon - recommend keep as Mormon"

    # Update metadata
    data["metadata"]["corrections_applied"] = datetime.now(timezone.utc).isoformat()
    data["metadata"]["correction_notes"] = [
        "Mormon/Moroni transition fixed (line 25601)",
        "Segments 238-242: Mormon",
        "Segment 243: excluded (transition)",
        "Segments 244-246: Moroni",
        "Segment 67: excluded (Omni multi-narrator)",
        "Segment 66: excluded (boundary noise)"
    ]

    # Recount by narrator (excluding recommended exclusions)
    final_counts = {}
    excluded_count = 0
    review_remaining = 0

    for seg in segments:
        if seg.get("exclude_recommended"):
            excluded_count += 1
            continue
        n = seg["narrator"]
        final_counts[n] = final_counts.get(n, 0) + 1
        if seg.get("needs_review"):
            review_remaining += 1

    data["narrator_summary"] = final_counts
    data["metadata"]["excluded_segments"] = excluded_count
    data["metadata"]["segments_still_needing_review"] = review_remaining

    # Save corrected file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    # Print summary
    print(f"\nCorrections applied: {len(corrections)}")
    for c in corrections:
        print(f"  {c}")

    print(f"\nExclusions recommended: {len(exclusions)}")
    for e in exclusions:
        print(f"  {e}")

    print(f"\nFinal narrator counts (excluding {excluded_count} recommended exclusions):")
    for narrator, count in sorted(final_counts.items(), key=lambda x: -x[1]):
        print(f"  {narrator}: {count}")

    print(f"\nSegments still needing review: {review_remaining}")
    print(f"\nOutput: {OUTPUT_FILE}")

    return 0


if __name__ == "__main__":
    exit(apply_corrections())
