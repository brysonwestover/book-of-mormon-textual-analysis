# Book of Mormon Segment Annotation Guide

## Overview

The file `bom-1830-segments.json` contains 269 segments of approximately 1000 words each, pre-annotated with claimed narrators based on book boundaries.

**23 segments are flagged for review.** This guide explains the decisions needed.

---

## Automatic Assignments (246 segments - no review needed)

| Narrator | Segments | Books |
|----------|----------|-------|
| Nephi | 55 | 1 Nephi, 2 Nephi |
| Jacob | 9 | Jacob |
| Enos | 1 | Enos |
| Jarom | 1 | Jarom |
| Mormon | 170 | Words of Mormon, Mosiah, Alma, Helaman, 3 Nephi, 4 Nephi, Mormon 1-7 |
| Moroni | 10 | Mormon 8-9, Ether, Moroni |

---

## Segments Requiring Review

### 1. Book Boundary Segments (12 segments)

These segments span two books. Current assignment uses the midpoint.

| Seg | Lines | Spans | Current | Recommendation |
|-----|-------|-------|---------|----------------|
| 26 | 2850-2960 | 1 Nephi/2 Nephi | Nephi | **Keep** - same narrator |
| 55 | 5874-5983 | 2 Nephi/Jacob | Nephi | Review: mostly 2 Nephi? |
| 64 | 6839-6954 | Jacob/Enos | Jacob | Review: mostly Jacob? |
| 66 | 7058-7170 | Enos/Omni | Jarom | **Exclude** - boundary noise |
| 68 | 7279-7401 | Omni/Words of Mormon | Mormon | Review: mostly WoM? |
| 69 | 7401-7513 | Words of Mormon/Mosiah | Mormon | **Keep** - same narrator |
| 100 | 10640-10754 | Mosiah/Alma | Mormon | **Keep** - same narrator |
| 186 | 19635-19755 | Alma/Helaman | Mormon | **Keep** - same narrator |
| 206 | 21694-21801 | Helaman/3 Nephi | Mormon | **Keep** - same narrator |
| 235 | 24672-24793 | 3 Nephi/4 Nephi | Mormon | **Keep** - same narrator |
| 237 | 24895-25009 | 4 Nephi/Mormon | Mormon | **Keep** - same narrator |
| 247 | 25912-26018 | Mormon/Ether | Moroni | **Keep** - same narrator |
| 263 | 27514-27618 | Ether/Moroni | Moroni | **Keep** - same narrator |

**Decision needed:** Segments 55, 64, 66, 68 - assign to dominant narrator or exclude.

### 2. Omni Segment (1 segment)

**Segment 67** (lines 7170-7279) contains text from 5 mini-narrators:
- Omni (lines 7158-7174)
- Amaron (lines 7175-7189)
- Chemish (lines 7190-7198)
- Abinadom (lines 7199-7267)
- Amaleki (lines 7268-7316)

**Recommendation:** Exclude from analysis or mark as "multiple-uncertain" - too heterogeneous for meaningful narrator assignment.

### 3. Mormon/Moroni Transition (9 segments)

In the Book of Mormon, chapters 1-7 are by Mormon, chapters 8-9 by Moroni.

The transition occurs at **line 25601**: "Behold I, Moroni, do finish the record of my father Mormon."

| Seg | Lines | Contains | Should be |
|-----|-------|----------|-----------|
| 238 | 25009-25107 | Before 25601 | **Mormon** |
| 239 | 25107-25208 | Before 25601 | **Mormon** |
| 240 | 25208-25314 | Before 25601 | **Mormon** |
| 241 | 25314-25419 | Before 25601 | **Mormon** |
| 242 | 25420-25517 | Before 25601 | **Mormon** |
| 243 | 25517-25623 | **Transition at 25601** | Exclude or split |
| 244 | 25623-25717 | After 25601 | **Moroni** |
| 245 | 25717-25814 | After 25601 | **Moroni** |
| 246 | 25814-25912 | After 25601 | **Moroni** |

**Recommendation:**
- Update segments 238-242: narrator = "Mormon"
- Update segments 244-246: narrator = "Moroni"
- Exclude segment 243 (transition segment) or mark as boundary

---

## Recommended Final Counts (after corrections)

| Narrator | Segments | Notes |
|----------|----------|-------|
| Nephi | 55 | Includes boundary segments 26, 55 |
| Jacob | 9 | Includes boundary segment 64 |
| Enos | 1 | |
| Jarom | 1 | Or exclude if too short |
| Mormon | 175 | +5 from Mormon 1-7 corrections |
| Moroni | 22 | +3 from Mormon 8-9 corrections |
| **Excluded** | 5 | Boundary noise, transition segments |

---

## How to Apply Corrections

Edit `bom-1830-segments.json` and update the `narrator` field for flagged segments, or use the provided correction script.

After review, set `needs_review: false` and clear `review_reason` for each corrected segment.
