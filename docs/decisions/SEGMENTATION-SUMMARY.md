# Segmentation Analysis Summary

**Date:** 2026-02-01
**Status:** Complete (v3.0 schema implemented)
**Project:** Book of Mormon Textual Analysis

---

## Overview

This document summarizes the complete segmentation analysis work for the Book of Mormon stylometric study. The goal was to create a rigorous, reproducible annotation of the Book of Mormon text that correctly identifies speaking voices for authorship analysis.

## Problem Statement

For stylometric analysis asking "Do claimed authors exhibit distinct stylistic signatures?", we need to accurately identify who is speaking in each passage. The initial approach of labeling by book boundaries conflated multiple concepts:

1. **Who compiled/wrote the plates** (editorial layer)
2. **Who is speaking in a given passage** (surface voice)
3. **Quoted material from other sources** (Isaiah, Zenos, etc.)

## Solution: Dual-Layer Annotation (v3.0)

After consultation with GPT-5.2 Pro, we implemented a dual-layer annotation schema:

| Field | Purpose | Example Values |
|-------|---------|----------------|
| `frame_narrator` | Who compiled this section | FRAME_NEPHI, FRAME_MORMON, FRAME_MORONI |
| `voice` | Who is speaking | NEPHI, JACOB, MORMON, ISAIAH, JESUS_CHRIST |
| `quote_source` | Source if quotation | ISAIAH, ZENOS, MALACHI, MATTHEW, null |

This allows precise filtering for stylometric analysis (e.g., "give me all original NEPHI voice, excluding quotations").

---

## Key Decisions Made

### 1. Segmentation Strategy
**Decision:** Verse-level annotation (6,604 verses)
**Rationale:** Allows boundary-aligned analysis without crossing narrator transitions. Fixed-length blocks can be derived as needed.

### 2. Mormon Label Handling
**Decision:** Dual-layer separation of frame vs voice
**Rationale:** Separates Mormon's editorial frame from embedded speeches he records.

### 3. Embedded Discourse Handling
**Decision:** Automatic detection with separate voice labeling
**Rationale:** Major speeches (Jacob's discourse, Zenos's allegory) get distinct voice labels.

### 4. Scripture/Isaiah Handling
**Decision:** Label separately with quote_source tracking
**Rationale:** 568 verses tagged as quotations for easy filtering.

### 5. Small Sample Narrators
**Decision:** Include with explicit caveats (≥100 verses for quantitative claims)
**Rationale:** Enos (27), Jarom (15), Omni (30) included for descriptive analysis only.

### 6. Evaluation Strategy
**Decision:** Blocked cross-validation with bootstrap confidence intervals
**Rationale:** Prevents text-dependence leakage, quantifies uncertainty.

---

## Special Cases Implemented

| Case | Location | frame_narrator | voice | quote_source |
|------|----------|----------------|-------|--------------|
| Jacob's Discourse | 2 Ne 6:2-10:25 | FRAME_NEPHI | JACOB | null |
| Mormon's Content in Moroni | Moroni 7-9 | FRAME_MORONI | MORMON | null |
| Zenos Allegory | Jacob 5 | FRAME_JACOB | ZENOS | ZENOS |
| Sermon on Mount | 3 Ne 12-14 | FRAME_MORMON | JESUS_CHRIST | MATTHEW |
| Malachi Quotation | 3 Ne 24-25 | FRAME_MORMON | MALACHI | MALACHI |
| Isaiah Blocks | 1 Ne 20-21, 2 Ne 12-24 | FRAME_NEPHI | ISAIAH | ISAIAH |
| Isaiah Paraphrase | 2 Ne 27 | FRAME_NEPHI | NEPHI | ISAIAH |

---

## Final Voice Distribution

| Voice | Verses | % | Analysis Status |
|-------|--------|---|-----------------|
| MORMON | 4,229 | 64.0% | Primary |
| NEPHI | 942 | 14.3% | Primary |
| MORONI | 570 | 8.6% | Primary |
| ISAIAH | 323 | 4.9% | Quotation (exclude) |
| JACOB | 258 | 3.9% | Primary |
| JESUS_CHRIST | 109 | 1.7% | Quotation (exclude) |
| ZENOS | 77 | 1.2% | Quotation (exclude) |
| OMNI | 30 | 0.5% | Exploratory |
| ENOS | 27 | 0.4% | Exploratory |
| MALACHI | 24 | 0.4% | Quotation (exclude) |
| JAROM | 15 | 0.2% | Exploratory |

**For stylometric analysis:** Filter with `quote_source is None` to get 6,036 original-voice verses.

---

## Validation

### External Cross-Check
- **Source:** bcgmaxwell "Writer" annotations (WordPress)
- **Agreement:** 97.5% at frame level
- **Cohen's κ:** 0.96 (almost perfect agreement)
- **Main difference:** 2 Nephi 6-10 (intentional - our Jacob voice attribution)

### GPT-5.2 Pro Methodology Review
- Critique of initial approach (6 issues identified)
- Recommended dual-layer annotation (Option C)
- Sample passage validation (5/6 correct, 1 minor fix)
- Approved final v3.0 schema

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| `voice-annotation-schema-v3.md` | Full v3.0 schema specification |
| `segment-annotation-decisions.md` | GPT critique and 6 decision points (resolved) |
| `segmentation-validation-review.md` | V2.0 validation and 4 review questions (answered) |
| `SEGMENTATION-SUMMARY.md` | This overview document |

## Data Files

| File | Description |
|------|-------------|
| `bom-verses-annotated-v3.json` | Final annotated dataset (6,604 verses) |
| `bom-verses-annotated-v2.json` | Previous version (frame_narrator only) |
| `VERSE-ANNOTATION-v3.md` | Quick reference for v3 schema |

## Scripts

| Script | Purpose |
|--------|---------|
| `add_voice_annotation.py` | Creates v3 from v2 with voice/quote_source |
| `compare_voice_sources.py` | Compare against bcgmaxwell |
| `compare_voice_sources_detailed.py` | Detailed comparison analysis |
| `compare_voice_sources_segments.py` | Segment-level comparison |

---

## Consultation Log

| Date | Consultant | Topic | Outcome |
|------|------------|-------|---------|
| 2026-02-01 | GPT-5.2 Pro | Initial methodology critique | 6 decision points identified |
| 2026-02-01 | GPT-5.2 Pro | Annotation protocol | Full protocol provided |
| 2026-02-01 | GPT-5.2 Pro | Implementation review | Fixes needed identified |
| 2026-02-01 | GPT-5.2 Pro | V2.0 validation | 5/6 correct, 1 fix needed |
| 2026-02-01 | GPT-5.2 Pro | Dual-layer recommendation | Option C approved |
| 2026-02-01 | GPT-5.2 Pro | 2 Ne 6-10 frame vs voice | Consensus on frame=NEPHI, voice=JACOB |

---

## Usage for Stylometry

### Primary Analysis (excluding quotations)

```python
import json

with open('data/text/processed/bom-verses-annotated-v3.json') as f:
    data = json.load(f)

# Get non-quoted verses for author-voice stylometry
stylometry_verses = [v for v in data['verses']
                     if v['quote_source'] is None
                     and v['voice'] in ['NEPHI', 'JACOB', 'MORMON', 'MORONI']]

# Group by voice
by_voice = {}
for v in stylometry_verses:
    voice = v['voice']
    by_voice.setdefault(voice, []).append(v)

print(f"Verses for analysis: {len(stylometry_verses)}")
for voice, verses in sorted(by_voice.items(), key=lambda x: -len(x[1])):
    print(f"  {voice}: {len(verses)}")
```

### Sensitivity Analysis

```python
# Run with different inclusion criteria:

# 1. All original voice (no quotations)
original = [v for v in data['verses'] if v['quote_source'] is None]

# 2. Excluding Jesus Christ's words
no_jesus = [v for v in original if v['voice'] != 'JESUS_CHRIST']

# 3. Major narrators only
major_only = [v for v in original
              if v['voice'] in ['NEPHI', 'JACOB', 'MORMON', 'MORONI']]
```

---

## Conclusion

The segmentation analysis is complete with a rigorous, well-documented v3.0 annotation schema. The dual-layer approach (frame + voice + quote_source) addresses all methodological concerns raised by GPT-5.2 Pro and achieves 97.5% agreement with external validation sources.

**Ready to proceed to stylometric analysis.**
