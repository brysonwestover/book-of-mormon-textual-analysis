# Verse-Level Annotation Schema v3.0

**File:** `bom-verses-annotated-v3.json`
**Date:** 2026-02-01
**Verses:** 6,604

---

## Overview

This file contains verse-level annotations for the Book of Mormon with dual-layer attribution:

1. **frame_narrator** - Who compiled/wrote the plates (editorial layer)
2. **voice** - Who is speaking (surface voice for stylometry)
3. **quote_source** - Source of quotation if applicable

This schema was developed through GPT-5.2 Pro consultation and validated against external sources (97.5% agreement, κ=0.96).

---

## Key Fields

| Field | Description | Example Values |
|-------|-------------|----------------|
| `reference` | Verse reference | "1 Nephi 1:1" |
| `book` | Book name | "1 Nephi" |
| `chapter` | Chapter number | 1 |
| `verse` | Verse number | 1 |
| `text` | Verse text | "I, Nephi, having been born..." |
| `frame_narrator` | Plate compiler | FRAME_NEPHI, FRAME_MORMON, etc. |
| `voice` | Speaking voice | NEPHI, JACOB, MORMON, ISAIAH, etc. |
| `quote_source` | Quotation source | ISAIAH, ZENOS, MATTHEW, MALACHI, null |
| `is_isaiah_block` | Legacy Isaiah flag | true/false |
| `embed_type` | Embedded speech type | NONE, EMBED_SPEECH, etc. |
| `confidence` | Annotation confidence | HIGH, MEDIUM, LOW |

---

## Voice Distribution

| Voice | Verses | % | Description |
|-------|--------|---|-------------|
| MORMON | 4,229 | 64.0% | Mormon's narration and abridgment |
| NEPHI | 942 | 14.3% | Nephi's first-person account |
| MORONI | 570 | 8.6% | Moroni's writing |
| ISAIAH | 323 | 4.9% | Isaiah quotations |
| JACOB | 258 | 3.9% | Jacob's discourse |
| JESUS_CHRIST | 109 | 1.7% | Jesus speaking in 3 Nephi |
| ZENOS | 77 | 1.2% | Allegory of Olive Tree |
| ENOS | 27 | 0.4% | Enos's account |
| OMNI | 30 | 0.5% | Omni section narrators |
| MALACHI | 24 | 0.4% | Malachi quotations |
| JAROM | 15 | 0.2% | Jarom's account |

---

## Quote Source Distribution

| Source | Verses | Location |
|--------|--------|----------|
| ISAIAH | 358 | 1 Ne 20-21, 2 Ne 12-24, 2 Ne 27 |
| MATTHEW | 109 | 3 Ne 12-14 (Sermon on Mount) |
| ZENOS | 77 | Jacob 5 |
| MALACHI | 24 | 3 Ne 24-25 |

---

## Usage for Stylometry

### Primary Voice Analysis (excluding quotations)

```python
import json

with open('bom-verses-annotated-v3.json') as f:
    data = json.load(f)

# Get non-quoted verses for stylometry
original_voices = [v for v in data['verses']
                   if v['quote_source'] is None]

# Group by voice
by_voice = {}
for v in original_voices:
    voice = v['voice']
    by_voice.setdefault(voice, []).append(v)
```

### Frame Narrator Analysis

```python
# Group by frame (editorial layer)
by_frame = {}
for v in data['verses']:
    frame = v['frame_narrator']
    by_frame.setdefault(frame, []).append(v)
```

---

## Special Cases

| Case | frame_narrator | voice | quote_source |
|------|----------------|-------|--------------|
| 2 Ne 6:2-10:25 | FRAME_NEPHI | JACOB | null |
| Moroni 7-9 | FRAME_MORONI | MORMON | null |
| Jacob 5 | FRAME_JACOB | ZENOS | ZENOS |
| 3 Ne 12-14 | FRAME_MORMON | JESUS_CHRIST | MATTHEW |
| 3 Ne 24-25 | FRAME_MORMON | MALACHI | MALACHI |

See `docs/decisions/voice-annotation-schema-v3.md` for full documentation.

---

## Validation

- External agreement: 97.5% with bcgmaxwell
- Cohen's κ: 0.96 (almost perfect)
- All major cases validated against GPT-5.2 Pro recommendations
