# Voice Annotation Schema v3.0

**Date:** 2026-02-01
**Status:** Implemented
**Consulted:** GPT-5.2 Pro (methodology validation)

---

## Summary

This document describes the dual-annotation schema implemented in `bom-verses-annotated-v3.json`, which separates **frame narrator** (editorial/compiler layer) from **voice** (surface speaking voice for stylometry).

## Rationale

### Problem Statement

The original annotation used a single `frame_narrator` field that conflated two distinct concepts:

1. **Who compiled/wrote the plates** (editorial layer)
2. **Who is speaking in a given passage** (surface voice)

For stylometric analysis asking "Do claimed authors exhibit distinct stylistic signatures?", we need to identify the **speaking voice**, not just the plate compiler.

### GPT Consultation

GPT-5.2 Pro recommended **Option C: Dual Annotation**:

> "Use Option C. Treat 2 Nephi 6–10 as FRAME=Nephi, VOICE=Jacob, and explicitly tag/remove Isaiah quotations for author-signal stylometry. This preserves methodological validity and keeps the analysis aligned with the stated hypothesis."

---

## Schema Definition

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `frame_narrator` | string | Who compiled/wrote this section of the plates (editorial layer) |
| `voice` | string | Who is speaking in this passage (surface voice for stylometry) |
| `quote_source` | string \| null | Source of quotation if applicable (ISAIAH, MALACHI, ZENOS, MATTHEW, null) |

### Voice Values

| Value | Description | Verse Count |
|-------|-------------|-------------|
| MORMON | Mormon's voice (narration, abridgment, commentary) | 4,229 |
| NEPHI | Nephi's voice | 942 |
| MORONI | Moroni's voice | 570 |
| ISAIAH | Isaiah quotations | 323 |
| JACOB | Jacob's voice | 258 |
| JESUS_CHRIST | Jesus Christ speaking (3 Nephi) | 109 |
| ZENOS | Zenos quotations (Jacob 5) | 77 |
| ENOS | Enos's voice | 27 |
| OMNI | Omni's voice | 30 |
| MALACHI | Malachi quotations | 24 |
| JAROM | Jarom's voice | 15 |

### Quote Source Values

| Value | Description | Verse Count |
|-------|-------------|-------------|
| ISAIAH | Isaiah quotations (1 Ne 20-21, 2 Ne 12-24, 2 Ne 27) | 358 |
| MATTHEW | Sermon on the Mount parallel (3 Ne 12-14) | 109 |
| ZENOS | Allegory of the Olive Tree (Jacob 5) | 77 |
| MALACHI | Malachi quotations (3 Ne 24-25) | 24 |
| null | Not a quotation | 6,036 |

---

## Special Cases

### 1. 2 Nephi 6-10 (Jacob's Discourse)

**Context:** Nephi records Jacob's sermon delivered to the Nephites.

| Verse Range | frame_narrator | voice | Rationale |
|-------------|----------------|-------|-----------|
| 2 Nephi 6:1 | FRAME_NEPHI | NEPHI | Nephi's editorial header introducing Jacob |
| 2 Nephi 6:2-10:25 | FRAME_NEPHI | JACOB | Jacob speaking in first person |

**Rationale:** Nephi is the plate compiler (frame), but Jacob is the speaker (voice). For stylometry of "Jacob's style", we need to identify where Jacob speaks.

### 2. Moroni 7-9 (Mormon's Content)

**Context:** Moroni includes his father Mormon's sermon and letters.

| Chapter | frame_narrator | voice | Content |
|---------|----------------|-------|---------|
| Moroni 7 | FRAME_MORONI | MORMON | Mormon's sermon |
| Moroni 8 | FRAME_MORONI | MORMON | Mormon's letter to Moroni |
| Moroni 9 | FRAME_MORONI | MORMON | Mormon's letter to Moroni |
| Moroni 10 | FRAME_MORONI | MORONI | Moroni's conclusion |

**Rationale:** Moroni compiled these chapters, but Mormon is the voice in chapters 7-9.

### 3. Jacob 5 (Zenos Allegory)

**Context:** Jacob quotes the prophet Zenos's allegory of the olive tree.

| Chapter | frame_narrator | voice | quote_source |
|---------|----------------|-------|--------------|
| Jacob 5 | FRAME_JACOB | ZENOS | ZENOS |

**Rationale:** Jacob is recording, but Zenos is the original author. For stylometry, this is Zenos's voice, not Jacob's.

### 4. Isaiah Blocks

| Location | frame_narrator | voice | quote_source |
|----------|----------------|-------|--------------|
| 1 Nephi 20-21 | FRAME_NEPHI | ISAIAH | ISAIAH |
| 2 Nephi 12-24 | FRAME_NEPHI | ISAIAH | ISAIAH |
| 2 Nephi 27 | FRAME_NEPHI | NEPHI | ISAIAH (paraphrase) |

**Note:** 2 Nephi 27 is Isaiah-dependent but Nephi's paraphrase, so voice=NEPHI but quote_source=ISAIAH.

### 5. 3 Nephi 12-14 (Sermon on the Mount)

| Chapters | frame_narrator | voice | quote_source |
|----------|----------------|-------|--------------|
| 3 Nephi 12-14 | FRAME_MORMON | JESUS_CHRIST | MATTHEW |

**Rationale:** Mormon records Jesus speaking words parallel to Matthew's Sermon on the Mount.

### 6. 3 Nephi 24-25 (Malachi)

| Chapters | frame_narrator | voice | quote_source |
|----------|----------------|-------|--------------|
| 3 Nephi 24-25 | FRAME_MORMON | MALACHI | MALACHI |

**Rationale:** Jesus quotes Malachi; Mormon records it.

---

## Usage Guidelines

### For Stylometric Analysis

**Primary analysis:** Use `voice` field with `quote_source=null` filter:

```python
# Get verses suitable for author-voice stylometry
stylometry_verses = [v for v in verses
                     if v['quote_source'] is None
                     and v['voice'] in ['NEPHI', 'JACOB', 'MORMON', 'MORONI']]
```

**With quotations excluded:**

```python
# Exclude all quotations
original_voice_verses = [v for v in verses if v['quote_source'] is None]
```

### For Editorial Layer Analysis

**Use `frame_narrator` field:**

```python
# Analyze editorial layers
nephi_plates = [v for v in verses if 'NEPHI' in v['frame_narrator']]
mormon_compilation = [v for v in verses if 'MORMON' in v['frame_narrator']]
```

### Sensitivity Analysis

Run stylometry both with and without:
1. Isaiah blocks (`quote_source='ISAIAH'`)
2. All quotations (`quote_source is not None`)
3. Jesus Christ's words (`voice='JESUS_CHRIST'`)

Report whether conclusions are robust to these exclusions.

---

## Validation

### External Cross-Check

Compared against bcgmaxwell "Writer" level annotations:
- **Agreement:** 97.5% at frame level
- **Cohen's κ:** 0.96 (almost perfect agreement)
- **Main difference:** 2 Nephi 6-10 (our Jacob attribution is intentional)

### Consistency Check

After v3.0 changes:
- 2 Nephi 6-10 now correctly has `frame_narrator=FRAME_NEPHI` (was FRAME_JACOB)
- `voice=JACOB` captures Jacob's speaking voice for stylometry
- Both layers are now independently meaningful

---

## Change History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-01 | Initial frame_narrator annotation |
| 2.0 | 2026-02-01 | Added embed_type, speaker, is_isaiah_block |
| 3.0 | 2026-02-01 | Added voice, quote_source; fixed 2 Ne 6-10 frame_narrator |

---

## References

- GPT-5.2 Pro consultation (2026-02-01): Dual-annotation recommendation
- bcgmaxwell comparison: 97.5% frame-level agreement, κ=0.96
- Project methodology: See METHODOLOGY.md
