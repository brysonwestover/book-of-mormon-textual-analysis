# Paratext Inclusion/Exclusion Policy

This document specifies the treatment of non-narrative textual elements (paratext) in the 1830 Book of Mormon for stylometric and textual analysis.

**Version:** 1.0.0
**Date:** 2026-02-01
**Status:** Pre-registered (locked before analysis)

---

## Overview

The 1830 Book of Mormon contains several types of paratext that may confound authorship analysis:

1. **Front matter**: Title page, copyright notice, preface
2. **Structural markers**: Book headings, chapter headings, chapter summaries
3. **End matter**: Witness testimonies

These elements have different authorship status, register, and function than the main narrative.

---

## Paratext Inventory

### Front Matter (Lines 1-177)

| Element | Lines | Content | Authorship Status |
|---------|-------|---------|-------------------|
| Title page | 1-31 | Extended title description | Attributed to Mormon/Moroni in narrative |
| Authorship line | 41-42 | "BY JOSEPH SMITH, JUNIOR, AUTHOR AND PROPRIETOR" | Joseph Smith (19th c.) |
| Publisher info | 45-49 | "PALMYRA: PRINTED BY E. B. GRANDIN..." | Publisher (19th c.) |
| Copyright notice | 52-96 | Legal registration text | Legal boilerplate (19th c.) |
| Preface | 105-151 | "TO THE READER—" lost pages explanation | Joseph Smith (19th c.) |
| First book heading | 154-158 | "THE FIRST BOOK OF NEPHI" + subheading | Structural marker |
| Chapter summary | 161-177 | Synopsis of 1 Nephi contents | Unknown (possibly editorial) |

### Structural Markers (Throughout)

| Element | Pattern | Example | Frequency |
|---------|---------|---------|-----------|
| Book headings | "THE [ORDINAL] BOOK OF [NAME]" | "THE FIRST BOOK OF NEPHI" | ~15 |
| Chapter headings | "CHAPTER [ROMAN]" or "CHAP. [ROMAN]" | "CHAPTER I." | ~239 (1830 divisions) |
| Chapter summaries | Italic synopsis paragraphs | Various | ~239 |
| "Reign and ministry" | Subheadings in some books | "HIS REIGN AND MINISTRY" | ~5 |

### End Matter (Lines 28563-28631)

| Element | Lines | Content | Authorship Status |
|---------|-------|---------|-------------------|
| "THE END" marker | 28563 | End of narrative | Structural |
| Three Witnesses | 28566-28598 | Testimony signed by Cowdery, Whitmer, Harris | Three Witnesses (19th c.) |
| Eight Witnesses | 28601-28628 | Testimony signed by 8 named individuals | Eight Witnesses (19th c.) |

---

## Inclusion/Exclusion Decisions

### DEFAULT: Exclude from Primary Stylometric Analysis

The following are **excluded** from stylometric analysis by default:

| Element | Rationale |
|---------|-----------|
| Title page (1-31) | Disputed authorship, formulaic register |
| Authorship/Publisher (41-49) | Unambiguously 19th-century, non-narrative |
| Copyright notice (52-96) | Legal boilerplate, non-authorial |
| Preface (105-151) | Unambiguously Joseph Smith, different register |
| Witness testimonies (28566-28628) | Different authors (witnesses), non-narrative |
| "THE END" marker | Non-content |
| Book headings | Structural markers, not narrative prose |
| Chapter headings | Structural markers, not narrative prose |

### INCLUDED in Primary Analysis

The following are **included** in stylometric analysis:

| Element | Rationale |
|---------|-----------|
| Narrative text (180-28560) | Main content attributed to claimed authors |
| Chapter summaries | TENTATIVELY included (see Sensitivity Analysis) |

### Sensitivity Analysis Required

| Element | Issue | Handling |
|---------|-------|----------|
| Chapter summaries | Authorship uncertain; may be editorial additions | Run analysis WITH and WITHOUT; report both |
| Title page | Claims to be from Mormon; could be original | Run separate analysis as supplementary |

---

## Implementation

### File Variants to Create

1. **`bom-1830-narrative-only.txt`**: Lines 180-28560 only (excludes ALL paratext)
2. **`bom-1830-with-summaries.txt`**: Lines 161-28560 (includes chapter summaries)
3. **`bom-1830-full.txt`**: All content (for reference/comparison)

### Segmentation Markers

For analyses requiring segment boundaries, mark:
- Book divisions at book heading lines
- Chapter divisions at chapter heading lines
- Exclude chapter summaries from segment word counts unless explicitly including them

### Metadata File

Create `segments.json` with:
```json
{
  "segments": [
    {
      "id": "1nephi-1",
      "book": "1 Nephi",
      "chapter": 1,
      "start_line": 180,
      "end_line": 468,
      "includes_summary": false
    }
  ]
}
```

---

## Rationale

### Why Exclude Front/End Matter

1. **Different register**: Preface and testimonies are in 19th-century declarative prose, not narrative
2. **Known authorship**: Preface is explicitly Joseph Smith; testimonies are explicit witnesses
3. **Confounding risk**: Including 19th-century paratext could bias toward H3 (19th-century composition)

### Why Include Chapter Summaries (With Sensitivity)

1. **Uncertain origin**: May be from source text or may be editorial
2. **Similar register**: Written in summarizing narrative voice
3. **Potential signal**: If editorial, could show translator/editor voice distinct from narrative

### Why Exclude Structural Markers

1. **Non-prose**: Book/chapter headings are labels, not sentences
2. **Tokenization artifacts**: Would introduce spurious capitalized words, numbers

---

## Limitations

1. **Line-based segmentation is approximate**: OCR text doesn't preserve exact 1830 page/line structure
2. **Chapter summary boundaries are fuzzy**: Some summaries blend into narrative
3. **This policy may need revision**: If analysis reveals important signals in excluded text

---

## Verification Checklist

Before analysis, verify:
- [ ] Paratext extraction script correctly identifies boundaries
- [ ] Line counts match expected ranges
- [ ] No narrative content accidentally excluded
- [ ] No paratext accidentally included
- [ ] Sensitivity variants generated and hashed

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-01 | Initial policy, pre-registered |
