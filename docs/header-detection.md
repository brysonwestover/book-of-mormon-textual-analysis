# Page Header Detection Documentation

This document describes the page header detection logic used in preprocessing the 1830 Book of Mormon text.

**Version:** 1.0.0
**Date:** 2026-02-01

---

## Overview

The 1830 printed Book of Mormon has running page headers that alternate between:
- Even pages: `[PAGE_NUMBER] BOOK NAME.`
- Odd pages: `BOOK NAME. [PAGE_NUMBER]`

These are non-content lines that should be removed or marked for analysis.

---

## Detection Patterns

### Regex Patterns Used

```python
PAGE_HEADER_PATTERNS = [
    # Pattern 1: Number first (even pages)
    r'^\d+\s+(?:FIRST |SECOND |THIRD |FOURTH )?BOOK OF (?:NEPHI|MORMON|ALMA|MOSIAH|JACOB|ENOS|JAROM|OMNI|HELAMAN|ETHER|MORONI)\.*\s*$',

    # Pattern 2: Number last (odd pages)
    r'^(?:FIRST |SECOND |THIRD |FOURTH )?BOOK OF (?:NEPHI|MORMON|ALMA|MOSIAH|JACOB|ENOS|JAROM|OMNI|HELAMAN|ETHER|MORONI)\.*\s+\d+\s*$',

    # Pattern 3: Words of Mormon, number first
    r'^\d+\s+WORDS OF MORMON\.*\s*$',

    # Pattern 4: Words of Mormon, number last
    r'^WORDS OF MORMON\.*\s+\d+\s*$',
]
```

### Pattern Explanation

| Component | Meaning |
|-----------|---------|
| `^\d+` | Line starts with one or more digits (page number) |
| `\s+` | One or more whitespace characters |
| `(?:FIRST |...)?` | Optional ordinal prefix for multi-book names |
| `BOOK OF (?:NEPHI|...)` | Book name alternatives |
| `\.*` | Optional period(s) |
| `\s*$` | Optional trailing whitespace, then end of line |

### Matching Behavior

- Case-insensitive matching (`re.IGNORECASE`)
- Requires entire line to match (anchored with `^` and `$`)
- Only matches complete book names (not partial matches within sentences)

---

## Results Summary

| Metric | Count |
|--------|-------|
| **Total headers detected** | 570 |
| **True positives** | 566 |
| **False positives** | 0 |
| **False negatives** | 4 |

---

## Sample Matches (True Positives)

First 5 matches:
```
Line 549: "6 FIRST BOOK OF NEPHI."
Line 602: "FIRST BOOK OF NEPHI. 7"
Line 656: "8 FIRST BOOK OF NEPHI."
Line 708: "FIRST BOOK OF NEPHI. 9"
Line 759: "10 FIRST BOOK OF NEPHI."
```

Last 5 matches:
```
Line 29232: "BOOK OF MORONI. 583"
Line 29279: "584 BOOK OF MORONI."
Line 29327: "BOOK OF MORONI. 585"
Line 29376: "586 BOOK OF MORONI."
Line 29424: "BOOK OF MORONI. 587"
```

**Observation:** Page numbers increment correctly (6,7,8,9,10... 583,584,585,586,587), confirming these are page headers.

---

## False Negative Analysis

### Missed Headers (4)

| Line | Text | Cause | Correct Page Number |
|------|------|-------|---------------------|
| 16906 | `BOOK OF ALMA. B47` | OCR: "3" → "B" | 347 |
| 21808 | `BOOK OF HELAMAN. 4A7` | OCR: "4" → "A" | 447 |
| 23092 | `47A BOOK OF NEPHI.` | OCR: "4" → "A" | 474 |
| 26615 | `BOOK OF ETHER. 5AT` | OCR: "4" → "A", "7" → "T" | 547 |

### Cause

OCR errors corrupted the page numbers, replacing digits with letters:
- `3` → `B`
- `4` → `A`
- `7` → `T`

### Impact

4 out of 574 total headers (0.7%) were not removed, leaving spurious lines in the cleaned text.

### Mitigation Options

1. **Manual correction**: Add these 4 lines to known corrections list
2. **Fuzzy matching**: Expand regex to allow single letter substitutions (risk: false positives)
3. **Document as limitation**: Accept 0.7% miss rate as acceptable

**Decision:** Document as limitation; 0.7% miss rate is acceptable and manual correction risks introducing errors.

---

## False Positive Analysis

### Potential False Positive Patterns

Lines that could theoretically match but shouldn't:

| Pattern | Example | Risk |
|---------|---------|------|
| Book division titles | "THE FIRST BOOK OF NEPHI." | No number - NOT matched |
| Narrative mentions | "...the book of Nephi..." | Lowercase, in sentence - NOT matched |
| Chapter headings | "CHAPTER I." | No "BOOK OF" - NOT matched |

### Verification

Searched for any matches that are NOT page headers:
- All 570 matches follow the `[number] BOOK NAME` or `BOOK NAME [number]` pattern
- No narrative content was incorrectly flagged
- No book division titles were incorrectly flagged

**Conclusion:** Zero false positives detected.

---

## Lines NOT Matched (Correctly Excluded)

### Book Division Titles

These are NOT page headers and were correctly excluded:

```
Line 154: "THE FIRST BOOK OF NEPHI."
Line 2899: "THE SECOND BOOK OF NEPHI."
Line 10793: "THE BOOK OF ALMA,"
Line 22033: "THE BOOK OF NEPHI,"
Line 25040: "THE BOOK OF NEPHI,"
Line 25262: "BOOK OF MORMON."
```

**Why excluded:** These don't have page numbers, or have different formatting ("THE" prefix, comma instead of period).

### Title Page

```
Line 4: "BOOK OF MORMON:"
```

**Why excluded:** No page number, has colon instead of period.

---

## Verification Commands

To verify header detection:

```bash
# Count detected headers
grep -c "<!-- PAGE_HEADER:" bom-1830-clean-headers-marked.txt
# Expected: 570

# Check for any remaining "BOOK OF" lines that might be headers
grep -E "^\d+\s+.*BOOK OF|BOOK OF.*\d+\s*$" bom-1830-clean.txt | wc -l
# Expected: 4 (the false negatives)

# List the false negatives
grep -E "^[A-Z0-9]+\s+.*BOOK OF|BOOK OF.*[A-Z][0-9]" bom-1830-clean.txt
```

---

## Audit Trail

Full list of matched headers stored in:
- `preprocessing_log.json` → `page_headers_matched` array
- Contains first 10 and last 10 matches with line numbers

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-01 | Initial documentation |
