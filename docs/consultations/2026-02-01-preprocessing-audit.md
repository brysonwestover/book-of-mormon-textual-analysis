# GPT Consultation: Preprocessing Data Audit

**Date:** 2026-02-01
**Type:** Data Integrity Audit
**Model:** gpt-5.2-pro (reasoning_effort: high)

---

## Prompt Sent

```
Please review the following data acquisition and preprocessing work for a Book of Mormon textual analysis project. Critique the process for methodological rigor, identify any gaps or concerns, and confirm whether the integrity of the data pipeline is sound.

## Project Context
This is an LLM-assisted textual analysis examining Book of Mormon authorship claims. The methodology was previously reviewed and endorsed by you (GPT-5.2 Pro) with emphasis on: operational definitions, control corpora, translation layer modeling, pre-registration, and full audit trails.

## Source Text Decisions

### Primary Text Selected: 1830 First Edition
- Source: Internet Archive - "Book of Mormon 1830 Digital Replica" by Thomas A. Jenson
- URL: https://archive.org/details/book-of-mormon-1830-digital-replica
- Rationale: Original publication without later editorial changes, preserves original grammar/spelling, avoids post-publication theological revisions, unambiguously public domain
- SHA-256: da8c01d2b89b528b75780dba6bca6d038099d7b083a9b544efda092378b48902

### Alternatives Considered and Rejected:
1. **Project Gutenberg eBook #17**: Discovered to be mislabeled - actually modern edition (post-1981) with verse numbers and textual changes. Verified by finding "pure and delightsome" (1981 change) instead of "white and delightsome" (1830 original).

2. **Joseph Smith Papers Project**: Has scholarly transcription but copyrighted, no bulk download available. Noted as potential future validation source.

3. **BYU Religious Studies Center**: Restricted use policies incompatible with open research.

### Modern Edition for Reference Mapping:
- Source: scriptures.nephi.org / beandog/lds-scriptures GitHub
- Format: Tab-delimited, one verse per line (6,604 verses)
- Verified modern reading: "pure and a delightsome" in 2 Nephi 30:6
- SHA-256: 1253158ebdc31fa410245382e7b11a3a5d75ae30f03e45fca360a92d5d35b1cc

## Preprocessing Pipeline

### Script: preprocess_1830.py (292 lines)

### Transformations Applied:
1. **Removed modern preface** (lines 1-43): Jenson's modern introduction
2. **Removed garbled TOC** (lines 44-350): OCR-damaged table of contents
3. **Page header handling**: Found 570 page headers matching patterns like "PAGE_NUM BOOK_NAME"
4. **OCR corrections**: 3 documented corrections:
   - "Neput" → "Nephi" (1 Nephi 1:1 opening, 1 occurrence)
   - "day 5" → "days" (1 Nephi 1:1, 1 occurrence)
   - "utttered" → "uttered" (double-t typo, 1 occurrence)

### Output Files Created:
| File | Lines | Words | Description |
|------|-------|-------|-------------|
| bom-1830-clean.txt | 28,631 | 273,786 | Headers removed, corrections applied |
| bom-1830-clean-headers-marked.txt | 29,201 | 277,893 | Headers marked with HTML comments |
| bom-1830-clean-no-corrections.txt | 28,631 | 273,787 | No OCR corrections (for validation) |

### Audit Trail:
- Full preprocessing log saved as JSON with:
  - Input file SHA-256 hash
  - Line counts before/after each transformation
  - Correction details with approximate line numbers
  - Timestamp and script version

## Documentation
- README.md updated with provenance chains, hashes, edition differences
- All decisions documented with rationale

## Outstanding Items
1. Validation study: Compare ~30 random pages against JSPP images to estimate residual OCR error rate
2. Edition alignment script: Map 1830 text positions to modern verse references

## Questions for Review
1. Is the source text selection methodologically defensible?
2. Are the preprocessing transformations appropriate and well-documented?
3. Are there any gaps in the audit trail?
4. What concerns should be documented as limitations?
5. Is anything missing before proceeding to analysis modules?
```

---

## GPT-5.2 Pro Response

### Overall Assessment

> Your acquisition choices and preprocessing approach are directionally strong: you (a) selected a clearly-defined base edition, (b) rejected mislabeled/contaminated alternatives using a concrete diagnostic, (c) kept a no-corrections variant for validation, and (d) created an audit trail with transformation-by-transformation counts. Those are the right components for a defensible pipeline.
>
> That said, I **cannot confirm end-to-end integrity is "sound" in the strict sense** without inspecting (i) the *exact* raw input text file you downloaded from IA, (ii) the preprocessing script, and (iii) the JSON log outputs (plus hashes for all derived artifacts).

### Source Text Selection: Defensible with Caveats

**Strengths:**
- 1830 first edition is coherent choice
- Content-based falsification check (white vs pure) was correct approach

**Caveats:**
1. "1830 edition" ≠ "earliest text" - embeds compositor interventions, copy errors, printer normalization
2. IA "digital replica" may include OCR artifacts, silent cleanup, normalization
3. Single-source dependency risk - need triangulation

**Conclusion:** Defensible for open pipeline, but document as analysis of "1830 print/OCR transcription layer"

### Preprocessing: Mostly Appropriate, Missing Key Decisions

**What was done well:**
- Removing modern intro/garbled TOC
- Header stripping
- Minimal OCR corrections + no-corrections variant

**Main methodological concerns:**
1. **Header detection risk** - need exact regex, near-misses, manual review of ambiguous cases
2. **Hyphenation/line-wrap not addressed** - HIGH RISK for stylometry (e.g., `com-\nmandments`)
3. **Unicode normalization not specified** - curly vs straight quotes, em-dashes
4. **Scope control missing** - Title page, Witnesses, chapter headings not specified
5. **OCR corrections policy not formalized** - were corrections image-verified?

### Audit Trail: Good Foundation, Missing Elements

**What's strong:**
- Input file hash
- Line counts before/after
- Corrections with line numbers
- Timestamp + version

**What to add:**
1. Hashes for all outputs
2. Git commit hash of script
3. Environment capture (Python version, OS, locale)
4. Machine-checkable provenance chain
5. Transformation diffs for destructive steps (preserve removed blocks)

### Limitations to Document

- OCR residual error (unknown rate, likely non-uniform)
- Edition-specific compositor effects
- Selection bias in OCR corrections
- Non-authorial paratext contamination risk
- Alignment uncertainty (1830 → modern verse mapping)

### Gate Checks Before Analysis

**A. Validation gate:**
- ~30 random pages against JSPP images
- Define what counts as "error"
- Error rate per 1,000 tokens
- Stratified sampling

**B. Segmentation/paratext policy:**
- Lock include/exclude decisions
- Label segments with metadata

**C. Hyphenation decision:**
- Detect and quantify
- Correct or justify not correcting

**D. Artifact manifest:**
- All filenames, hashes, creation times, parent artifacts, script commit

### Answers to Questions

1. **Source text defensible?** Yes, for 1830 printed edition analysis with stated caveats
2. **Preprocessing appropriate?** Mostly - gaps in hyphenation, Unicode, paratext policy
3. **Audit trail gaps?** Yes - output hashes, script commit, environment, removed blocks
4. **Limitations to document?** OCR error, compositor effects, paratext contamination, alignment uncertainty
5. **Missing before analysis?** Validation gate, segmentation policy, hyphenation decision, artifact manifest

---

## Action Items

| Priority | Item | Status |
|----------|------|--------|
| Critical | Add output hashes | Task #1 |
| Critical | Investigate hyphenation | Task #2 |
| Critical | Document paratext policy | Task #3 |
| High | Preserve removed blocks | Task #4 |
| High | Document header regex | Task #5 |
| Medium | Environment capture | Task #1 |
| Medium | Unicode normalization policy | TBD |

---

## Decisions Made

1. Accept all identified gaps as legitimate concerns
2. Address critical gaps before proceeding to analysis
3. Document 1830 print layer limitation explicitly
4. Create artifact manifest with full hashes
