# Source Text Documentation

This directory contains the Book of Mormon texts used for analysis.

## Available Texts

### 1. `book-of-mormon-1830-replica.txt` (PRIMARY)

**Source:** Internet Archive - "Book of Mormon 1830 Digital Replica" by Thomas A. Jenson
**URL:** https://archive.org/details/book-of-mormon-1830-digital-replica
**Download Date:** 2026-02-01

**Characteristics:**
- Faithful reproduction of the 1830 first edition (Palmyra, E.B. Grandin)
- Original chapter divisions (different from modern editions)
- No verse numbers (continuous prose, as originally published)
- Contains 1830 textual readings (e.g., "white and delightsome" not "pure and delightsome")
- Includes modern preface by Thomas A. Jenson (lines 1-43) and garbled TOC (lines 44-350)

**Known Issues:**
- Minor OCR artifacts present (e.g., "Neput" for "Nephi" on line 530)
- Some spacing/formatting inconsistencies from PDF-to-text conversion
- Modern creator's preface included at beginning (should be stripped for analysis)

**Statistics:**
- Lines: 29,550
- Words: ~277,000

**Verification:**
- Contains "white and a delightsome" (1830 reading) at line 6198
- Contains 1830 copyright page referencing "E. B. GRANDIN" and "PALMYRA" and "1830"

### 2. `book-of-mormon-1830-gutenberg.txt` (COMPARISON)

**Source:** Project Gutenberg, eBook #17
**URL:** https://www.gutenberg.org/ebooks/17
**Download Date:** 2026-02-01

**Characteristics:**
- Modern edition with verse numbers (e.g., "1:1", "1:2")
- Modern chapter divisions
- Post-1981 textual readings (e.g., "pure and delightsome" not "white and delightsome")
- Clean machine-readable text
- Includes Project Gutenberg header and license footer

**Known Issues:**
- NOT the 1830 edition despite Gutenberg's catalog listing
- Contains thousands of post-1830 textual changes
- Verse numbering system not present in 1830

**Statistics:**
- Lines: 31,568
- Words: ~279,000

**Use Case:**
- Comparison with modern LDS editions
- Secondary analysis where modern verse references are needed
- NOT for primary 1830 edition analysis

### 3. `book-of-mormon-modern-lds.txt` (MODERN REFERENCE)

**Source:** scriptures.nephi.org / beandog/lds-scriptures GitHub
**URL:** https://github.com/beandog/lds-scriptures
**Download Date:** 2026-02-01

**Characteristics:**
- Clean, structured modern LDS edition
- Tab-delimited format: `Reference\tText`
- Modern verse numbers (e.g., "1 Nephi 1:1")
- One verse per line (6,604 verses total)
- Post-1981 textual readings (confirmed: "pure and a delightsome")

**Statistics:**
- Verses: 6,604
- Words: 282,413

**Use Case:**
- **Preferred** for modern verse reference mapping
- Structured format ideal for alignment scripts
- Cleaner than Gutenberg (no header/footer noise)

**Verification:**
- Contains "pure and a delightsome" (modern reading) in 2 Nephi 30:6
- SHA-256: `1253158ebdc31fa410245382e7b11a3a5d75ae30f03e45fca360a92d5d35b1cc`

---

## Edition Differences

The 1830 edition differs from modern editions in several ways relevant to our analysis:

### Structural Differences

| Feature | 1830 Edition | Modern Editions |
|---------|--------------|-----------------|
| Verse numbers | None | Present |
| Chapter divisions | Original (fewer, longer) | Revised (more, shorter) |
| Book titles | Slightly different wording | Standardized |

### Textual Variants

Selected examples of changes between 1830 and modern editions:

| Location (Modern) | 1830 Reading | Modern Reading | Change Date |
|-------------------|--------------|----------------|-------------|
| 2 Nephi 30:6 | "white and a delightsome" | "pure and delightsome" | 1981 |
| 1 Nephi 11:18 | "the mother of God" | "the mother of the Son of God" | 1837 |
| 1 Nephi 13:40 | "the Lamb of God is the Eternal Father" | "the Lamb of God is the Son of the Eternal Father" | 1837 |
| Mosiah 21:28 | "king Benjamin" | "king Mosiah" | 1837 |

**Note:** A comprehensive list of textual variants is beyond the scope of this project. For detailed variant documentation, see:
- Royal Skousen, *The Book of Mormon: The Earliest Text* (Yale University Press, 2009)
- The Joseph Smith Papers Project: https://www.josephsmithpapers.org/

### Implications for Analysis

1. **Stylometric Analysis**: Most stylometric features (function words, syntax) are unaffected by post-1830 changes
2. **Theological Analysis**: Some theological formulations differ between editions (Christological passages especially)
3. **Linguistic Analysis**: The 1830 text preserves original grammar and spelling
4. **Segmentation**: Chapter boundaries differ; 1830 chapters should be used for authentic structure

---

## Recommended Usage

### For Primary Analysis (1830 Edition Focus)

Use the preprocessed files in `processed/` directory:

| File | Use Case |
|------|----------|
| `bom-1830-clean.txt` | **Primary analysis** - headers removed, OCR corrected |
| `bom-1830-clean-headers-marked.txt` | Traceability - headers marked with `<!-- PAGE_HEADER -->` |
| `bom-1830-clean-no-corrections.txt` | Validation - compare with/without OCR corrections |

### For Modern Reference Mapping

Use `book-of-mormon-modern-lds.txt` (preferred) or `book-of-mormon-1830-gutenberg.txt`:
- Mapping findings to modern verse references
- Comparing with modern LDS editions
- Readers need familiar reference system

The modern-lds file is preferred because:
- Cleaner format (tab-delimited, one verse per line)
- No header/footer noise
- Structured for programmatic access

### Preprocessing Scripts

| Script | Status | Purpose |
|--------|--------|---------|
| `preprocess_1830.py` | **Complete** | Clean 1830 text for analysis |
| `align_editions.py` | *Planned* | Map 1830 text to modern verse references |

---

## Processed Files (`processed/`)

The `processed/` subdirectory contains cleaned versions of the 1830 text ready for analysis.

### Preprocessing Applied

Run: 2026-02-01 via `scripts/preprocess_1830.py`

**Transformations:**
1. Removed modern preface (lines 1-43)
2. Removed garbled table of contents (lines 44-350)
3. Identified 570 page headers
4. Applied 3 documented OCR corrections

**OCR Corrections:**

| Original | Corrected | Context | Occurrences |
|----------|-----------|---------|-------------|
| `Neput` | `Nephi` | 1 Nephi 1:1 opening | 1 |
| `day 5` | `days` | 1 Nephi 1:1 | 1 |
| `utttered` | `uttered` | Double-t typo | 1 |

### Output Files

| File | Lines | Words | Description |
|------|-------|-------|-------------|
| `bom-1830-clean.txt` | 28,631 | 273,786 | Headers removed, corrections applied |
| `bom-1830-clean-headers-marked.txt` | 29,201 | 277,893 | Headers marked, corrections applied |
| `bom-1830-clean-no-corrections.txt` | 28,631 | 273,787 | Headers removed, no corrections |
| `preprocessing_log.json` | - | - | Full audit trail |

### Verification

All preprocessing is logged in `processed/preprocessing_log.json` with:
- Input file hash (SHA-256)
- Line-by-line transformation counts
- Correction details with approximate line numbers
- Timestamp and script version

---

## Provenance Chain

### 1830 Replica

```
Original 1830 Print (E.B. Grandin, Palmyra)
    ↓
Digital Scans (various libraries)
    ↓
Thomas A. Jenson's Digital Replica (2012)
    ↓
Internet Archive DJVU/TXT extraction
    ↓
This repository (2026-02-01)
```

### Gutenberg Modern

```
Modern LDS Edition (post-1981)
    ↓
David Widger's Digitization
    ↓
Project Gutenberg eBook #17 (2008, updated 2025)
    ↓
This repository (2026-02-01)
```

### scriptures.nephi.org Modern

```
Modern LDS Edition (post-1981)
    ↓
beandog/lds-scriptures project
    ↓
GitHub release 2020.12.08
    ↓
Extracted Book of Mormon verses (lines 31103-37706)
    ↓
This repository (2026-02-01)
```

---

## Hash Verification

For reproducibility, file hashes at time of acquisition:

```
# SHA-256 hashes (run: sha256sum filename)
book-of-mormon-1830-replica.txt:   da8c01d2b89b528b75780dba6bca6d038099d7b083a9b544efda092378b48902
book-of-mormon-1830-gutenberg.txt: 94e57fec8b145ab41f7c31cbf94a91b640b7c5b25b66ffeda57c7cca91a1c1c4
book-of-mormon-modern-lds.txt:     1253158ebdc31fa410245382e7b11a3a5d75ae30f03e45fca360a92d5d35b1cc
```

---

## Copyright Status

All source texts are in the **public domain**:

- The Book of Mormon text itself (pre-1930) is public domain in the United States
- Project Gutenberg releases under their license (essentially public domain)
- The 1830 digital replica is explicitly released for free distribution
- The beandog/lds-scriptures project is released under CC0/public domain dedication

Modern chapter headings and footnotes (not included in our texts) remain copyrighted by the LDS Church.

---

## Future Work

### Desired Improvements

1. **Validation Study**: Compare ~30 random pages against JSPP images to estimate remaining error rate
2. **Verse Alignment**: Create mapping between 1830 text positions and modern verse references
3. **Variant Documentation**: Document all textual variants that might affect analysis
4. **Additional Editions**: Consider adding Skousen's Earliest Text for comparison

### Known Resources Not Yet Acquired

- Joseph Smith Papers Project transcription of printer's manuscript (copyrighted, no bulk download)
- Royal Skousen's critical text work
- Original manuscript fragments (where available)

### Completed

- ✓ Preprocessing pipeline with full audit trail (`scripts/preprocess_1830.py`)
- ✓ Three output variants for different use cases
- ✓ Initial OCR correction (3 corrections documented)
- ✓ Modern LDS edition acquired for verse reference mapping

---

## Contact

For questions about source texts or to report errors, open an issue in the repository.
