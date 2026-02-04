# Data

This directory contains the primary text and reference materials for analysis.

**Last Updated:** 2026-02-01

## Directory Structure

```
data/
├── README.md           # This file
├── text/               # Book of Mormon source texts
│   ├── README.md       # Detailed text documentation
│   ├── book-of-mormon-1830-replica.txt    # PRIMARY: 1830 edition
│   └── book-of-mormon-1830-gutenberg.txt  # COMPARISON: Modern edition
├── reference/          # Control corpora for calibration
└── labels/             # Silver labels (Phase 1) / Gold labels (Phase 2)
```

## Primary Texts (`text/`)

### Currently Acquired

| File | Edition | Status | Use |
|------|---------|--------|-----|
| `book-of-mormon-1830-replica.txt` | 1830 First Edition | **PRIMARY** | Main analysis |
| `book-of-mormon-1830-gutenberg.txt` | Modern (post-1981) | Comparison | Modern verse mapping |

### Edition Selection Rationale

We use the **1830 first edition** as our primary text because:

1. It represents the original publication without later editorial changes
2. It preserves original grammar, spelling, and structure
3. It avoids introducing post-publication theological revisions
4. It is unambiguously in the public domain

See `text/README.md` for detailed documentation including:
- Provenance and download sources
- Known textual variants between editions
- OCR quality issues and limitations
- SHA-256 hashes for verification

## Reference Texts (`reference/`)

### Required Control Corpora

Per METHODOLOGY.md, we need the following for calibration:

| Corpus Type | Purpose | Status |
|-------------|---------|--------|
| Single-author 19th-c religious texts | Baseline for H1 | **Needed** |
| Multi-author compiled texts | Voice detection calibration | **Needed** |
| Pseudo-archaic imitations (18th-19th c) | Baseline for H5 | **Needed** |
| Translated multi-author (single translator) | Baseline for H4 | **Needed** |
| Diachronic corpora (COHA, EEBO) | Linguistic dating | **Pointers needed** |

### Corpus Instrument Specs

Each control corpus requires documentation of:
- Selection rationale
- Matching variables (genre, register, length)
- Known confounds
- Intended inferential target
- Licensing/copyright status

### Contributing Control Corpora

See CONTRIBUTING.md for how to suggest or contribute reference texts.

## Labels (`labels/`)

### Phase 1: Silver Labels

Single-annotator expert-coded labels for:
- Segment boundaries
- Discourse mode classifications
- Speaker identifications

**Status:** Not yet created

### Phase 2: Gold Labels

Multi-annotator validated labels (requires 3+ annotators).

**Status:** Future phase

## Data Provenance Requirements

All data files must have documented:
1. **Source URL** and access date
2. **Version/edition** information
3. **Hash** for integrity verification
4. **License/copyright** status
5. **Any preprocessing** applied

## Legal Status

### Book of Mormon

The Book of Mormon text is in the **public domain** in the United States:
- First published 1830 (pre-1930 = no U.S. copyright)
- Modern chapter headings and footnotes remain copyrighted (not included)

### Reference Texts

- Verify copyright status before adding
- Prefer public domain or openly licensed materials
- Document sources and licenses for all materials

## File Naming Convention

```
{text-name}-{edition/version}-{source}.{ext}

Examples:
  book-of-mormon-1830-replica.txt
  book-of-mormon-modern-gutenberg.txt
  kjv-bible-1611-gutenberg.txt
```

## Checksums

For reproducibility, all text files should have SHA-256 hashes recorded in their directory's README.md.

Verify with:
```bash
sha256sum data/text/*.txt
```
