# Data Provenance

**Version:** 1.0.0
**Date:** 2026-02-07
**Status:** Archival documentation

---

## 1. Source Text

### 1.1 Edition
- **Text:** The Book of Mormon (1830 First Edition)
- **Source:** Project Gutenberg
- **URL:** https://www.gutenberg.org/ebooks/17
- **Access Date:** 2026-01-28
- **License:** Public Domain (US)

### 1.2 Why 1830 Edition?
- Closest to original dictation
- Minimizes editorial modernization
- Consistent with prior stylometric studies
- Public domain, freely available

### 1.3 Known Differences from Other Editions
- Spelling and punctuation differ from modern LDS editions
- Some grammatical constructions normalized in later editions
- Versification differs slightly from current chapter/verse system
- See Skousen's "The Book of Mormon: The Earliest Text" for critical apparatus

---

## 2. Preprocessing Pipeline

### 2.1 Text Normalization
| Step | Script | Description |
|------|--------|-------------|
| 1 | `preprocess_1830.py` | Initial cleaning, paratext removal |
| 2 | `dehyphenate.py` | Resolve line-break hyphens |
| 3 | `normalize_unicode.py` | NFC normalization, smart quotes |
| 4 | `segment_bom.py` | Segment into chapters/verses |

### 2.2 Normalization Details
- **Encoding:** UTF-8
- **Line endings:** Unix (LF)
- **Unicode:** NFC normalized
- **Case:** Preserved (lowercased only during feature extraction)
- **Punctuation:** Preserved
- **Numerals:** Preserved as-is

### 2.3 Preprocessing Logs
- `data/text/processed/preprocessing_log.json`
- `data/text/processed/dehyphenation_log.json`
- `data/text/processed/normalization_log.json`

---

## 3. Narrator Segmentation

### 3.1 Segmentation Schema
Narrator segments are contiguous spans of text attributed to a single narrator based on textual framing (e.g., "I, Nephi", "Mormon abridging").

| Field | Description |
|-------|-------------|
| `voice` | Narrator label (MORMON, NEPHI, MORONI, JACOB) |
| `run_id` | Unique identifier for contiguous narrator run |
| `book` | Book within Book of Mormon |
| `verses` | Verse range covered |

### 3.2 Segmentation Sources
- Primary: Textual self-identification markers
- Secondary: Scholarly consensus on narrative framing
- Reference: Hardy, Grant. "Understanding the Book of Mormon" (2010)

### 3.3 Segmentation Files
- `data/text/processed/bom-verses-annotated-v3.json` - Verse-level annotations
- `data/text/processed/bom-voice-blocks.json` - Aggregated blocks for analysis

### 3.4 Narrator Distribution
| Narrator | Runs | Blocks | Words | In Primary? |
|----------|------|--------|-------|-------------|
| JACOB | 3 | 12 | 10,839 | Yes |
| MORMON | 4 | 171 | 174,869 | Yes |
| MORONI | 2 | 22 | 22,031 | No (n=2) |
| NEPHI | 5 | 39 | 38,481 | Yes |
| **Total** | **14** | **244** | **246,220** | 12 |

### 3.5 Exclusions
- MORONI excluded from primary analysis (only 2 runs)
- Quoted material excluded (marked `quote_status: "quote"`)
- Blocks < target_size excluded from some analyses

---

## 4. Feature Extraction

### 4.1 Function Words
171 function words used for stylometric analysis, including:
- Articles: a, an, the
- Pronouns: I, me, my, we, us, our, you, he, she, it, they, etc.
- Prepositions: in, on, at, by, for, with, about, etc.
- Conjunctions: and, but, or, nor, yet, so, etc.
- Auxiliary verbs: be, have, do, will, would, shall, should, etc.
- Archaic forms: ye, thee, thou, thy, hath, doth, etc.

Full list in `scripts/run_aggregated_analysis.py` lines 139-172.

### 4.2 Feature Computation
- **Tokenization:** Lowercase, word boundary regex (`\b[a-z]+\b`)
- **Aggregation:** Sum raw counts across blocks within run, then convert to per-1000-word frequencies
- **Scaling:** StandardScaler fit inside each CV fold (prevents leakage)

---

## 5. Derived Datasets

### 5.1 Primary Analysis Input
- **File:** `data/text/processed/bom-voice-blocks.json`
- **Structure:** Array of block objects with text, voice, run_id, book, word counts
- **Checksum:** See `MANIFEST.md`

### 5.2 Run-Level Aggregation
The analysis script (`run_aggregated_analysis.py`) aggregates blocks to runs at runtime:
- 244 blocks → 14 runs
- Features summed within run, then normalized to frequencies
- Zero-word runs excluded with warning

---

## 6. Integrity Verification

### 6.1 Checksums
See `MANIFEST.md` for SHA-256 checksums of all data files.

### 6.2 Reconstruction
To reconstruct the analysis input from source:
```bash
# 1. Download 1830 text from Project Gutenberg
# 2. Run preprocessing pipeline
python scripts/preprocess_1830.py
python scripts/dehyphenate.py
python scripts/normalize_unicode.py
python scripts/segment_bom.py
python scripts/annotate_verses_v2.py
python scripts/aggregate_blocks_v2.py
```

---

## 7. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-07 | Initial archival documentation |
