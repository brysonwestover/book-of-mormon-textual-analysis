# Data Dictionary

**Version:** 1.0.0
**Date:** 2026-02-07

This document describes the schema for all machine-readable datasets in this project.

---

## 1. bom-voice-blocks.json

**Purpose:** Primary input for run-aggregated stylometric analysis
**Location:** `data/text/processed/bom-voice-blocks.json`

### Top-Level Structure
```json
{
  "metadata": { ... },
  "blocks": [ ... ]
}
```

### Metadata Fields
| Field | Type | Description |
|-------|------|-------------|
| `generated_at` | string (ISO 8601) | Timestamp of file generation |
| `source_file` | string | Input file used |
| `target_size` | integer | Target words per block |
| `total_blocks` | integer | Number of blocks generated |

### Block Object Fields
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `block_id` | string | Unique block identifier | `"MORMON_001"` |
| `run_id` | string | Contiguous narrator run ID | `"MORMON_run_1"` |
| `voice` | string | Narrator label | `"MORMON"`, `"NEPHI"`, `"MORONI"`, `"JACOB"` |
| `book` | string | Book of Mormon book name | `"1 Nephi"`, `"Mosiah"` |
| `chapters` | array[string] | Chapters covered | `["1", "2"]` |
| `verses` | string | Verse range | `"1:1-2:15"` |
| `text` | string | Raw text content | (full text) |
| `word_count` | integer | Number of words in block | `1023` |
| `target_size` | integer | Target block size | `1000` |
| `quote_status` | string | Quote classification | `"original"`, `"quote"`, `"mixed"` |

### Valid Values
- **voice:** `"MORMON"`, `"NEPHI"`, `"MORONI"`, `"JACOB"`
- **quote_status:** `"original"` (narrator's own words), `"quote"` (quoted speech), `"mixed"`
- **target_size:** `1000` (standard), `500` (sensitivity analysis)

---

## 2. bom-verses-annotated-v3.json

**Purpose:** Verse-level narrator annotations
**Location:** `data/text/processed/bom-verses-annotated-v3.json`

### Verse Object Fields
| Field | Type | Description |
|-------|------|-------------|
| `book` | string | Book name |
| `chapter` | integer | Chapter number |
| `verse` | integer | Verse number |
| `text` | string | Verse text |
| `voice` | string | Primary narrator |
| `voice_confidence` | string | Confidence level |
| `quote_speaker` | string | Speaker if quoted speech |
| `frame_narrator` | string | Narrator providing frame |

### Confidence Levels
- `"high"`: Clear textual markers
- `"medium"`: Contextual inference
- `"low"`: Scholarly interpretation

---

## 3. run-aggregated-results.json

**Purpose:** Output from run_aggregated_analysis.py
**Location:** `results/run-aggregated-results.json`

### Top-Level Structure
```json
{
  "metadata": { ... },
  "methodology_card": { ... },
  "data_summary": { ... },
  "primary_analysis": { ... },
  "exploratory_analyses": { ... },
  "sensitivity_analyses": { ... },
  "uncertainty": { ... },
  "figures": { ... },
  "limitations": { ... }
}
```

### Key Fields

#### metadata
| Field | Type | Description |
|-------|------|-------------|
| `generated_at` | string | ISO 8601 timestamp |
| `script_version` | string | Analysis script version |
| `random_seed` | integer | RNG seed used |
| `n_permutations_primary` | integer | Permutations for primary test |
| `quick_mode` | boolean | Whether quick mode was used |

#### primary_analysis
| Field | Type | Description |
|-------|------|-------------|
| `classes` | array[string] | Class labels |
| `n_samples` | integer | Number of runs |
| `balanced_accuracy` | float | Observed BA |
| `blocked_permutation.p_value` | float | Primary p-value |
| `blocked_permutation.null_mean` | float | Null distribution mean |
| `significant` | boolean | Whether p < 0.05 |

#### per_class_metrics
| Field | Type | Description |
|-------|------|-------------|
| `precision` | float | Precision for class |
| `recall` | float | Recall for class |
| `recall_ci_95` | array[float] | Wilson 95% CI |
| `f1` | float | F1 score |
| `support` | integer | Number of samples |

---

## 4. Feature Matrix (Runtime)

The feature matrix is constructed at runtime by `run_aggregated_analysis.py`:

### Dimensions
- **Rows:** 14 runs (12 in primary analysis)
- **Columns:** 171 function word frequencies

### Feature Values
- **Type:** float64
- **Units:** Per-1000-word frequency
- **Range:** [0, ~50] (varies by function word)
- **Missing:** None (zero if function word absent)

### Feature Names
See `FUNCTION_WORDS` list in `scripts/run_aggregated_analysis.py` lines 139-172.

---

## 5. Strata (Book-Based)

For blocked permutation testing, runs are grouped by primary book:

| Stratum | Narrators Present | Runs |
|---------|-------------------|------|
| 1 Nephi | NEPHI | 2 |
| 2 Nephi | NEPHI, JACOB | 3 |
| Jacob | JACOB | 1 |
| Mosiah | MORMON | 1 |
| Alma | MORMON | 1 |
| Helaman-Mormon | MORMON, MORONI | 4 |

**Note:** Runs are assigned to strata by their **primary book** (first book if run spans multiple books). Within each stratum, labels are permuted only among runs in that stratum.

### Primary Analysis Strata (excluding MORONI)
After excluding MORONI (n=2), the primary analysis has 12 runs across 6 book-strata.

---

## 6. Checksums

See `MANIFEST.md` for file integrity verification.
