# Project Status: Book of Mormon Textual Analysis

**Last Updated:** 2026-02-06
**Status:** Phase 2.0 & 2.D Complete; Phase 2.A In Progress
**Current Step:** Awaiting Robustness Analysis Completion (AWS)
**Pre-Registration:** [OSF DOI 10.17605/OSF.IO/4W3KH](https://osf.io/4W3KH)
**Repository:** https://github.com/brysonwestover/book-of-mormon-textual-analysis

---

## Quick Start for New Claude Code Instance

If resuming this project in a new session, read these files in order:

1. **This file** (`PROJECT-STATUS.md`) - Current status and next steps
2. `METHODOLOGY.md` - Research framework and hypotheses
3. `docs/decisions/SEGMENTATION-SUMMARY.md` - Voice annotation decisions
4. `docs/decisions/block-derivation-strategy.md` - GPT-approved analysis approach

---

## Phase 2 Status (February 2026)

### Phase 2.0: Primary Confirmatory Analysis ✅ COMPLETE
**Script:** `scripts/run_classification_v3.py`
**Results:** `results/classification-results-v3.json`

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Balanced Accuracy | **24.2%** | Near 25% chance baseline |
| Permutation p-value | **0.177** | NOT significant |
| TOST Equivalence | p = 0.06 | Near-equivalence to chance |
| Bayes Factor BF₀₁ | 2.85 | Weak evidence for null |

**Methodology:** Logistic regression with LOWO (leave-one-work-out) CV, 100,000 group-level permutations, run-weighted balanced accuracy.

### Phase 2.D: Garnett Calibration ✅ COMPLETE
**Purpose:** Validate that stylometry can detect authorial signal through a single translator's voice.
**Script:** `scripts/run_garnett_analysis_optimized.py`

| Analysis | Accuracy | Chance | p-value | Status |
|----------|----------|--------|---------|--------|
| Novels Only (Primary) | **58.2%** | 33.3% | **0.0016** | ✅ Significant |
| Full Corpus | 54.4% | 25% | **0.0001** | ✅ Significant |

**Key Finding:** Method validated - authorial signal IS detectable through translation. The BoM null result is therefore **informative**, not inconclusive.

### Phase 2.A: Robustness Testing 🔄 IN PROGRESS
**Script:** `scripts/run_robustness_optimized.py`
**Status:** Running on AWS c7i.8xlarge (~80% complete as of Feb 6)

| Variant | Description | Preliminary |
|---------|-------------|-------------|
| A1 | 500-word blocks | ~28% (near chance) |
| A2 | 2000-word blocks | ~19% (below chance) |
| A3 | Include quotations | ~31% (slightly above) |
| A4 | Character 3-grams | ~26% (near chance) |
| A5 | Combined features | ~25% (near chance) |
| A6 | SVM classifier | ~24% (near chance) |

### OSF Pre-Registration & Amendments
- **Pre-registration:** [DOI 10.17605/OSF.IO/4W3KH](https://osf.io/4W3KH) (Feb 3, 2026)
- **Amendment #1:** Robustness testing deviations (Feb 5, 2026)
- **Amendment #2:** Bootstrap CI feasibility criterion (Feb 5, 2026)

See `docs/osf-preregistration.md`, `docs/osf-amendment-2026-02-05.md`, `docs/osf-amendment-2-2026-02-05.md`

---

## Phase 1 Summary (Historical)

Phase 1 established the foundation: text acquisition, voice annotation (v3 schema), block derivation, and initial exploratory analyses. See below for details.

---

## Project Overview

**Research Question:** Do the claimed ancient authors of the Book of Mormon (Nephi, Jacob, Mormon, Moroni, etc.) exhibit detectably different writing styles?

**Approach:** Stylometric analysis using computational linguistics methods, with rigorous methodology validated by GPT-5.2 Pro consultation.

**Hypotheses (from METHODOLOGY.md):**
- H1: Single modern author
- H2: Multiple modern collaborators
- H3: Single author mimicking multiple voices
- H4: Ancient text through single translator
- H5: Modern text using ancient sources
- H0: Null (insufficient signal)

---

## Completed Work

### 1. Text Acquisition & Processing ✅

| Item | Status | Location |
|------|--------|----------|
| 1830 first edition | ✅ Acquired | `data/text/book-of-mormon-1830-replica.txt` |
| OCR corrections | ✅ Applied | Logged in `preprocessing_log.json` |
| Clean text | ✅ Generated | `data/text/processed/bom-1830-clean.txt` |
| SHA-256 hash | ✅ Documented | `data/text/README.md` |

**Source:** Internet Archive (Thomas A. Jenson digital replica)
**Hash:** `da8c01d2b89b528b75780dba6bca6d038099d7b083a9b544efda092378b48902`

### 2. Voice Annotation (v3.0) ✅

| Item | Status | Location |
|------|--------|----------|
| Verse-level annotation | ✅ Complete | `data/text/processed/bom-verses-annotated-v3.json` |
| Dual-layer schema | ✅ Implemented | frame_narrator + voice + quote_source |
| External validation | ✅ 97.5% agreement | κ=0.96 with bcgmaxwell |

**Schema (GPT-5.2 Pro "Option C" recommendation):**
```
frame_narrator: Who compiled the plates (editorial layer)
voice: Who is speaking (surface voice for stylometry)
quote_source: Source of quotation (ISAIAH, ZENOS, MALACHI, MATTHEW, null)
```

**Voice Distribution (6,604 verses):**
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

**Special Cases Implemented:**
- 2 Nephi 6:2-10:25 → frame=NEPHI, voice=JACOB
- Moroni 7-9 → frame=MORONI, voice=MORMON
- Jacob 5 → voice=ZENOS, quote_source=ZENOS
- 3 Nephi 12-14 → voice=JESUS_CHRIST, quote_source=MATTHEW
- Isaiah blocks → voice=ISAIAH, quote_source=ISAIAH

### 3. Control Corpora ✅

| Corpus | Words | Purpose | Location |
|--------|-------|---------|----------|
| Finney | 202,757 | Single-author 19th-c baseline | `data/reference/processed/` |
| Late War | 56,636 | Pseudo-archaic baseline | `data/reference/processed/` |
| Josephus | 546,148 | Single-translator multi-author | `data/reference/processed/` |
| KJV Bible | 192,115 | Multi-author calibration | `data/reference/processed/` |

### 4. Methodology & Documentation ✅

| Document | Purpose | Status |
|----------|---------|--------|
| `METHODOLOGY.md` | Research framework (v2.0) | ✅ Complete |
| `LIMITATIONS.md` | Explicit scope boundaries | ✅ Complete |
| `docs/decisions/SEGMENTATION-SUMMARY.md` | Voice annotation overview | ✅ Complete |
| `docs/decisions/voice-annotation-schema-v3.md` | Full v3 specification | ✅ Complete |
| `docs/decisions/segment-annotation-decisions.md` | 6 decision points resolved | ✅ Complete |
| `docs/decisions/block-derivation-strategy.md` | GPT-approved analysis approach | ✅ Complete |

### 5. Environment ✅

| Item | Status | Location |
|------|--------|----------|
| Python 3.12.3 | ✅ Available | System |
| Virtual environment | ✅ Created | `venv/` |
| Dependencies installed | ✅ Complete | anthropic, openai, python-dotenv |
| Lock file | ✅ Generated | `requirements-lock.txt` |

---

## Key Decisions Made (with Rationale)

### Decision 1: Dual-Layer Annotation
**Choice:** Separate frame_narrator from voice
**Rationale:** GPT-5.2 Pro identified that conflating compiler with speaker creates methodological problems for stylometry. The dual-layer approach allows filtering for "who is actually speaking" independent of "who wrote the plates."
**Documentation:** `docs/decisions/voice-annotation-schema-v3.md`

### Decision 2: Verse-Level as Authoritative
**Choice:** Use verse-level annotation, derive blocks programmatically
**Rationale:** Verses provide clean narrator boundaries. Fixed-length blocks that cross narrator transitions create label noise. GPT approved "Option C" approach.
**Documentation:** `docs/decisions/block-derivation-strategy.md`

### Decision 3: Quote Source Filtering
**Choice:** Tag quotations (Isaiah, Zenos, Malachi, Matthew) for exclusion
**Rationale:** Quoted material reflects source author style, not Book of Mormon narrator style. Must exclude for valid authorship attribution.
**Filter:** `quote_source == null` for primary analysis

### Decision 4: Primary Voices
**Choice:** Focus on MORMON, NEPHI, MORONI, JACOB (≥100 verses each)
**Rationale:** Statistical power requires sufficient samples. Small narrators (Enos, Jarom, Omni) included for exploratory analysis only with explicit caveats.

### Decision 5: Grouped Cross-Validation
**Choice:** CV splits by voice_run (contiguous same-voice passages)
**Rationale:** Prevents train/test leakage from near-duplicate text. GPT strongly recommended this approach.

---

## Next Steps: Stylometric Analysis

### Step 1: Implement Block Derivation Script ✅ COMPLETE
**Script:** `scripts/derive_blocks.py`
**Input:** `bom-verses-annotated-v3.json`
**Output:** `data/text/processed/bom-voice-blocks.json`

**Results:**
- 23 voice runs created (17 original, 6 quoted)
- 917 total blocks generated across 3 target sizes (500, 1000, 2000 words)

**Primary Analysis Data (1000-word, original voice):**
| Voice | Blocks | Words |
|-------|--------|-------|
| MORMON | 171 | 174,463 |
| NEPHI | 39 | 38,431 |
| MORONI | 22 | 21,998 |
| JACOB | 12 | 10,814 |
| Total (4 major) | 244 | 245,706 |

Algorithm implemented per GPT consultation:
1. ✅ Created voice runs (contiguous verses with same voice + quote_source status)
2. ✅ Assigned stable `run_id` to each run
3. ✅ Generated non-overlapping blocks within runs
4. ✅ Stored full provenance (start/end verses, word count, voice, run_id)

### Step 2: Feature Extraction ✅ COMPLETE
**Script:** `scripts/extract_features.py`
**Input:** `bom-voice-blocks.json`
**Output:** `data/text/processed/bom-stylometric-features.json`

**Results:**
- 244 blocks processed (1000-word, original voice, 4 major voices)
- 2,753 features extracted per block:
  - 163 function words (MFW - Burrows Delta style)
  - 2,574 character n-grams (2-4 grams)
  - 10 word length distribution features
  - 3 sentence-level features
  - 3 vocabulary richness features

### Step 3: Classification Experiments ✅ COMPLETE (REVISED)

#### v1 Analysis (FLAWED - superseded)
**Script:** `scripts/run_classification.py`
**Issues identified by GPT-5.2 Pro:**
- Wrong baseline (used 25% random instead of 70% majority class)
- Content words in features (topic/genre confound)
- No permutation test for significance

#### v2 Analysis (CORRECTED) ✅
**Script:** `scripts/run_classification_v2.py`
**Output:** `results/classification-results-v2.json`, `results/classification-report-v2.md`
**Documentation:** `docs/decisions/classification-methodology-corrections.md`

**Methodology corrections applied:**
1. Function words only (169 features, content-suppressed)
2. Proper baseline (70% majority class)
3. Class weights for imbalance
4. Permutation test for significance
5. Downsampled experiments (apples-to-apples)

**Results:**
| Metric | Value | Baseline | Interpretation |
|--------|-------|----------|----------------|
| Balanced Accuracy | **21.6%** | 25% random | BELOW chance |
| Macro F1 | 0.236 | 0.206 trivial | Barely above trivial |
| Permutation p-value | **1.0** | 0.05 threshold | NOT significant |
| Downsampled Macro F1 | 0.115 | - | Near zero |

**Key finding:** NO statistically significant stylistic differentiation detected using function words alone. The initial "signal" was topic/genre, not authorial style.

**Per-class performance:**
- Jacob: 0% recall (never correctly classified)
- Moroni: 0% recall (never correctly classified)
- Nephi: 31% recall (poor)
- Mormon: 56% recall (drives all results)

### Step 4: Additional Diagnostic Analyses ✅ COMPLETE
**Script:** `scripts/run_additional_analyses.py`
**Output:** `results/additional-analyses.json`, `results/additional-analyses-report.md`

| Analysis | Result | Interpretation |
|----------|--------|----------------|
| Unmasking | Started at 20.1% (below chance) | No initial separation to unmask |
| Mormon vs Nephi | 65.9% balanced acc | Only pair with any separation |
| All other pairs | At or below chance | No distinguishable patterns |
| Cohen's d | 0.440 | Small effect - weak differentiation |
| Within-class | Jacob most consistent | Mormon most variable |

### Step 5: Control Corpora Baselines ✅ COMPLETE
**Script:** `scripts/run_control_analysis.py`
**Output:** `results/control-analysis.json`, `results/control-analysis-report.md`

| Corpus | Expected | Result | Interpretation |
|--------|----------|--------|----------------|
| KJV Bible | Multi-author separation | **93.2%** balanced acc (vs 16.7% chance) | ✓ Method works |
| Finney | No separation | 69.5% acc, p=0.564 (not significant) | ✓ No spurious signal |

**Method Validation:** The stylometric method successfully detects multi-authorship in KJV while avoiding spurious separation in single-author Finney. This validates that the null result on BoM is meaningful.

---

## Overall Findings

### Primary Result
**NO statistically significant stylistic differentiation** detected among Book of Mormon narrators using function-word stylometry.

| Metric | BoM Narrators | KJV Authors |
|--------|---------------|-------------|
| Balanced Accuracy | 21.6% | 93.2% |
| vs Chance Baseline | -3.4% | +76.5% |
| Permutation p-value | 1.0 | N/A |

### Interpretation
The null result is **validated** by control corpora analysis:
1. The method CAN detect multi-authorship (KJV: 93.2% vs 16.7% chance)
2. The method does NOT create spurious separation (Finney: not significant)
3. The BoM English layer does not exhibit narrator-specific function-word patterns

### Hypotheses Addressed
- **H1 (Single modern author):** Consistent with null result
- **H3 (Intentional voice mimicry):** Consistent with null result
- **H4 (Ancient text, single translator):** Consistent with null result (translation homogenizes)
- **H0 (Insufficient signal):** Primary finding

### What This Does NOT Prove
Per `docs/decisions/expected-results-if-ancient.md`:
- Does NOT prove single authorship
- Does NOT falsify ancient multi-authorship (translation layer could erase signal)
- Shows only that the English layer lacks distinguishing function-word patterns

### Step 6: Sensitivity Analysis (Optional)
- Vary block sizes (500, 1000, 2000)
- With/without quotations
- Report robustness of conclusions

---

## File Structure

```
book-of-mormon-textual-analysis/
├── PROJECT-STATUS.md          ← YOU ARE HERE
├── METHODOLOGY.md             ← Research framework
├── LIMITATIONS.md             ← Scope boundaries
├── requirements.txt           ← Dependencies (minimums)
├── requirements-lock.txt      ← Dependencies (exact versions)
├── venv/                      ← Virtual environment
├── data/
│   ├── text/
│   │   ├── README.md          ← Source documentation
│   │   ├── book-of-mormon-1830-replica.txt
│   │   └── processed/
│   │       ├── bom-1830-clean.txt
│   │       ├── bom-verses-annotated-v3.json  ← VERSE-LEVEL DATA
│   │       ├── bom-voice-blocks.json         ← BLOCK-LEVEL DATA (Step 1)
│   │       ├── bom-stylometric-features.json ← FEATURE DATA (Step 2)
│   │       └── VERSE-ANNOTATION-v3.md
│   └── reference/
│       └── processed/         ← Control corpora
├── docs/
│   └── decisions/
│       ├── SEGMENTATION-SUMMARY.md
│       ├── voice-annotation-schema-v3.md
│       ├── block-derivation-strategy.md
│       └── segment-annotation-decisions.md
├── scripts/
│   ├── preprocess_1830.py
│   ├── add_voice_annotation.py
│   ├── derive_blocks.py           ← COMPLETE (Step 1)
│   ├── extract_features.py        ← COMPLETE (Step 2)
│   ├── run_classification.py      ← v1 (superseded)
│   ├── run_classification_v2.py   ← COMPLETE (Step 3)
│   ├── run_additional_analyses.py ← COMPLETE (Step 4)
│   └── run_control_analysis.py    ← COMPLETE (Step 5)
├── analysis/
│   └── modules/
└── results/
    ├── classification-results-v2.json
    ├── classification-report-v2.md
    ├── additional-analyses.json
    ├── additional-analyses-report.md
    ├── control-analysis.json
    └── control-analysis-report.md
```

---

## GPT-5.2 Pro Consultations

| Date | Topic | Outcome | Documentation |
|------|-------|---------|---------------|
| 2026-02-01 | Methodology critique | 6 decision points identified | segment-annotation-decisions.md |
| 2026-02-01 | Dual-layer annotation | Option C approved | voice-annotation-schema-v3.md |
| 2026-02-01 | 2 Ne 6-10 attribution | frame=NEPHI, voice=JACOB | voice-annotation-schema-v3.md |
| 2026-02-01 | Block derivation (Option C) | Approved with algorithm | block-derivation-strategy.md |
| 2026-02-01 | v1 Classification critique | Wrong baseline, feature issues | classification-methodology-corrections.md |
| 2026-02-01 | Expected results if ancient | Translation layer problem | expected-results-if-ancient.md |

---

## Resuming This Project

**For a new Claude Code instance:**

1. Read this file first: `PROJECT-STATUS.md`
2. Activate the virtual environment: `source venv/bin/activate`
3. Read the results reports in `results/` directory
4. See `docs/decisions/expected-results-if-ancient.md` for interpretation framework

**Phase 1 is COMPLETE. Phase 2.0 and 2.D are COMPLETE. Phase 2.A in progress.**

---

## Suggested Next Steps (Phase 2)

### Option A: Robustness Testing (Recommended First)
1. **Sensitivity analysis** - Test if null result holds across block sizes (500, 1000, 2000 words)
2. **With/without quotations** - Include quoted material to see if it changes results
3. **Different feature sets** - Try character n-grams, POS tags, or other non-function-word features

### Option B: Genre-Controlled Analysis
Test whether genre/topic confounds explain results:
1. Compare **sermon vs sermon** across narrators
2. Compare **narrative vs narrative** across narrators
3. If separation appears within genre, topic was the confound

### Option C: Alternative Methods
1. **Burrows' Delta** - Classic stylometric distance measure
2. **Rolling Delta** - Visualize style changes across text
3. **Deep learning** - Neural authorship attribution (requires more data)

### Option D: Comparative Analysis
1. **Josephus corpus** - Single translator, multiple source authors (parallel to BoM claim)
2. **Late War** - Another pseudo-archaic text for comparison
3. **Other religious texts** - Quran, Doctrine & Covenants, etc.

### Option E: Write-Up & Publication
1. Consolidate findings into formal report
2. Document methodology for reproducibility
3. Acknowledge limitations explicitly

**Key constraint:** All methodological decisions have been validated by GPT-5.2 Pro. Do not deviate from the documented approach without re-consultation.

---

## Contact / Attribution

**Project:** Book of Mormon Textual Analysis
**Methodology developed via:** Claude + GPT-5.2 Pro collaborative dialogue
**Status:** Phase 1 COMPLETE (Protocol Development + Pilot Analysis)
