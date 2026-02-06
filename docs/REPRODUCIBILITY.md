# Reproducibility Documentation

This document provides everything needed to reproduce the analyses in this project.

**Last Updated:** 2026-02-06
**Pre-Registration:** [OSF DOI 10.17605/OSF.IO/4W3KH](https://osf.io/4W3KH)

---

## 1. Environment Setup

### System Requirements

| Component | Version Used | Notes |
|-----------|--------------|-------|
| **Python** | 3.12.3 | Required |
| **OS** | Ubuntu 24.04 (Linux 6.14.0) | Tested on |
| **Architecture** | x86_64 | |
| **RAM** | 8+ GB recommended | For full analysis |

### Installation

```bash
# Clone repository
git clone https://github.com/brysonwestover/book-of-mormon-textual-analysis.git
cd book-of-mormon-textual-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install exact dependencies
pip install -r requirements-lock.txt
```

### Dependencies (Locked Versions)

Key packages (see `requirements-lock.txt` for complete list):

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.4.2 | Numerical operations |
| scipy | 1.17.0 | Statistical tests |
| scikit-learn | 1.8.0 | Classification, cross-validation |
| joblib | 1.5.3 | Parallel processing |

---

## 2. Random Seeds and Determinism

### Global Random Seed

All scripts use `RANDOM_SEED = 42` for reproducibility.

| Script | Seed Usage |
|--------|------------|
| `run_classification_v3.py` | Model fitting, permutation sampling |
| `run_robustness_optimized.py` | All variants, permutation sampling |
| `run_garnett_analysis_optimized.py` | LOWO CV, permutation test, bootstrap |
| `run_tost_equivalence.py` | Bootstrap sampling |

### Determinism Notes

1. **NumPy seeding:** `np.random.seed(RANDOM_SEED)` called at script start
2. **Permutation generation:** Each permutation uses `RANDOM_SEED + i` for deterministic sequence
3. **Block sampling:** Uses deterministic hash-based seeds for reproducibility across runs
4. **Parallel execution:** Results are order-independent (aggregated by permutation ID)

### Known Non-Determinism Sources

- **scikit-learn parallelism:** With `n_jobs > 1`, minor floating-point differences possible
- **AWS vs local:** CPU architecture differences may cause ~1e-15 floating-point variance
- **Mitigation:** All results rounded to reported precision; statistical conclusions unaffected

---

## 3. Data Provenance

### Source Text

| File | Source | SHA-256 |
|------|--------|---------|
| `book-of-mormon-1830-replica.txt` | Internet Archive (Thomas A. Jenson) | `da8c01d2...48902` |

See `data/text/README.md` for complete provenance chain and preprocessing details.

### Derived Datasets

| File | Description | SHA-256 |
|------|-------------|---------|
| `bom-verses-annotated-v3.json` | 6,604 verses with voice annotation | `e6d38a17930ecdbb1fb56fe8b50e942c77fe024d28ac16506edc687356973949` |
| `bom-voice-blocks.json` | 244 blocks in 14 runs (1000-word) | `83d45c1a38f6d1a81cb7acefa305ec752db599e9d4fa0753a0e4c3f8368054a1` |
| `bom-stylometric-features.json` | Feature vectors (2,753 features/block) | `627d86e58a24562cbab764ebd0cadcec68ff00c55f08c0216714d99f26622424` |

### Verification

```bash
# Verify derived dataset integrity
sha256sum data/text/processed/bom-verses-annotated-v3.json
# Expected: e6d38a17930ecdbb1fb56fe8b50e942c77fe024d28ac16506edc687356973949

sha256sum data/text/processed/bom-voice-blocks.json
# Expected: 83d45c1a38f6d1a81cb7acefa305ec752db599e9d4fa0753a0e4c3f8368054a1
```

---

## 4. Analysis Pipeline

### Rebuild from Scratch

```bash
# Activate environment
source venv/bin/activate

# Step 1: Preprocess source text (already done, creates bom-1830-clean.txt)
python scripts/preprocess_1830.py

# Step 2: Derive voice blocks from annotated verses
python scripts/derive_blocks.py

# Step 3: Extract stylometric features
python scripts/extract_features.py

# Step 4: Run primary classification (Phase 2.0)
python scripts/run_classification_v3.py

# Step 5: Run TOST equivalence test
python scripts/run_tost_equivalence.py

# Step 6: Run Garnett calibration (Phase 2.D)
python scripts/run_garnett_analysis_optimized.py

# Step 7: Run robustness analysis (Phase 2.A) - computationally intensive
python scripts/run_robustness_optimized.py --permutations 10000
```

### Expected Outputs

| Script | Output File | Key Result |
|--------|-------------|------------|
| `run_classification_v3.py` | `results/classification-results-v3.json` | BA=24.2%, p=0.177 |
| `run_tost_equivalence.py` | `results/tost-equivalence-results.json` | p=0.06, BF=2.85 |
| `run_garnett_analysis_optimized.py` | `results/garnett-checkpoint.json` | BA=58.2%, p=0.0016 |
| `run_robustness_optimized.py` | `results/robustness-results.json` | All variants ~25% |

---

## 5. Permutation Test Validity

### What Is Permuted

**Labels are permuted at the RUN level, not the block level.**

- Unit of permutation: Voice run (e.g., `run_0001`, `run_0002`, ...)
- Total runs: 14 (across 4 narrators)
- Blocks within runs: Keep their original run assignment

### Constraints Preserved

The restricted permutation scheme preserves:

1. **Class run counts:** Each permutation maintains the same number of runs per narrator class
   - MORMON: 4 runs
   - NEPHI: 5 runs
   - MORONI: 2 runs
   - JACOB: 3 runs

2. **Block-run structure:** Blocks within a run stay together (not shuffled individually)

3. **Total sample size:** Same 244 blocks in each permutation

### Why Run-Level Permutation?

Blocks within the same run are **pseudoreplicated**:
- Adjacent text from same narrator
- Correlated stylistic features
- Not exchangeable at block level

Permuting at block level would:
- Inflate significance (treat N=244 instead of N=14)
- Violate exchangeability assumptions
- Produce anti-conservative p-values

### Implementation

From `run_classification_v3.py`:

```python
def sample_restricted_permutations(voice_runs, n_samples, seed):
    """
    Sample permutations that preserve class run-counts.

    Each permutation reassigns run labels while keeping:
    - Same number of runs per class
    - Blocks within runs grouped together
    """
    rng = np.random.RandomState(seed)
    # ... generates valid permutations only
```

### Leakage Prevention

**Leave-One-Work-Out (LOWO) Cross-Validation:**

- "Work" = voice run (contiguous same-narrator passage)
- Each fold holds out one run entirely
- No blocks from held-out run appear in training
- Prevents train/test contamination from adjacent text

---

## 6. Sensitivity and Minimum Detectable Effect

### Study Design Parameters

| Parameter | Value |
|-----------|-------|
| Effective sample size | N = 14 runs |
| Classes | 4 (MORMON, NEPHI, MORONI, JACOB) |
| Chance baseline | 25% (balanced) |
| Permutations | 100,000 (Phase 2.0) / 10,000 (Phase 2.A) |
| Alpha | 0.05 (one-sided) |

### What Effect Sizes Could We Detect?

Given N=14 runs with severe class imbalance:

| Effect Size | Approximate Accuracy | Detectability |
|-------------|---------------------|---------------|
| Large (d > 0.8) | > 50% | Likely detectable |
| Medium (d ~ 0.5) | 35-50% | Possibly detectable |
| Small (d < 0.3) | 25-35% | Unlikely to detect |

### Calibration Evidence

The Garnett study provides empirical sensitivity evidence:

| Corpus | Accuracy | vs Chance | Detected? |
|--------|----------|-----------|-----------|
| Garnett (4 authors, single translator) | 58.2% | +25 pts | **Yes** (p=0.0016) |
| BoM (4 narrators) | 24.2% | -0.8 pts | **No** (p=0.177) |

**Interpretation:** The method can detect authorial differences of ~25 percentage points above chance (as demonstrated in Garnett). The BoM result is consistent with effects smaller than this threshold.

### Equivalence Bounds

TOST equivalence testing used bounds of ±10 percentage points from chance:
- Lower bound: 15% accuracy
- Upper bound: 35% accuracy
- Result: p = 0.06 (near-equivalence, not formally established)

### Sensitivity Statement

> "This study was designed to detect narrator-level stylometric differences that would produce classification accuracy substantially above chance (25%). The Garnett calibration demonstrates the pipeline can detect effects of ~25 percentage points above chance through a single translator's voice. The observed BoM result (24.2%) is consistent with effects smaller than this detectable threshold, but we cannot rule out small effects (< 10 percentage points) that may exist below our sensitivity floor."

---

## 7. Analysis-Specific Documentation

### Phase 2.0: Primary Confirmatory

| Aspect | Specification |
|--------|---------------|
| Features | 169 function words (content-suppressed) |
| Classifier | Logistic Regression (balanced class weights) |
| CV scheme | Leave-One-Work-Out (14 folds) |
| Metric | Run-weighted balanced accuracy |
| Permutations | 100,000 (restricted, group-level) |
| Block cap | 20 blocks max per run |

### Phase 2.A: Robustness

| Variant | Change from Primary | Rationale |
|---------|---------------------|-----------|
| A1 | 500-word blocks | Sensitivity to block size |
| A2 | 2000-word blocks | Sensitivity to block size |
| A3 | Include quotations | Test quote exclusion decision |
| A4 | Character 3-grams | Alternative feature space |
| A5 | FW + char 3-grams | Combined features |
| A6 | SVM classifier | Alternative algorithm |

Multiple testing correction: MaxT (family-wise error rate control)

### Phase 2.D: Garnett Calibration

| Aspect | Specification |
|--------|---------------|
| Corpus | 19 works by 4 Russian authors (Garnett translations) |
| Purpose | Validate method detects authorship through single translator |
| CV scheme | Leave-One-Work-Out (19 folds) |
| Permutations | 10,000 |
| Confound controls | Character name masking, period stratification |

---

## 8. File Manifest

### Scripts → Outputs

| Script | Primary Output | Secondary Outputs |
|--------|----------------|-------------------|
| `preprocess_1830.py` | `bom-1830-clean.txt` | preprocessing_log.json |
| `derive_blocks.py` | `bom-voice-blocks.json` | - |
| `extract_features.py` | `bom-stylometric-features.json` | - |
| `run_classification_v3.py` | `classification-results-v3.json` | classification-report-v3.md |
| `run_tost_equivalence.py` | `tost-equivalence-results.json` | - |
| `run_garnett_analysis_optimized.py` | `garnett-checkpoint.json` | garnett-analysis.log |
| `run_robustness_optimized.py` | `robustness-results.json` | robustness-checkpoint-v2.json |

---

## 9. Troubleshooting

### Common Issues

**Memory errors during robustness analysis:**
- Reduce `--jobs` parameter (default uses all CPUs)
- Run on machine with more RAM
- Use AWS c7i.8xlarge or similar (32 vCPU, 64 GB RAM)

**Different results than reported:**
- Verify Python version (3.12.x required)
- Verify exact package versions via `pip freeze`
- Check derived dataset checksums match

**Permutation test takes too long:**
- Reduce `--permutations` parameter for testing
- Full 100,000 permutations require ~2-4 hours on modern hardware

---

## 10. Contact

For reproducibility issues, open a GitHub issue at:
https://github.com/brysonwestover/book-of-mormon-textual-analysis/issues
