# Block Derivation Strategy (Option C)

**Date:** 2026-02-01
**Status:** Approved (GPT-5.2 Pro consensus)
**Supersedes:** Legacy 1000-word fixed blocks

---

## Decision

**Use verse-level annotation as the authoritative source, then derive fixed-length blocks programmatically when needed for ML models.**

This approach:
- Uses verses as ground truth (clean narrator boundaries)
- Concatenates contiguous verses with the same `voice` into blocks when ML models need fixed-length input
- Never crosses narrator/voice boundaries
- Applies minimum length thresholds (discard blocks < 500 words)

---

## Rationale (GPT-5.2 Pro Assessment)

### Why Option C is Methodologically Sound

1. **Boundary integrity is critical** - Stylometric features (function-word frequencies, character n-grams, POS n-grams) are highly sensitive to mixing authors/voices within a sample. Never crossing `voice` boundaries reduces label noise.

2. **Separates annotation from sampling** - Annotate at finest reliable granularity (verses), then generate model inputs systematically. Improves reproducibility and enables re-sampling for robustness checks.

3. **Enables group-aware evaluation** - With verse-level provenance, train/test splits can respect higher-level structure (speech runs, chapters, books), preventing leakage.

---

## Risks and Mitigations

### A. Non-independence / Pseudo-replication (Major)

**Risk:** Multiple blocks from contiguous verses are not independent samples. Can inflate accuracy if near-duplicate language appears in both train and test.

**Mitigation:** Use grouped cross-validation (`GroupKFold`) where the group is a contiguous "voice run" (or chapter/book).

### B. Topic/Genre Confounds (Major)

**Risk:** Differences attributed to "author style" may reflect sermon vs narrative vs genealogy, rather than authorship.

**Mitigations:**
- Use style-heavy/content-light features (most-frequent function words, character n-grams, POS n-grams)
- Include genre/topic controls
- Report results stratified by book/section/genre

### C. Editorial-Layer Contamination (Critical for BoM)

**Risk:** Stylometric signals may reflect the editorial hand (Mormon/Moroni) smoothing language, formulaic redaction phrases, or consistent translation register.

**Mitigations:**
- Run analyses within a single editorial frame (e.g., only within Mormon's compiled sections)
- Model hierarchically: voice differences conditional on frame narrator

### D. Quotation Material (High Impact)

**Risk:** Quoted KJV passages (Isaiah, Matthew) inject distinct register and vocabulary that can dominate attribution.

**Mitigation:** Exclude verses where `quote_source != null` for primary attribution experiments. Analyze quotations separately.

### E. Class Imbalance + Minimum-Length Filtering

**Risk:** Discarding blocks <500 words removes disproportionate text from short voices (Enos, Jarom, Omni).

**Mitigation:** Report per-voice retained token counts and number of blocks. Consider author inclusion thresholds.

---

## Block Derivation Algorithm

### Step 1: Create Voice Runs

Define a **voice run** as maximal contiguous verses with identical:
- `voice`
- `quote_source` status (quoted vs not)
- Optionally: `frame_narrator` (for editorial control)

Store a stable `run_id`. This is the **group** for CV splitting.

### Step 2: Target Block Sizes

Generate blocks at multiple sizes and report sensitivity:
- 500 words (minimum for character n-grams)
- 1000 words (standard for MFW/Delta methods)
- 2000 words (improved stability)

### Step 3: Window Strategy

**Primary results:** Non-overlapping blocks (reduces dependency, cleaner evaluation)

**If using overlap:** Must split train/test by `run_id` and disclose overlap percentage.

### Step 4: Handling Short Runs

Options (choose one and document):
1. Exclude short runs (simple, but biases against short voices)
2. Downshift threshold for minority voices
3. Use variable-length samples with normalized features
4. One-sample-per-run approach with length as covariate

**Recommendation:** Option 3 or 4 for representing all voices.

### Step 5: Block Construction Within a Run

Given a run with N words and target T:
1. Create k = floor(N/T) full blocks of ~T words (non-overlapping)
2. For remainder r:
   - If r >= min_len, keep as extra block
   - Else merge into previous block (if won't exceed max length)

### Step 6: Evaluation Design

- **Primary CV:** Group by `run_id` so blocks from same discourse never straddle folds
- **Harder test:** Hold out entire books (train on some, test on others)
- Report confidence intervals (bootstrap across groups/folds)

### Step 7: Record-Keeping

For every derived block, store:
- `block_id`
- `run_id`
- Start/end verse references
- Word count
- `voice`, `frame_narrator`
- Whether any quoted verses included
- Block index within run

---

## Legacy Approach Status

**Status:** Deprecated for primary modeling; retained for replication and sensitivity analyses.

The legacy 1000-word fixed blocks (`bom-1830-segments*.json`) are preserved for:
1. Reproducibility of any prior results
2. Robustness checks (conclusions should hold across segmentation schemes)
3. Diagnostic for boundary sensitivity

The 23 flagged boundary segments are no longer blocking issues under Option C.

---

## Implementation Checklist

- [ ] Create `scripts/derive_blocks.py` implementing the algorithm above
- [ ] Generate blocks at 500, 1000, 2000 word targets
- [ ] Store full provenance in output JSON
- [ ] Add `run_id` to verse-level data
- [ ] Document per-voice block counts and word totals
- [ ] Implement grouped CV evaluation

---

## References

- GPT-5.2 Pro methodology consultation (2026-02-01)
- Burrows, J. (2002). Delta: A measure of stylistic difference
- Eder, M. (2015). Does size matter? Authorship attribution, small samples, big problem
