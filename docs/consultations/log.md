# GPT Consultation Log

This log tracks all consultations with GPT-5.2 Pro during the project.

---

## Summary Table

| Date | Type | Topic | Key Findings | Actions Taken |
|------|------|-------|--------------|---------------|
| 2026-02-01 | Methodology | Initial framework critique | 6 major gaps identified | Revised methodology to v2.0 |
| 2026-02-01 | Methodology | Concrete module template | 15-element template provided | Adopted in METHODOLOGY.md |
| 2026-02-01 | Methodology | Resource constraints | Phase 1/Phase 2 structure | Documented in LIMITATIONS.md |
| 2026-02-01 | Data Audit | Source text options | Phased approach recommended | Proceeded with Jenson 1830 |
| 2026-02-01 | Data Audit | Preprocessing review | 5 critical gaps, 3 medium | Addressing (in progress) |
| 2026-02-01 | Methodology | Segment annotation review | "Mormon" label conflates sources; boundary noise | Decisions pending user input |

---

## Consultation Details

### 2026-02-01: Initial Methodology Critique

**File:** `2026-02-01-methodology-initial.md` (see docs/methodology-dialogue.md for full transcript)

**Type:** Methodology Review

**Key Findings:**
1. Operational definitions missing
2. No comparison framework
3. Translation confounding undertheorized
4. Binary hypothesis too simple
5. LLM limitations need stronger guardrails
6. Falsifiability criteria too vague

**Actions:** Complete methodology rewrite to v2.0

---

### 2026-02-01: Preprocessing Data Audit

**File:** `2026-02-01-preprocessing-audit.md`

**Type:** Data Integrity Audit

**Key Findings:**
- Pipeline "directionally strong" but not "audit-grade"
- Critical gaps:
  1. Hyphenation/line-wrap artifacts not addressed
  2. Unicode normalization policy not specified
  3. Paratext inclusion policy not locked
  4. Output hashes missing
- Medium gaps:
  5. Header detection regex not documented
  6. Environment capture missing
  7. Removed blocks not preserved

**Actions:** Created 5 tasks to address gaps (in progress)

---

---

### 2026-02-01: Gap Resolution Verification

**File:** `2026-02-01-gap-verification.md`

**Type:** Data Integrity Audit (follow-up)

**Key Findings:**

**Now Sufficient:**
- Output hashing for all files
- Environment capture (basic level)
- Dehyphenation with auditable change log
- Paratext policy documentation
- Preservation of removed blocks
- Header detection documentation
- GPT governance protocol

**Still Needs Attention:**
1. Add input hashes, dependency lock, full artifact manifest
2. Fix 4 missed headers via overrides
3. Add Unicode normalization as parallel output
4. OCR validation sample (for quantitative claims)

**Verdict:** Can proceed with exploratory analysis while documenting limitations. Fine-grained quantitative claims require addressing remaining items.

---

---

### 2026-02-01: Voice Segmentation Module Review

**Type:** Methodology Review

**Key Findings:**

1. **Fixed JSD thresholds indefensible** - Must derive from controls empirically
2. **H4 vs H2 discrimination weak** - Method detects heterogeneity, not mechanism
3. **Critical flaw: Confounding** - Narrator correlates with book/genre/position
4. **Autocorrelation** - Contiguous blocks not independent; need blocked permutations

**Recommended Change:** Replace clustering-first with PERMANOVA + covariates

**Actions Taken:**
- Updated specification v0.1.0 → v0.2.0
- Added calibrated threshold approach
- Added PERMANOVA as primary test
- Added confound analysis section
- Added blocked permutation strategy

---

### 2026-02-01: Segment Annotation Methodology Review

**File:** `docs/decisions/segment-annotation-decisions.md`

**Type:** Methodology Review

**Key Findings:**

1. **"Mormon" label is not a single stylistic source** - In Mosiah–4 Nephi, "Mormon" includes redactor frame + embedded speeches/documents by other figures (Benjamin, Alma, etc.). This creates massive within-class heterogeneity.

2. **Book boundaries ≠ narrator boundaries** - Internal shifts (editorial asides, quoted records, letters, sermons, scripture quotations) are systematically mislabeled.

3. **Genre/quotation confounds** - Isaiah blocks, Jesus's discourse (3 Nephi), epistles cluster by register/source, not narrator.

4. **Fixed windows + midpoint assignment creates label noise** - Boundary segments are exactly where detection is most sensitive.

5. **Extreme sample imbalance** - Mormon (175) vs Enos (1) is statistically problematic.

**GPT Deliverables:**
- Complete annotation protocol with 16 embedded label types
- Stack-based embedding detector with regex OPEN/CLOSE patterns
- Verse-based hybrid segmentation recommendation
- Three minimum viable approaches (MVA-1, MVA-2, MVA-3)
- Priority/tie-breaking rules for label assignment

**GPT Recommendations:**
- Use verse-based segmentation with span annotation
- Add discourse-type annotation (frame vs embedded vs scripture)
- Split "Mormon" into ABRIDGER vs AUTHOR categories
- Require ≥10 segments per narrator for quantitative claims
- Use blocked cross-validation
- Implement MVA-3 (heuristic pre-annotation + human correction)

**Status:** Complete - pending user decisions on 6 methodology points

---

## Pending Consultations

- [ ] Validation study design review
- [x] Control corpora selection review (skipped - standard preprocessing)
