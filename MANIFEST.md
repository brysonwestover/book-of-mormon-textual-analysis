# File Manifest

**Version:** 1.5.1
**Date:** 2026-02-07
**Purpose:** Integrity verification for archival

---

## Core Analysis Files

### Scripts
| File | SHA-256 | Size |
|------|---------|------|
| `scripts/run_aggregated_analysis.py` | `07c6bb1b54d7859a8534e1bfb75a7343578f14d4f4c8e1420cb982e4d640dfbd` | ~95KB |

### Input Data
| File | SHA-256 | Size |
|------|---------|------|
| `data/text/processed/bom-voice-blocks.json` | `83d45c1a38f6d1a81cb7acefa305ec752db599e9d4fa0753a0e4c3f8368054a1` | ~6.5MB |

### Output Data
| File | SHA-256 | Size |
|------|---------|------|
| `results/run-aggregated-results.json` | `c46c098f1eb3f38b2bfbbd9f23849a6ee8aaaa95ff45df92abac8b51cbd07918` | ~50KB |

### Documentation
| File | SHA-256 | Size |
|------|---------|------|
| `docs/METHODOLOGY-v1.5.1.md` | `5adfd745cd9aa32086ba1b547d40265ce6fbcc45ae69996ebaa013802eaf277a` | ~8KB |

---

## Verification Commands

```bash
# Verify all checksums
sha256sum -c <<EOF
07c6bb1b54d7859a8534e1bfb75a7343578f14d4f4c8e1420cb982e4d640dfbd  scripts/run_aggregated_analysis.py
83d45c1a38f6d1a81cb7acefa305ec752db599e9d4fa0753a0e4c3f8368054a1  data/text/processed/bom-voice-blocks.json
c46c098f1eb3f38b2bfbbd9f23849a6ee8aaaa95ff45df92abac8b51cbd07918  results/run-aggregated-results.json
5adfd745cd9aa32086ba1b547d40265ce6fbcc45ae69996ebaa013802eaf277a  docs/METHODOLOGY-v1.5.1.md
EOF
```

---

## Figure Files

| File | Description |
|------|-------------|
| `results/figures/primary-permutation-null.png` | Blocked permutation null distribution |
| `results/figures/primary-confusion-matrix.png` | 3-class confusion matrix |
| `results/figures/exploratory-4class-cm.png` | 4-class confusion matrix |
| `results/figures/exploratory-delta-cm.png` | Burrows' Delta confusion matrix |

---

## Audit Trail

| File | Description |
|------|-------------|
| `docs/private/AUDIT-FINAL-v1.5.1-GPT.md` | Final GPT audit (PASS) |
| `docs/private/AUDIT-v1.5.1-GPT-REVIEW.md` | GPT review of v1.5.1 |
| `docs/private/AUDIT-v1.5.0-GPT-REVIEW.md` | GPT review of v1.5.0 |
| `docs/private/AUDIT-RESPONSE-PLAN-v1.5.1.md` | Audit response documentation |
| `docs/private/OUTSTANDING-ITEMS-v1.5.1.md` | Tracked items and resolutions |

---

## Archive Completeness Checklist

- [x] Analysis script (v1.5.1)
- [x] Input data (bom-voice-blocks.json)
- [x] Output results (run-aggregated-results.json)
- [x] Figures (4 PNG files)
- [x] Methodology documentation
- [x] Audit trail
- [x] Checksums (this file)
- [x] CITATION.cff
- [x] DATA_PROVENANCE.md
- [x] DATA_DICTIONARY.md
- [x] CHANGELOG.md
- [x] Requirements (pinned versions)

---

## Notes

1. **IMPORTANT:** The results JSON file (`results/run-aggregated-results.json`) was
   truncated during a previous interrupted run. Before production use, regenerate:
   ```bash
   python scripts/run_aggregated_analysis.py --quick  # Development (~2 min)
   python scripts/run_aggregated_analysis.py          # Production (~4-6 hours)
   ```

2. After regeneration, update checksums in this file.

3. Checksums generated with `sha256sum` on Linux.

4. Key expected results after regeneration:
   - Primary BA: 45.0%
   - Blocked p-value: ~0.51
   - Unrestricted p-value: ~0.08-0.12
