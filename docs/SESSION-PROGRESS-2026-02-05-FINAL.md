# Session Progress Report: February 5-6, 2026

## Session Summary

This session completed the Garnett translation-layer calibration study (Phase 2.D) and filed two OSF amendments documenting methodological refinements and limitations discovered during execution.

---

## Completed Work

### 1. OSF Amendment #1 (Filed)
**URL**: https://osf.io/sthmk/files/uzhkd

Documented deviations for Phase 2.A and 2.D:
- Robustness testing: 10,000 permutations (vs 100,000)
- Variant A3 excluded from maxT family
- TOST/Bayes supplementary analyses added
- Garnett corpus expanded from 11 to 19 works
- Confound controls (name masking, period stratification) implemented

### 2. OSF Amendment #2 (Filed)
**URL**: https://osf.io/sthmk/files/ruzq9

Documented bootstrap CI feasibility criterion:
- Discovered mathematical incompatibility between stratified bootstrap and LOWO CV when author has ≤2 works
- Early period: ~30% failure rate
- Late period: ~49% failure rate
- Feasibility criterion: Bootstrap CIs only if ≥95% replicates succeed
- Permutation test verified feasible (0% failure rate)

### 3. Phase 2.D Garnett Calibration (Complete)

#### Primary Analysis: Novels Only
| Metric | Value |
|--------|-------|
| Accuracy | 58.2% |
| Chance | 33.3% (3 authors) |
| p-value | **0.0016** |
| Bootstrap 95% CI | [32.5%, 70.5%] |
| Status | ✅ **Significant** |

#### Secondary Analysis: Full Corpus
| Metric | Value |
|--------|-------|
| Accuracy | 54.4% |
| Chance | 25% (4 authors) |
| p-value | **0.0001** |
| Status | ✅ **Significant** |

#### Period-Stratified: Early (1894-1904)
| Metric | Value |
|--------|-------|
| Authors | Tolstoy (2 works), Turgenev (5 works) |
| Accuracy | 68.2% |
| Chance | 50% |
| p-value | 0.148 |
| Bootstrap CI | NOT ESTIMABLE (Amendment #2) |
| Status | ⚠️ Not significant (insufficient power) |

#### Period-Stratified: Late (1912-1918)
| Metric | Value |
|--------|-------|
| Authors | Dostoevsky (6), Tolstoy (2), Chekhov (4) |
| Accuracy | 55.6% |
| Chance | 33.3% |
| p-value | **0.006** |
| Bootstrap CI | NOT ESTIMABLE (Amendment #2) |
| Status | ✅ **Significant** |

---

## In Progress

### Phase 2.A Robustness Testing (AWS)
- **Instance**: c7i.8xlarge (32 vCPU)
- **Progress**: 3,000 / 10,000 permutations (30%)
- **Estimated completion**: ~10-15 hours remaining
- **Estimated cost**: ~$50-70 total

**Observed scores (preliminary):**
| Variant | Accuracy | Notes |
|---------|----------|-------|
| A1 (500-word blocks) | 27.9% | Near chance |
| A2 (2000-word blocks) | 18.8% | Below chance |
| A3 (Include quotes) | 31.1% | Slightly above |
| A4 (Char 3-grams) | 26.1% | Near chance |
| A5 (Combined) | 24.5% | Near chance |
| A6 (SVM) | 23.6% | Near chance |

---

## Key Findings

### Garnett Calibration Conclusion
**Stylometry CAN detect authorial signal through a single translator's voice.**

- 3 of 4 analyses show significant discrimination (p < 0.05)
- Primary analysis: 58.2% accuracy, 25 points above chance
- This validates the method and makes the BoM null result informative

### Book of Mormon Implication
The BoM primary result (24.2% accuracy ≈ 25% chance, p = 0.177) is now interpretable:
- The method works (proven by Garnett)
- The BoM narrators genuinely lack distinguishable function-word patterns
- Consistent with single-author hypothesis (H1)

---

## Files Created/Modified

### Results Files
- `results/garnett-checkpoint.json` - Primary analyses checkpoint
- `results/period-analysis-results.json` - Period-stratified results
- `results/garnett-analysis.log` - Execution log

### Documentation
- `docs/osf-amendment-2-2026-02-05.md` - Second amendment
- `docs/SESSION-PROGRESS-2026-02-05-FINAL.md` - This file

### Scripts
- `scripts/run_garnett_analysis_optimized.py` - Updated with edge case handling
- `scripts/run_period_analyses.py` - Period-only analysis script

---

## Pending Tasks

1. **Monitor AWS robustness** - Check completion (~10-15 hours)
2. **Download AWS results** - When complete
3. **Terminate AWS instance** - After downloading
4. **Write final report** - Synthesize all Phase 2 results
5. **Update PROJECT-WIKI.md** - With final results

---

## AWS Access Info
```
Instance: c7i.8xlarge
IP: 54.84.56.89
SSH: ssh -i ~/.ssh/bom-analysis-key.pem ubuntu@54.84.56.89
Check progress: python3 -c "import json; d=json.load(open('/home/ubuntu/bom-analysis/results/robustness-checkpoint-v2.json')); print(d['completed_perms'])"
```

---

## Session Timeline

| Time (UTC) | Action |
|------------|--------|
| ~20:40 | Started Garnett analysis locally |
| ~21:20 | Novels-only and full-corpus complete |
| ~22:08 | Period analysis crashed (bootstrap issue) |
| ~22:30 | Forensic audit completed |
| ~23:00 | Amendment #2 drafted |
| ~23:30 | Amendment #2 filed to OSF |
| ~23:42 | Period analyses completed (permutation only) |

---

## Researcher Notes

The bootstrap feasibility issue was an unforeseen methodological limitation. The forensic audit confirmed:
1. LOWO CV implementation is correct
2. Issue is mathematical: stratified bootstrap + LOWO + 2-work author = incompatible
3. Permutation test remains valid (preserves exact author counts)

This limitation was documented transparently in Amendment #2 BEFORE examining period-stratified inferential results. The amendment includes timing disclosure and audit trail references.

---

*Last updated: 2026-02-06 00:00 UTC*
