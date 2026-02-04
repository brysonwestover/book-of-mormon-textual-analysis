# Additional Stylometric Analyses

**Generated:** 2026-02-02T05:53:08.109558+00:00

These analyses investigate the null result from the primary classification.

---

## 1. Unmasking Analysis

**Question:** Do differences collapse quickly (shallow/topic) or gradually (deep/authorial)?

**Interpretation:** No initial separation to unmask (started at/below chance)

| Iteration | Features | Balanced Acc |
|-----------|----------|--------------|
| 0 | 171 | 0.201 |
| 1 | 166 | 0.165 |
| 2 | 161 | 0.121 |
| 3 | 156 | 0.119 |
| 4 | 151 | 0.112 |
| 5 | 146 | 0.109 |
| 6 | 141 | 0.091 |
| 7 | 136 | 0.086 |
| 8 | 131 | 0.079 |
| 9 | 126 | 0.062 |

---

## 2. Pairwise Comparisons

**Question:** Can we distinguish pairs of narrators (simpler than 4-way)?

| Comparison | N | Balanced Acc | vs 50% Baseline |
|------------|---|--------------|-----------------|
| MORMON vs NEPHI | 210 | 65.9% | +15.9% |
| MORMON vs MORONI | 193 | 30.9% | -19.1% |
| MORMON vs JACOB | 183 | 52.1% | +2.1% |
| NEPHI vs MORONI | 61 | 28.2% | -21.8% |
| NEPHI vs JACOB | 51 | 39.7% | -10.3% |
| MORONI vs JACOB | 34 | 10.6% | -39.4% |

---

## 3. Same-Author Verification

**Question:** Are cross-narrator pairs more distant than same-narrator pairs?

- Same-narrator distance: 0.9400 ± 0.1338
- Different-narrator distance: 0.9965 ± 0.1231
- **Cohen's d:** 0.440
- **Interpretation:** SMALL effect - weak narrator differentiation

---

## 4. Within-Class Consistency

**Question:** How internally consistent is each narrator?

| Narrator | N | Mean Distance | Std |
|----------|---|---------------|-----|
| JACOB | 12 | 0.9100 | 0.0935 |
| MORMON | 171 | 0.9920 | 0.1310 |
| MORONI | 22 | 0.9534 | 0.1457 |
| NEPHI | 39 | 0.9140 | 0.1345 |

- **Most consistent:** JACOB
- **Least consistent:** MORMON

---

## Summary

**Conclusion:** Results are mixed. Further investigation with control corpora needed.