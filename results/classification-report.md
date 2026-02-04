# Stylometric Classification Results

**Generated:** 2026-02-02T05:11:48.289227+00:00
**Input:** data/text/processed/bom-stylometric-features.json

---

## Summary

- **Total blocks:** 244
- **Features:** 2753
- **Classes:** JACOB, MORMON, MORONI, NEPHI
- **Cross-validation:** 5-fold GroupKFold by run_id

---

## Classification Performance

### SVM (RBF)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 0.139 | [0.098, 0.184] |
| Macro F1 | 0.061 | [0.045, 0.078] |
| Weighted F1 | 0.171 | - |

**Confusion Matrix:**

```
Pred →    JACOB  MORMON  MORONI   NEPHI
True ↓
 JACOB        0      12       0       0
MORMON        0      34       0     137
MORONI        0      22       0       0
 NEPHI        0      39       0       0
```

### SVM (Linear)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 0.500 | [0.439, 0.566] |
| Macro F1 | 0.201 | [0.176, 0.229] |
| Weighted F1 | 0.507 | - |

**Confusion Matrix:**

```
Pred →    JACOB  MORMON  MORONI   NEPHI
True ↓
 JACOB        0       4       0       8
MORMON       11     117       4      39
MORONI        0      16       0       6
 NEPHI        7      26       1       5
```

### Random Forest

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 0.430 | [0.369, 0.492] |
| Macro F1 | 0.156 | [0.138, 0.175] |
| Weighted F1 | 0.425 | - |

**Confusion Matrix:**

```
Pred →    JACOB  MORMON  MORONI   NEPHI
True ↓
 JACOB        0      11       0       1
MORMON        1     104      15      51
MORONI        0      22       0       0
 NEPHI        0      38       0       1
```

---

## Top Discriminating Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | wherefore | 0.0326 |
| 2 | therefore | 0.0193 |
| 3 | 4g:lord | 0.0126 |
| 4 | 3g: to | 0.0096 |
| 5 | 3g:er  | 0.0091 |
| 6 | this | 0.0077 |
| 7 | upon | 0.0072 |
| 8 | my | 0.0072 |
| 9 | now | 0.0069 |
| 10 | after | 0.0068 |
| 11 | 4g:nd i | 0.0060 |
| 12 | 3g:of  | 0.0057 |
| 13 | 2g:on | 0.0057 |
| 14 | 2g:lo | 0.0052 |
| 15 | 4g:her  | 0.0051 |
| 16 | 2g:ri | 0.0050 |
| 17 | 3g:ord | 0.0050 |
| 18 | 3g: lo | 0.0049 |
| 19 | 2g:l  | 0.0049 |
| 20 | 3g: i  | 0.0048 |

---

## Class Distribution

| Voice | Blocks | % |
|-------|--------|---|
| MORMON | 171 | 70.1% |
| NEPHI | 39 | 16.0% |
| MORONI | 22 | 9.0% |
| JACOB | 12 | 4.9% |

---

## Interpretation

The classification results indicate whether the claimed narrators (Mormon, Nephi, Moroni, Jacob) 
exhibit statistically distinguishable stylistic signatures.

**Key observations:**

- Accuracy (50.0%) is moderately above chance (25.0%), suggesting some stylistic differentiation

**Limitations:**

- Sample sizes are imbalanced (Mormon >> others)
- Enos, Jarom, Omni excluded due to insufficient data
- Results may reflect topic/genre differences, not just author
- Translation layer effects cannot be fully separated

---

*See METHODOLOGY.md for full research framework and LIMITATIONS.md for scope boundaries.*