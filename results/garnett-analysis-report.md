# Phase 2.D: Translation-Layer Calibration (Garnett Corpus)

## Research Question

Can function-word stylometry distinguish between Russian authors
(Dostoevsky, Tolstoy, Chekhov, Turgenev) when all works are translated
by a single translator (Constance Garnett)?

## Methodology

- **Block size**: 1000 words
- **Features**: 171 function words
- **Classifier**: Logistic Regression (balanced class weights)
- **CV**: Leave-one-work-out
- **Block weighting**: Each work contributes equally
- **Inference**: Work-level permutation test

## Novels Only (Primary)

### Corpus

- **Works**: 15
- **Words**: 2,762,711
- **Blocks**: 2759
- **Authors**: 3

| Author | Works | Words | Blocks |
|--------|-------|-------|--------|
| Dostoevsky | 6 | 1,154,027 | 1152 |
| Tolstoy | 4 | 1,003,218 | 1002 |
| Turgenev | 5 | 605,466 | 605 |

### Results

- **Work-weighted balanced accuracy**: 58.2%
- **Chance level**: 33%
- **Permutation p-value**: 0.1667
- **Bootstrap 95% CI**: [40.0%, 68.7%]

**Signal strength**: NONE

## Full Corpus (Secondary)

### Corpus

- **Works**: 19
- **Words**: 2,998,804
- **Blocks**: 2994
- **Authors**: 4

| Author | Works | Words | Blocks |
|--------|-------|-------|--------|
| Chekhov | 4 | 236,093 | 235 |
| Dostoevsky | 6 | 1,154,027 | 1152 |
| Tolstoy | 4 | 1,003,218 | 1002 |
| Turgenev | 5 | 605,466 | 605 |

### Results

- **Work-weighted balanced accuracy**: 54.4%
- **Chance level**: 25%
- **Permutation p-value**: 0.1667
- **Bootstrap 95% CI**: [39.9%, 56.9%]

**Signal strength**: NONE

## Interpretation for BoM Analysis

**Authorial signal does NOT reliably survive translation.**

The BoM null result remains **ambiguous** - translation homogenization
could explain the lack of narrator-specific patterns even if multiple
ancient authors existed.

*Generated: 2026-02-05T23:36:08.861162+00:00*