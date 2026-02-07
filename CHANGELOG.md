# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.5.1] - 2026-02-07

### Added
- **BLOCKED permutation as PRIMARY inference** - Addresses exchangeability violation from narrator-book collinearity
- Bootstrap 95% CI for balanced accuracy (labeled "descriptive only")
- Per-class recall with Wilson binomial CIs
- RNG seeds documented explicitly in metadata
- Collinearity metric defined in docstring
- Strata definition documented
- `docs/METHODOLOGY-v1.5.1.md` - Full methodology documentation
- `docs/private/AUDIT-FINAL-v1.5.1-GPT.md` - GPT audit (PASS verdict)
- Archive documentation (CITATION.cff, DATA_PROVENANCE.md, DATA_DICTIONARY.md)

### Changed
- Unrestricted permutation demoted to reference only (not valid for inference)
- Output now reports blocked-null mean (~42%) as baseline instead of theoretical 33.3%
- Bootstrap CI labeled "(descriptive only)" to prevent misinterpretation

### Removed
- Unsupported power claim ("~25+ pts above chance")

### Fixed
- JSON serialization of numpy bool types

---

## [1.5.0] - 2026-02-06

### Added
- Pre-specified PRIMARY analysis structure (3-class LR as confirmatory)
- FDR correction (Benjamini-Hochberg) for exploratory analyses
- Blocked permutation test (within book-strata) as sensitivity analysis
- Narrator-book contingency table in output
- Methodology card with all hyperparameters
- Feature ranking inside CV folds (leak-proof sensitivity analysis)

### Changed
- Renamed "confound_probe" to "narrator_vs_book_comparison" with explicit limitations
- Restructured output: primary_analysis vs exploratory_analyses vs sensitivity

---

## [1.4.0] - 2026-02-05

### Added
- Burrows' Delta baseline (canonical stylometry method)
- Per-class metrics (precision, recall, F1 for each class)
- Permutation-based uncertainty quantification
- Visualization: permutation null distribution plot
- Visualization: confusion matrix heatmaps (LR and Delta)

---

## [1.3.0] - 2026-02-05

### Added
- Zero-word guard in aggregation (prevents NaN/inf from empty runs)
- Strong caveats to bootstrap_ci about duplicate-leakage bias

### Changed
- Renamed 'restricted_permutation_test' to 'permutation_test'
- Increased max_iter to 2000 for better convergence in high-dim setting

### Fixed
- Wilson interval clarified as valid for RAW accuracy only

---

## [1.2.0] - 2026-02-04

### Added
- Count-based aggregation: sum raw FW counts across blocks, then convert to frequencies
- +1 correction to p-value (Phipson & Smyth 2010)
- C sensitivity analysis (tests C ∈ {0.01, 0.1, 1.0, 10.0, 100.0})
- Wilson confidence interval for raw accuracy
- Jackknife influence analysis

---

## [1.0.0] - 2026-02-04

### Added
- Initial run-aggregated analysis script
- Run-level aggregation addressing pseudoreplication
- Leave-one-out cross-validation
- Basic permutation test
- 4-class classification (all narrators)

---

## Prior Versions

See git history for changes to the broader project infrastructure:
- Text preprocessing pipeline
- Narrator segmentation
- Control corpora integration
- OSF pre-registration documentation
