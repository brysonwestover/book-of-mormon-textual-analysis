# Implementation Plan: Publication-Quality Run-Aggregated Analysis

**Created:** 2026-02-06
**Target:** DSH/LLC Journal
**Status:** PLANNING

---

## Scope

Expand `run_aggregated_analysis.py` from a supplementary script to a publication-quality analysis that would satisfy DSH/LLC reviewers.

---

## Implementation Order

### Phase 1: Core Enhancements (Tier 1 - Essential)

| # | Item | Priority | Complexity | Dependencies |
|---|------|----------|------------|--------------|
| 1.1 | Permutation-based CI for BA | HIGH | Medium | None |
| 1.2 | Enhanced per-class metrics | HIGH | Low | None |
| 1.3 | Permutation null distribution plot | HIGH | Low | None |
| 1.4 | Burrows' Delta baseline | HIGH | Medium | None |
| 1.5 | MDE/sensitivity analysis | HIGH | High | 1.1 |

### Phase 2: Robustness (Tier 2 - Strong)

| # | Item | Priority | Complexity | Dependencies |
|---|------|----------|------------|--------------|
| 2.1 | 3-class analysis (exclude MORONI) | MEDIUM | Low | Phase 1 |
| 2.2 | Feature set sensitivity (50,100,150,169) | MEDIUM | Medium | Phase 1 |
| 2.3 | Segment-length sensitivity | MEDIUM | High | Phase 1 |
| 2.4 | Confound probe (predict BOOK) | MEDIUM | Medium | Phase 1 |

---

## Detailed Implementation Notes

### 1.1 Permutation-Based CI for Balanced Accuracy

**Approach:** Bootstrap over runs with full pipeline recomputation

```python
def bootstrap_ci_runs(X, y, n_bootstrap=10000, seed=42):
    """
    Bootstrap CI at run level.
    Resample runs (not blocks), recompute full pipeline each time.
    """
    rng = np.random.RandomState(seed)
    n_runs = len(y)
    ba_boot = []

    for b in range(n_bootstrap):
        # Resample run indices with replacement
        idx = rng.choice(n_runs, size=n_runs, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]

        # Full pipeline: LOO + BA
        try:
            y_pred = leave_one_out_cv(X_boot, y_boot)
            ba = balanced_accuracy_score(y_boot, y_pred)
            ba_boot.append(ba)
        except:
            continue  # Skip degenerate samples

    # Percentile CI (or BCa for small samples)
    ci_lower = np.percentile(ba_boot, 2.5)
    ci_upper = np.percentile(ba_boot, 97.5)

    return ci_lower, ci_upper, ba_boot
```

**Key decision:** Use percentile CI (simpler) or BCa (better for small N, more complex)?
- **Recommendation:** Implement BCa for publication quality

---

### 1.2 Enhanced Per-Class Metrics

**Add to results:**
- Confusion matrix (already present, enhance display)
- Per-class recall, precision, F1
- Macro-F1 and weighted-F1
- Bootstrap CI for each per-class metric

```python
from sklearn.metrics import classification_report, precision_recall_fscore_support

def compute_per_class_metrics(y_true, y_pred, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {
        'per_class': {
            c: {'precision': p, 'recall': r, 'f1': f, 'support': s}
            for c, p, r, f, s in zip(classes, precision, recall, f1, support)
        },
        'macro_f1': np.mean(f1),
        'weighted_f1': np.average(f1, weights=support)
    }
```

---

### 1.3 Permutation Null Distribution Plot

**Implementation:**
```python
import matplotlib.pyplot as plt

def plot_permutation_null(null_scores, observed, p_value, output_path):
    """
    Plot histogram of null distribution with observed value marked.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram
    ax.hist(null_scores, bins=50, density=True, alpha=0.7,
            color='steelblue', edgecolor='white')

    # Observed value
    ax.axvline(observed, color='red', linewidth=2,
               label=f'Observed BA = {observed:.1%}')

    # Shade rejection region
    extreme = null_scores[null_scores >= observed]
    if len(extreme) > 0:
        ax.axvspan(observed, max(null_scores), alpha=0.3, color='red',
                   label=f'p = {p_value:.4f}')

    # Annotations
    ax.axvline(np.mean(null_scores), color='gray', linestyle='--',
               label=f'Null mean = {np.mean(null_scores):.1%}')

    ax.set_xlabel('Balanced Accuracy')
    ax.set_ylabel('Density')
    ax.set_title('Permutation Null Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

---

### 1.4 Burrows' Delta Baseline

**Implementation:** Nearest-centroid classifier with z-scored features

```python
def burrows_delta_loo(X, y, classes):
    """
    Leave-one-out cross-validation using Burrows' Delta.
    """
    n = len(y)
    y_pred = np.empty(n, dtype=y.dtype)

    for i in range(n):
        # Train/test split
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]

        # Z-score using training data only
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0, ddof=1)
        sd[sd == 0] = 1  # Avoid division by zero

        Z_train = (X_train - mu) / sd
        Z_test = (X_test - mu) / sd

        # Compute class centroids
        centroids = {}
        for c in classes:
            mask = (y_train == c)
            if mask.sum() > 0:
                centroids[c] = Z_train[mask].mean(axis=0)

        # Delta distance: mean absolute difference
        deltas = {}
        for c, centroid in centroids.items():
            deltas[c] = np.mean(np.abs(Z_test - centroid))

        # Predict: minimum Delta
        y_pred[i] = min(deltas, key=deltas.get)

    return y_pred
```

---

### 1.5 MDE/Sensitivity Analysis

**Approach:** Simulation-based power analysis

```python
def mde_analysis(X, y, deltas=None, n_simulations=200, n_permutations=1000):
    """
    Estimate minimum detectable effect.

    Inject signal of varying strength, measure detection rate.
    """
    if deltas is None:
        deltas = np.linspace(0, 1.5, 16)

    # Get first principal component as signal direction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(X)
    v = pca.components_[0]

    classes = np.unique(y)
    class_codes = {c: i for i, c in enumerate(classes)}

    results = []
    for delta in deltas:
        detections = 0
        ba_values = []

        for sim in range(n_simulations):
            # Inject signal: shift each class centroid
            X_sim = X.copy()
            for c in classes:
                mask = (y == c)
                shift = delta * v * (class_codes[c] - len(classes)/2)
                X_sim[mask] += shift

            # Run pipeline
            y_pred = leave_one_out_cv(X_sim, y)
            ba = balanced_accuracy_score(y, y_pred)
            ba_values.append(ba)

            # Quick permutation test (fewer permutations for speed)
            null_scores = []
            rng = np.random.RandomState(sim)
            for _ in range(n_permutations):
                y_perm = y[rng.permutation(len(y))]
                y_pred_perm = leave_one_out_cv(X_sim, y_perm)
                null_scores.append(balanced_accuracy_score(y_perm, y_pred_perm))

            p_value = (np.sum(null_scores >= ba) + 1) / (n_permutations + 1)
            if p_value < 0.05:
                detections += 1

        power = detections / n_simulations
        results.append({
            'delta': delta,
            'power': power,
            'mean_ba': np.mean(ba_values),
            'std_ba': np.std(ba_values)
        })

    # Find MDE (80% power)
    mde = None
    for r in results:
        if r['power'] >= 0.80:
            mde = r['delta']
            break

    return results, mde
```

**Note:** This is computationally expensive. May need to reduce simulations for practical runtime.

---

### 2.1 3-Class Analysis (Exclude MORONI)

```python
def run_analysis(X, y, run_ids, run_info, exclude_classes=None):
    """
    Run full analysis, optionally excluding certain classes.
    """
    if exclude_classes:
        mask = ~np.isin(y, exclude_classes)
        X = X[mask]
        y = y[mask]
        run_ids = [rid for i, rid in enumerate(run_ids) if mask[i]]
        run_info = {k: v for k, v in run_info.items() if k in run_ids}

    # ... rest of analysis
```

Run twice: once with all 4 classes, once excluding MORONI.

---

### 2.2 Feature Set Sensitivity

```python
def feature_sensitivity_analysis(X, y, feature_names, k_values=None):
    """
    Test sensitivity to number of features.

    Rank features by corpus frequency, test with top-k.
    """
    if k_values is None:
        k_values = [50, 100, 150, len(feature_names)]

    # Rank features by mean frequency (across all runs)
    feature_ranks = np.argsort(-X.mean(axis=0))  # Descending

    results = {}
    for k in k_values:
        top_k = feature_ranks[:k]
        X_k = X[:, top_k]

        y_pred = leave_one_out_cv(X_k, y)
        ba = balanced_accuracy_score(y, y_pred)

        # Permutation test
        perm_results = permutation_test(X_k, y, ba, n_permutations=10000)

        results[k] = {
            'balanced_accuracy': ba,
            'p_value': perm_results['p_value'],
            'features_used': [feature_names[i] for i in top_k]
        }

    return results
```

---

### 2.3 Segment-Length Sensitivity

**Note:** This requires re-aggregating from raw blocks with different sizes.
Need to check if we have access to raw text or only pre-aggregated blocks.

---

### 2.4 Confound Probe (Predict BOOK)

```python
def confound_probe_book(X, y_narrator, y_book):
    """
    Test if BOOK is more predictable than NARRATOR.

    If so, suggests topical/structural confound.
    """
    # Narrator prediction
    y_pred_narrator = leave_one_out_cv(X, y_narrator)
    ba_narrator = balanced_accuracy_score(y_narrator, y_pred_narrator)

    # Book prediction
    y_pred_book = leave_one_out_cv(X, y_book)
    ba_book = balanced_accuracy_score(y_book, y_pred_book)

    return {
        'narrator_ba': ba_narrator,
        'book_ba': ba_book,
        'difference': ba_book - ba_narrator,
        'interpretation': 'Possible confound' if ba_book > ba_narrator + 0.1 else 'No strong confound'
    }
```

---

## Timeline Estimate

| Phase | Items | Estimated Effort |
|-------|-------|------------------|
| Phase 1 | 1.1-1.5 | 4-6 hours coding |
| Phase 2 | 2.1-2.4 | 4-6 hours coding |
| Testing | All | 2-3 hours |
| Documentation | All | 2-3 hours |

**Total:** ~15-20 hours

---

## Output Structure

The enhanced script will produce:

```
results/
├── run-aggregated-results.json      # Full results
├── run-aggregated-report.md         # Markdown report
├── figures/
│   ├── permutation-null-dist.png    # Null distribution
│   ├── confusion-matrix.png         # Confusion matrix
│   ├── per-class-metrics.png        # Per-class bar chart
│   ├── feature-sensitivity.png      # Feature set sensitivity
│   └── mde-power-curve.png          # MDE analysis
└── tables/
    ├── per-class-metrics.csv        # Detailed metrics
    └── sensitivity-results.csv      # All sensitivity analyses
```

---

## Decision Points

Before implementation, confirm:

1. **BCa vs Percentile CI:** BCa is better but more complex. Implement BCa?
2. **MDE simulations:** 200 simulations × 1000 permutations = expensive. Reduce?
3. **Segment-length:** Do we have raw text access, or only pre-aggregated blocks?
4. **Book labels:** What book labels are available in the data?

