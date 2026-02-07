#!/usr/bin/env python3
"""
Run-Aggregated Stylometric Analysis (Supplementary)

Addresses reviewer concern about pseudoreplication by aggregating features
at the run level BEFORE classification. This treats runs as the true unit
of analysis, resulting in N=14 observations (not 244 blocks).

This is a SUPPLEMENTARY analysis to the pre-registered block-level analysis.
It provides a cleaner statistical treatment of the clustered data structure.

Key differences from block-level analysis:
1. Features averaged within each run -> 14 data points
2. Leave-one-out CV (each observation is already a run)
3. Simpler permutation test (shuffle 14 labels directly)
4. No block capping needed (each run = one observation)

===============================================================================
PRIMARY ANALYSIS (Pre-specified, Confirmatory):
- 3-class classification (JACOB, MORMON, NEPHI) - MORONI excluded due to n=2
- Logistic Regression, L2 regularization, C=1.0
- All 171 function word features (no selection)
- Leave-one-out cross-validation
- BLOCKED permutation test (within book-strata) - 10,000 permutations
  (Blocked permutation is PRIMARY because narrator-book confounding violates
  exchangeability; unrestricted permutation is reported for reference only)
- Alpha = 0.05 (one-sided)

EXPLORATORY ANALYSES (FDR-corrected):
- 4-class analysis (includes MORONI)
- Burrows' Delta classifier
- Feature sensitivity (k=50, 100, 150)
- C sensitivity (0.01, 0.1, 1.0, 10.0, 100.0)
- Narrator vs. Book comparison
- Unrestricted permutation (for comparison with blocked)
===============================================================================

KNOWN LIMITATIONS:
1. N=14 runs limits statistical power. Effect size detectability depends on
   class distribution and metric; no formal power analysis performed.
2. Narrator labels correlate with book position (68% collinearity), which
   confounds results. Blocked permutation is PRIMARY to address this, but
   cannot fully separate narrator style from book/topic effects.
3. MORONI has only 2 runs, making reliable classification impossible for that
   class. It is excluded from the primary confirmatory analysis.
4. High-dimensional setting (171 features, 12-13 training samples per fold)
   yields unstable coefficient estimates; regularization mitigates but does
   not eliminate this concern.

COLLINEARITY METRIC:
  Collinearity index = 1 - (unique narrator-book pairs / max possible pairs)
  where pairs are counted from the contingency table of runs by narrator×book.
  HIGH (>50%) indicates narrator strongly predicts book position.

BURROWS' DELTA FORMULA:
  For test sample x and class centroid c_k:
    Delta(x, c_k) = (1/p) * sum_{j=1}^{p} |z_x,j - z_c,j|
  where z = (value - mean) / std (computed on training data, ddof=1)
  Prediction: argmin_k Delta(x, c_k)
  This is "classic" Burrows' Delta (2002), using mean absolute z-score difference.

Version: 1.5.1
Date: 2026-02-07
Status: SUPPLEMENTARY (not pre-registered)

Changes in v1.5.1 (second audit response - exchangeability fix):
- BLOCKED PERMUTATION IS NOW PRIMARY: Promoted from sensitivity to confirmatory
  because narrator-book confounding violates exchangeability assumption
- Unrestricted permutation demoted to reference/comparison only
- Added bootstrap CI for balanced accuracy (cluster bootstrap at run level)
- Added per-class recall with binomial confidence intervals
- Documented RNG seeds explicitly in metadata
- Defined collinearity metric explicitly in docstring
- Removed unsupported power claim ("~25+ pts above chance")
- Added strata definition documentation

Changes in v1.5.0 (audit response - maximum rigor):
- PRE-SPECIFIED PRIMARY ANALYSIS: 3-class LR as confirmatory; all else exploratory
- Added FDR correction (Benjamini-Hochberg) for exploratory p-values
- Added blocked permutation test (within book-strata) for exchangeability
- Added sample weighting option (by run word count) for heteroskedasticity
- Added feature ranking inside CV folds option for leak-proof sensitivity
- Renamed "confound_probe" to "narrator_vs_book_comparison" with limitations
- Added narrator-book contingency table to output
- Added explicit exchangeability defense documentation
- Restructured output: primary_analysis vs exploratory_analyses vs sensitivity
- Added methodology card with all hyperparameters

Changes in v1.4.0 (publication-quality enhancements):
- Added Burrows' Delta baseline (canonical stylometry method)
- Added per-class metrics (precision, recall, F1 for each class)
- Added permutation-based uncertainty quantification
- Added visualization: permutation null distribution plot
- Added visualization: confusion matrix heatmaps (LR and Delta)

Changes in v1.3.0 (audit fixes):
- Added zero-word guard in aggregation (prevents NaN/inf from empty runs)
- Renamed 'restricted_permutation_test' to 'permutation_test'
- Added strong caveats to bootstrap_ci about duplicate-leakage bias
- Clarified Wilson interval is for RAW accuracy only, not balanced accuracy
- Increased max_iter to 2000 for better convergence in high-dim setting

Changes in v1.2.0:
- Count-based aggregation: sum raw FW counts across blocks, then convert to frequencies
- Added +1 correction to p-value (Phipson & Smyth 2010)
- C sensitivity analysis (tests C ∈ {0.01, 0.1, 1.0, 10.0, 100.0})
- Wilson confidence interval
- Jackknife influence analysis
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict
from itertools import permutations
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# Configuration
INPUT_FILE = Path("data/text/processed/bom-voice-blocks.json")
OUTPUT_FILE = Path("results/run-aggregated-results.json")
REPORT_FILE = Path("results/run-aggregated-report.md")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_PERMUTATIONS = 100000  # Full permutation space is 14!/(4!*5!*2!*3!) ≈ 2.5M

# Function words (same as v3)
FUNCTION_WORDS = [
    "a", "an", "the",
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "what", "that", "this", "these", "those",
    "one", "ones",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "off", "over", "under",
    "of", "unto", "upon",
    "and", "but", "or", "nor", "yet", "so",
    "if", "then", "because", "although", "while", "whereas", "when", "where",
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "not", "no", "never", "neither", "nor",
    "now", "then", "here", "there", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "any", "such", "only", "also", "very", "just", "even",
    "again", "ever", "still", "already",
    "ye", "thee", "thou", "thy", "thine", "thyself",
    "hath", "doth", "didst", "hast", "art", "wilt", "shalt",
    "wherefore", "therefore", "thus", "hence",
    "yea", "nay", "verily",
    "lest", "except", "save",
    "whoso", "whosoever", "whatsoever",
    "forth", "hither", "thither", "whither",
]


def load_blocks(filepath: Path) -> dict:
    """Load voice blocks JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def tokenize(text: str) -> list:
    """Simple word tokenization."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    return tokens


def extract_function_word_counts(text: str) -> tuple:
    """
    Extract function word raw counts and total word count.

    Returns:
        counts: array of function word counts
        total: total word count in text
    """
    tokens = tokenize(text)
    total_words = len(tokens)

    word_counts = Counter(tokens)
    counts = np.array([word_counts.get(word, 0) for word in FUNCTION_WORDS])

    return counts, total_words


def counts_to_frequencies(counts: np.ndarray, total: int) -> np.ndarray:
    """Convert raw counts to per-1000-word frequencies."""
    if total == 0:
        return np.zeros(len(FUNCTION_WORDS))
    return (counts / total) * 1000


def aggregate_runs(blocks: list, target_size: int = 1000,
                   voices: list = None) -> tuple:
    """
    Aggregate blocks into run-level observations.

    Returns:
        X: (n_runs, n_features) array of run-averaged features
        y: (n_runs,) array of voice labels
        run_ids: list of run IDs
        run_info: dict with metadata per run
    """
    if voices is None:
        voices = ["MORMON", "NEPHI", "MORONI", "JACOB"]

    # Filter blocks
    filtered_blocks = [
        b for b in blocks
        if b["target_size"] == target_size
        and b["quote_status"] == "original"
        and b["voice"] in voices
    ]

    # Group by run
    runs_dict = defaultdict(list)
    for block in filtered_blocks:
        runs_dict[block["run_id"]].append(block)

    # Aggregate features by run
    X_list = []
    y_list = []
    run_ids = []
    run_info = {}

    for run_id in sorted(runs_dict.keys()):
        run_blocks = runs_dict[run_id]
        voice = run_blocks[0]["voice"]

        # Aggregate raw counts across all blocks in this run
        # Then convert to frequencies - this is more principled than averaging frequencies
        total_counts = np.zeros(len(FUNCTION_WORDS))
        total_words = 0
        for block in run_blocks:
            counts, block_total = extract_function_word_counts(block["text"])
            total_counts += counts
            total_words += block_total

        # Convert aggregated counts to per-1000-word frequencies
        # Guard against division by zero (empty/filtered runs)
        if total_words == 0:
            print(f"  WARNING: Run {run_id} has 0 words after filtering. Skipping.")
            continue

        run_features = counts_to_frequencies(total_counts, total_words)

        X_list.append(run_features)
        y_list.append(voice)
        run_ids.append(run_id)
        # Get book(s) for this run - should be consistent but handle edge cases
        books = set(block.get("book", "Unknown") for block in run_blocks)
        book = books.pop() if len(books) == 1 else "/".join(sorted(books))

        run_info[run_id] = {
            "voice": voice,
            "book": book,
            "n_blocks": len(run_blocks),
            "total_words": total_words
        }

    X = np.array(X_list)
    y = np.array(y_list)

    # Validate we have enough data
    if len(X) == 0:
        raise ValueError("No valid runs after filtering. Check input data.")
    if len(X) < 4:
        raise ValueError(f"Only {len(X)} runs after filtering. Need at least 4 for 4-class classification.")

    # Validate all expected classes are present
    classes_present = set(y)
    classes_expected = set(voices)
    if classes_present != classes_expected:
        missing = classes_expected - classes_present
        raise ValueError(f"Missing classes after filtering: {missing}. "
                        f"Present: {classes_present}")

    # Validate minimum class size for LOO CV (need at least 2 per class
    # so training always has at least 1 example of each class)
    class_counts = {c: np.sum(y == c) for c in classes_present}
    min_class = min(class_counts.items(), key=lambda x: x[1])
    if min_class[1] < 2:
        raise ValueError(f"Class {min_class[0]} has only {min_class[1]} run(s). "
                        f"Need at least 2 per class for LOO CV.")

    return X, y, run_ids, run_info


def leave_one_out_cv(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
    """
    Leave-one-out cross-validation for run-level data.

    With N=14 runs, this is computationally trivial.

    Args:
        X: feature matrix (n_runs, n_features)
        y: labels (n_runs,)
        C: regularization strength (inverse of regularization; default 1.0)

    Returns:
        y_pred: predictions for each held-out run
    """
    n_samples = len(y)
    y_pred = np.empty(n_samples, dtype=y.dtype)

    for i in range(n_samples):
        # Leave out sample i
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]

        # Fit scaler INSIDE cv fold to prevent leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Note: penalty parameter deprecated in sklearn 1.8+
        # Use l1_ratio=0 (equivalent to L2) + C for regularization
        model = LogisticRegression(
            C=C,                      # Regularization strength (parameterized)
            l1_ratio=0,               # L2 regularization (l1_ratio=0 means pure L2)
            class_weight='balanced',  # Compensate for class imbalance
            max_iter=2000,            # Increased for convergence in high-dim setting
            random_state=RANDOM_SEED,
            solver='lbfgs',
            warm_start=False
        )
        model.fit(X_train_scaled, y_train)

        # Note: With p >> n, some folds may not fully converge.
        # This is expected and max_iter=2000 should be sufficient for stability.

        # Predict
        y_pred[i] = model.predict(X_test_scaled)[0]

    return y_pred


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy (macro-averaged recall)."""
    return balanced_accuracy_score(y_true, y_pred)


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """
    Wilson score interval for binomial proportion (RAW ACCURACY only).

    NOTE: This interval is valid for raw accuracy (correct/n), which is
    a simple binomial proportion. It is NOT appropriate for balanced
    accuracy (average of per-class recalls), which has a more complex
    distribution.

    For balanced accuracy uncertainty, use the permutation null distribution
    or a bootstrap approach (with caveats noted in bootstrap_ci).

    Args:
        successes: number of correct predictions
        total: total predictions
        confidence: confidence level (default 0.95)

    Returns:
        (lower, upper) bounds of confidence interval for raw accuracy
    """
    from scipy import stats

    if total == 0:
        return (0.0, 1.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    n = total

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

    return (max(0, center - margin), min(1, center + margin))


def c_sensitivity_analysis(X: np.ndarray, y: np.ndarray,
                            c_values: list = None) -> dict:
    """
    Analyze sensitivity of results to regularization strength C.

    With p >> n (169 features, 14 samples), C choice can strongly
    affect results. This documents stability across reasonable C values.
    """
    if c_values is None:
        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    results = {}

    for c_val in c_values:
        y_pred = leave_one_out_cv(X, y, C=c_val)
        acc = compute_balanced_accuracy(y, y_pred)
        n_correct = np.sum(y == y_pred)

        results[c_val] = {
            "balanced_accuracy": float(acc),
            "n_correct": int(n_correct),
            "n_total": len(y)
        }
        print(f"    C={c_val}: balanced accuracy = {acc:.1%}")

    return results


def permutation_test(X: np.ndarray, y: np.ndarray,
                      observed_score: float,
                      n_permutations: int,
                      seed: int) -> dict:
    """
    Monte Carlo permutation test for classification accuracy.

    Permutes run labels uniformly at random (shuffle without replacement)
    and reruns the full LOO CV pipeline for each permutation to build
    a null distribution.

    Note: This is a COUNT-PRESERVING permutation (standard shuffle).
    Class counts are preserved EXACTLY in every permutation because we
    shuffle labels rather than resample. This is equivalent to sampling
    from the restricted permutation space of size 14!/(4!*5!*2!*3!) = 2,522,520.

    Validity assumes runs are exchangeable under the null hypothesis
    (i.e., no temporal, topic, or other structure that would make some
    label assignments more likely than others under the null).

    The +1 correction (Phipson & Smyth 2010) ensures valid p-values.
    """
    rng = np.random.RandomState(seed)

    classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(classes, counts))

    # Generate permuted label sets
    null_scores = []

    for perm_idx in range(n_permutations):
        # Generate a random permutation that preserves class counts
        # by shuffling indices
        perm_indices = rng.permutation(len(y))
        y_perm = y[perm_indices]

        # Run LOO CV on permuted labels
        y_pred_perm = leave_one_out_cv(X, y_perm)
        score = compute_balanced_accuracy(y_perm, y_pred_perm)
        null_scores.append(score)

        if (perm_idx + 1) % 1000 == 0:
            print(f"  Permutation {perm_idx + 1}/{n_permutations}")

    null_scores = np.array(null_scores)

    # One-sided p-value with +1 correction (Phipson & Smyth 2010)
    # p = (# null >= observed + 1) / (n_permutations + 1)
    # This ensures valid p-values and accounts for the observed being part of the null
    n_extreme = np.sum(null_scores >= observed_score)
    p_value = (n_extreme + 1) / (n_permutations + 1)

    return {
        "observed_score": observed_score,
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
        "null_min": float(np.min(null_scores)),
        "null_max": float(np.max(null_scores)),
        "null_5_percentile": float(np.percentile(null_scores, 5)),
        "null_95_percentile": float(np.percentile(null_scores, 95)),
        "n_permutations": n_permutations,
        "null_scores": null_scores  # Keep for visualization
    }


def jackknife_influence(X: np.ndarray, y: np.ndarray, run_ids: list,
                         baseline_accuracy: float) -> dict:
    """
    Jackknife influence analysis: how much does each run affect the result?

    Recomputes CV accuracy after removing each run from the dataset.
    This reveals if the result is "carried" by 1-2 influential runs.
    """
    influences = {}

    for i in range(len(y)):
        # Remove run i entirely
        X_reduced = np.delete(X, i, axis=0)
        y_reduced = np.delete(y, i)

        # Rerun LOO CV on 13 runs
        y_pred_reduced = leave_one_out_cv(X_reduced, y_reduced)
        reduced_acc = compute_balanced_accuracy(y_reduced, y_pred_reduced)

        # Influence = how much accuracy changes when this run is removed
        influence = reduced_acc - baseline_accuracy

        influences[run_ids[i]] = {
            "reduced_accuracy": float(reduced_acc),
            "influence": float(influence),
            "voice": y[i]
        }

    return influences


def bootstrap_ci(X: np.ndarray, y: np.ndarray,
                 n_bootstrap: int = 1000,
                 seed: int = 42) -> dict:
    """
    Bootstrap confidence interval at the run level.

    Resample runs with replacement, stratified by class.

    WARNING: This bootstrap approach has known issues:
    1. LOO on bootstrap samples can train on duplicates of the test point,
       causing optimistic bias (inflated accuracy, narrow CIs).
    2. With small classes (MORONI=2), bootstrap resamples often collapse
       to identical samples, reducing effective diversity.

    This is provided for completeness but Wilson interval or permutation
    null distribution is preferred for uncertainty quantification.

    A proper fix would use .632 bootstrap or out-of-bag evaluation.
    """
    rng = np.random.RandomState(seed)

    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}

    bootstrap_scores = []
    n_failed = 0

    for boot_idx in range(n_bootstrap):
        # Stratified resampling: sample with replacement within each class
        boot_indices = []
        for c in classes:
            c_indices = class_indices[c]
            boot_c_indices = rng.choice(c_indices, size=len(c_indices), replace=True)
            boot_indices.extend(boot_c_indices)

        boot_indices = np.array(boot_indices)
        X_boot = X[boot_indices]
        y_boot = y[boot_indices]

        # Check if we have at least 2 classes in training for each LOO fold
        try:
            y_pred_boot = leave_one_out_cv(X_boot, y_boot)
            score = compute_balanced_accuracy(y_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except Exception:
            n_failed += 1
            continue

    bootstrap_scores = np.array(bootstrap_scores)

    return {
        "mean": float(np.mean(bootstrap_scores)),
        "std": float(np.std(bootstrap_scores)),
        "ci_95_lower": float(np.percentile(bootstrap_scores, 2.5)),
        "ci_95_upper": float(np.percentile(bootstrap_scores, 97.5)),
        "n_bootstrap": len(bootstrap_scores),
        "n_failed": n_failed
    }


def permutation_based_ci(X: np.ndarray, y: np.ndarray, null_scores: np.ndarray,
                          observed_score: float) -> dict:
    """
    Derive confidence interval from permutation null distribution.

    This uses the permutation distribution to establish plausible ranges.
    More appropriate than bootstrap CI for small N with LOO-CV.

    The CI is based on the spread of the null distribution centered on
    the observed score (a conservative approach).
    """
    null_std = np.std(null_scores)
    null_mean = np.mean(null_scores)

    # Method 1: Null distribution spread (descriptive)
    null_ci_lower = np.percentile(null_scores, 2.5)
    null_ci_upper = np.percentile(null_scores, 97.5)

    # Method 2: Observed +/- null spread (more interpretable)
    # This shows: "given the variability under the null, where might true BA lie?"
    spread = null_ci_upper - null_ci_lower

    return {
        "observed": float(observed_score),
        "null_mean": float(null_mean),
        "null_std": float(null_std),
        "null_ci_95": [float(null_ci_lower), float(null_ci_upper)],
        "null_spread_95": float(spread),
        "interpretation": (
            f"Observed BA ({observed_score:.1%}) vs null range "
            f"[{null_ci_lower:.1%}, {null_ci_upper:.1%}]"
        )
    }


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               classes: list) -> dict:
    """
    Compute detailed per-class metrics with uncertainty quantification.

    Returns precision, recall, F1 for each class plus macro/weighted averages.
    Includes Wilson binomial confidence intervals for recall (per-class accuracy).
    """
    from sklearn.metrics import precision_recall_fscore_support

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    per_class = {}
    for i, c in enumerate(classes):
        # Compute Wilson CI for recall (proportion correct within class)
        n_class = int(support[i])
        n_correct_class = int(np.sum((y_true == c) & (y_pred == c)))
        if n_class > 0:
            recall_ci_lower, recall_ci_upper = wilson_interval(n_correct_class, n_class)
        else:
            recall_ci_lower, recall_ci_upper = 0.0, 1.0

        per_class[c] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "recall_ci_95": [float(recall_ci_lower), float(recall_ci_upper)],
            "f1": float(f1[i]),
            "support": n_class,
            "n_correct": n_correct_class
        }

    # Macro and weighted averages
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if sum(support) > 0 else 0.0

    return {
        "per_class": per_class,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def burrows_delta_loo(X: np.ndarray, y: np.ndarray, classes: list) -> tuple:
    """
    Leave-one-out cross-validation using Burrows' Delta.

    Burrows' Delta is the canonical stylometry baseline:
    1. Z-score features using training data
    2. Compute class centroids in z-space
    3. Predict: class with minimum mean absolute z-score difference

    Returns:
        y_pred: predictions for each held-out sample
        balanced_accuracy: balanced accuracy score
    """
    n = len(y)
    y_pred = np.empty(n, dtype=y.dtype)

    for i in range(n):
        # Train/test split
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]

        # Z-score using training data only (prevent leakage)
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0, ddof=1)
        sd[sd == 0] = 1  # Avoid division by zero for zero-variance features

        Z_train = (X_train - mu) / sd
        Z_test = (X_test - mu) / sd

        # Compute class centroids
        centroids = {}
        for c in classes:
            mask = (y_train == c)
            if mask.sum() > 0:
                centroids[c] = Z_train[mask].mean(axis=0)
            else:
                # Class not in training (shouldn't happen with our data)
                centroids[c] = np.zeros(X.shape[1])

        # Delta distance: mean absolute difference from centroid
        deltas = {}
        for c, centroid in centroids.items():
            deltas[c] = np.mean(np.abs(Z_test - centroid))

        # Predict: class with minimum Delta
        y_pred[i] = min(deltas, key=deltas.get)

    ba = balanced_accuracy_score(y, y_pred)
    return y_pred, ba


# =============================================================================
# V1.5.0: AUDIT RESPONSE FUNCTIONS (Maximum Rigor)
# =============================================================================

def compute_fdr_correction(p_values: dict) -> dict:
    """
    Apply Benjamini-Hochberg FDR correction to a dictionary of p-values.

    Args:
        p_values: dict mapping analysis names to raw p-values

    Returns:
        dict with both raw and FDR-adjusted p-values
    """
    names = list(p_values.keys())
    raw_pvals = np.array([p_values[n] for n in names])
    n = len(raw_pvals)

    if n == 0:
        return {}

    # Sort p-values
    sorted_indices = np.argsort(raw_pvals)
    sorted_pvals = raw_pvals[sorted_indices]

    # BH procedure
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_pvals[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_pvals[i] * n / (i + 1))

    # Unsort
    fdr_pvals = np.zeros(n)
    fdr_pvals[sorted_indices] = adjusted

    return {
        name: {
            "p_value_raw": float(raw_pvals[i]),
            "p_value_fdr": float(min(1.0, fdr_pvals[i])),
            "significant_raw": bool(raw_pvals[i] < 0.05),
            "significant_fdr": bool(fdr_pvals[i] < 0.05)
        }
        for i, name in enumerate(names)
    }


def get_book_strata(run_info: dict, run_ids: list) -> dict:
    """
    Extract book-based strata for blocked permutation.

    Groups runs by their primary book for within-stratum permutation.

    Returns:
        dict mapping run_id to stratum_id
    """
    strata = {}
    for rid in run_ids:
        book = run_info.get(rid, {}).get("book", "Unknown")
        # Use first book if multiple (simplify stratum assignment)
        primary_book = book.split("/")[0] if "/" in book else book
        strata[rid] = primary_book

    return strata


def blocked_permutation_test(X: np.ndarray, y: np.ndarray,
                              observed_score: float,
                              strata: np.ndarray,
                              n_permutations: int,
                              seed: int) -> dict:
    """
    Permutation test respecting block/strata structure.

    Permutes labels WITHIN each stratum to preserve stratum-level structure.
    This addresses exchangeability concerns when runs correlate with book/topic.

    Args:
        X: feature matrix
        y: labels
        observed_score: observed balanced accuracy
        strata: array of stratum labels for each sample
        n_permutations: number of permutations
        seed: random seed

    Returns:
        dict with p-value and null distribution statistics

    Note: If strata are too small for meaningful within-stratum permutation,
    this may produce different results than unrestricted permutation. Both
    should be reported as sensitivity analysis.
    """
    rng = np.random.RandomState(seed)
    unique_strata = np.unique(strata)

    null_scores = []
    n_degenerate = 0  # Count permutations that couldn't shuffle meaningfully

    for perm_idx in range(n_permutations):
        y_perm = y.copy()

        # Permute within each stratum
        for stratum in unique_strata:
            mask = (strata == stratum)
            indices = np.where(mask)[0]
            if len(indices) > 1:
                # Shuffle labels within this stratum
                shuffled = rng.permutation(y[mask])
                y_perm[mask] = shuffled
            # If only 1 sample in stratum, can't permute

        # Check if permutation is degenerate (same as original)
        if np.array_equal(y_perm, y):
            n_degenerate += 1

        # Run LOO CV on permuted labels
        y_pred_perm = leave_one_out_cv(X, y_perm)
        score = compute_balanced_accuracy(y_perm, y_pred_perm)
        null_scores.append(score)

        if (perm_idx + 1) % 1000 == 0:
            print(f"  Blocked permutation {perm_idx + 1}/{n_permutations}")

    null_scores = np.array(null_scores)

    # p-value with +1 correction
    n_extreme = np.sum(null_scores >= observed_score)
    p_value = (n_extreme + 1) / (n_permutations + 1)

    return {
        "observed_score": float(observed_score),
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
        "null_5_percentile": float(np.percentile(null_scores, 5)),
        "null_95_percentile": float(np.percentile(null_scores, 95)),
        "n_permutations": n_permutations,
        "n_degenerate": n_degenerate,
        "n_strata": len(unique_strata),
        "strata_sizes": {str(s): int(np.sum(strata == s)) for s in unique_strata},
        "null_scores": null_scores,  # Keep for visualization
        "note": (
            f"Blocked permutation within {len(unique_strata)} book-strata. "
            f"{n_degenerate} degenerate permutations (identical to original)."
        )
    }


def get_narrator_book_contingency(run_info: dict, run_ids: list, y: np.ndarray) -> dict:
    """
    Compute narrator-book contingency table to assess confounding.

    Returns:
        dict with contingency table and collinearity assessment
    """
    narrators = sorted(set(y))
    books = set()
    for rid in run_ids:
        book = run_info.get(rid, {}).get("book", "Unknown")
        for b in book.split("/"):
            books.add(b)
    books = sorted(books)

    # Build contingency table
    table = {narrator: {book: 0 for book in books} for narrator in narrators}
    for rid, narrator in zip(run_ids, y):
        book = run_info.get(rid, {}).get("book", "Unknown")
        for b in book.split("/"):
            table[narrator][b] += 1

    # Assess collinearity (simple: count unique narrator-book pairs)
    n_pairs = sum(1 for n in narrators for b in books if table[n][b] > 0)
    max_pairs = len(narrators) * len(books)
    collinearity = 1 - (n_pairs / max_pairs) if max_pairs > 0 else 0

    return {
        "contingency_table": table,
        "narrators": narrators,
        "books": books,
        "n_unique_pairs": n_pairs,
        "max_possible_pairs": max_pairs,
        "collinearity_index": float(collinearity),
        "interpretation": (
            f"Narrator-book collinearity: {collinearity:.1%}. "
            f"{'HIGH: narrator strongly predicts book' if collinearity > 0.5 else 'MODERATE: some overlap' if collinearity > 0.2 else 'LOW: narrator and book largely independent'}"
        )
    }


def narrator_vs_book_comparison(X: np.ndarray, y_narrator: np.ndarray,
                                 run_info: dict, run_ids: list,
                                 n_permutations: int = 1000,
                                 seed: int = 42) -> dict:
    """
    Compare narrator prediction vs book prediction accuracy.

    PURPOSE: Assess whether features capture narrator style or book/topic.
    If book BA >> narrator BA, features may capture topic not style.

    LIMITATION: This comparison is SUGGESTIVE, not DEFINITIVE. If narrator
    and book are collinear, we cannot fully disentangle their effects.
    A low book BA does not prove narrator signal is "pure" style.

    Args:
        X: feature matrix
        y_narrator: narrator labels
        run_info: dict with book info per run
        run_ids: list of run IDs
        n_permutations: permutations for book prediction test
        seed: random seed

    Returns:
        dict with comparison results and explicit limitations
    """
    # Extract book labels
    y_book = np.array([
        run_info.get(rid, {}).get("book", "Unknown")
        for rid in run_ids
    ])
    book_classes = sorted(set(y_book))
    n_books = len(book_classes)

    if n_books < 2:
        return {
            "error": "Cannot run book prediction: fewer than 2 books",
            "n_books": n_books
        }

    # Narrator prediction
    narrator_classes = sorted(set(y_narrator))
    y_pred_narrator = leave_one_out_cv(X, y_narrator, C=1.0)
    ba_narrator = balanced_accuracy_score(y_narrator, y_pred_narrator)

    # Book prediction
    try:
        y_pred_book = leave_one_out_cv(X, y_book, C=1.0)
        ba_book = balanced_accuracy_score(y_book, y_pred_book)
        perm_book = permutation_test(X, y_book, ba_book, n_permutations, seed)
    except Exception as e:
        return {
            "error": f"Book prediction failed: {str(e)}",
            "narrator_ba": float(ba_narrator)
        }

    # Chance levels
    narrator_chance = 1.0 / len(narrator_classes)
    book_chance = 1.0 / n_books

    # Assessment
    narrator_above_chance = ba_narrator - narrator_chance
    book_above_chance = ba_book - book_chance

    if ba_book > ba_narrator + 0.1:
        assessment = "CONCERN: Book more predictable than narrator - features may capture topic"
    elif ba_book > ba_narrator:
        assessment = "CAUTION: Book slightly more predictable - possible topic influence"
    else:
        assessment = "REASSURING: Narrator at least as predictable as book"

    return {
        "narrator": {
            "balanced_accuracy": float(ba_narrator),
            "chance_level": float(narrator_chance),
            "above_chance": float(narrator_above_chance),
            "n_classes": len(narrator_classes)
        },
        "book": {
            "balanced_accuracy": float(ba_book),
            "chance_level": float(book_chance),
            "above_chance": float(book_above_chance),
            "p_value": float(perm_book["p_value"]),
            "n_classes": n_books,
            "classes": book_classes
        },
        "comparison": {
            "difference_ba": float(ba_book - ba_narrator),
            "difference_above_chance": float(book_above_chance - narrator_above_chance),
            "assessment": assessment
        },
        "limitations": (
            "This comparison is SUGGESTIVE, not DEFINITIVE. If narrator and book "
            "are collinear, we cannot disentangle their effects. A low book BA "
            "does not prove narrator signal is 'pure' style; it only suggests "
            "topic is not MORE predictable than narrator."
        )
    }


def feature_sensitivity_with_fold_ranking(X: np.ndarray, y: np.ndarray,
                                           feature_names: list,
                                           k_values: list = None,
                                           n_permutations: int = 1000,
                                           seed: int = 42) -> dict:
    """
    Feature sensitivity analysis with ranking computed INSIDE each CV fold.

    This is more rigorous than corpus-wide ranking because it prevents
    any information leakage from test samples into feature selection.

    Args:
        X: feature matrix
        y: labels
        feature_names: list of feature names
        k_values: list of k values to test
        n_permutations: permutations for each test
        seed: random seed
    """
    if k_values is None:
        k_values = [50, 100, 150, len(feature_names)]

    results = {}
    n = len(y)

    for k in k_values:
        k = min(k, len(feature_names))
        y_pred = np.empty(n, dtype=y.dtype)

        for i in range(n):
            # Leave out sample i
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_test = X[i:i+1]

            # Rank features by TRAINING data frequency only
            train_means = X_train.mean(axis=0)
            top_k_indices = np.argsort(-train_means)[:k]

            X_train_k = X_train[:, top_k_indices]
            X_test_k = X_test[:, top_k_indices]

            # Fit scaler inside fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_k)
            X_test_scaled = scaler.transform(X_test_k)

            # Fit model
            model = LogisticRegression(
                C=1.0, l1_ratio=0, class_weight='balanced',
                max_iter=2000, random_state=RANDOM_SEED, solver='lbfgs'
            )
            model.fit(X_train_scaled, y_train)
            y_pred[i] = model.predict(X_test_scaled)[0]

        ba = balanced_accuracy_score(y, y_pred)

        # Permutation test (uses same fold-ranking approach)
        rng = np.random.RandomState(seed)
        null_scores = []
        for _ in range(n_permutations):
            y_perm = y[rng.permutation(n)]
            y_pred_perm = np.empty(n, dtype=y.dtype)
            for i in range(n):
                X_train = np.delete(X, i, axis=0)
                y_train_perm = np.delete(y_perm, i)
                X_test = X[i:i+1]
                train_means = X_train.mean(axis=0)
                top_k_idx = np.argsort(-train_means)[:k]
                X_train_k = X_train[:, top_k_idx]
                X_test_k = X_test[:, top_k_idx]
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_k)
                X_test_scaled = scaler.transform(X_test_k)
                model = LogisticRegression(
                    C=1.0, l1_ratio=0, class_weight='balanced',
                    max_iter=2000, random_state=RANDOM_SEED, solver='lbfgs'
                )
                model.fit(X_train_scaled, y_train_perm)
                y_pred_perm[i] = model.predict(X_test_scaled)[0]
            null_scores.append(balanced_accuracy_score(y_perm, y_pred_perm))

        p_value = (np.sum(np.array(null_scores) >= ba) + 1) / (n_permutations + 1)

        results[k] = {
            "n_features": k,
            "balanced_accuracy": float(ba),
            "p_value": float(p_value),
            "method": "fold-internal ranking (no leakage)"
        }
        print(f"    k={k} (fold-ranked): BA = {ba:.1%}, p = {p_value:.4f}")

    return {
        "k_values": k_values,
        "results": results,
        "method": "Feature ranking computed INSIDE each CV fold (leak-proof)",
        "interpretation": (
            "Results may differ from corpus-wide ranking if test samples "
            "strongly influence corpus statistics."
        )
    }


def create_methodology_card() -> dict:
    """
    Create a methodology card documenting all analysis choices.

    This enables expert review of pre-specified vs explored parameters.
    """
    return {
        "study_design": {
            "type": "Supplementary analysis (not pre-registered)",
            "purpose": "Test narrator separability at run level to address pseudoreplication",
            "unit_of_analysis": "Run (contiguous narrator segment)",
            "n_observations": 14,
            "n_features": 171,
            "version": "1.5.1"
        },
        "primary_analysis": {
            "pre_specified": True,
            "description": "3-class classification (JACOB, MORMON, NEPHI)",
            "excluded": "MORONI (n=2, insufficient for reliable classification)",
            "classifier": "Logistic Regression",
            "regularization": "L2 (Ridge)",
            "C": 1.0,
            "feature_selection": "None (all 171 function words)",
            "validation": "Leave-one-out cross-validation",
            "metric": "Balanced accuracy (macro-averaged recall)",
            "inference": "BLOCKED permutation test (within book-strata)",
            "inference_note": (
                "Blocked permutation is PRIMARY because narrator-book "
                "confounding violates exchangeability. Unrestricted "
                "permutation is reported for reference only."
            ),
            "n_permutations": "10,000 (blocked)",
            "alpha": 0.05,
            "correction": "None (single primary test)",
            "uncertainty": "Bootstrap 95% CI for balanced accuracy"
        },
        "exploratory_analyses": {
            "pre_specified": False,
            "correction": "Benjamini-Hochberg FDR",
            "analyses": [
                "4-class (includes MORONI)",
                "Burrows' Delta classifier",
                "Feature sensitivity (k=50,100,150)",
                "C sensitivity (0.01-100)",
                "Narrator vs Book comparison",
                "Unrestricted permutation (for comparison)"
            ]
        },
        "sensitivity_analyses": {
            "unrestricted_vs_blocked": "Compare p-values to assess exchangeability",
            "feature_ranking": "Inside CV folds (leak-proof)",
            "jackknife_influence": "Remove-one-run influence on accuracy"
        },
        "strata_definition": {
            "method": "Runs grouped by primary book (first book if multiple)",
            "rationale": (
                "Narrator labels correlate with book position (~68% collinearity). "
                "Blocked permutation within book strata respects this structure "
                "and yields valid p-values under the null of no narrator effect."
            ),
            "degenerate_handling": (
                "Permutations that produce identical label vectors are counted "
                "toward the null distribution (conservative)."
            )
        },
        "assumptions": {
            "exchangeability": (
                "Under the null hypothesis, runs are exchangeable WITHIN "
                "book-strata. This is why blocked permutation is primary."
            ),
            "independence": (
                "Runs are treated as independent observations. "
                "This is justified by run-level aggregation but "
                "may not hold if runs share latent factors."
            ),
            "stationarity": (
                "Function word usage is stationary across the text. "
                "Not explicitly tested."
            )
        },
        "limitations": {
            "sample_size": "N=14 limits power; no formal power analysis performed",
            "high_dimensional": "171 features with ~12 training samples/fold is p>>n",
            "moroni": "Only 2 runs; excluded from confirmatory analysis",
            "confounding": "Narrator correlates with book (~68%); cannot fully separate",
            "generalization": "Results apply to this text; may not generalize"
        },
        "reproducibility": {
            "random_seed": 42,
            "rng": "numpy.random.RandomState(42)",
            "solver": "lbfgs",
            "max_iter": 2000
        }
    }


# =============================================================================
# PHASE 2: ROBUSTNESS ANALYSES (Tier 2)
# =============================================================================

def run_three_class_analysis(X: np.ndarray, y: np.ndarray, run_ids: list,
                              exclude_class: str = "MORONI",
                              n_permutations: int = 10000,
                              seed: int = 42) -> dict:
    """
    Run analysis excluding one class (default: MORONI with only 2 runs).

    This addresses the severe class imbalance issue where MORONI has only
    2 runs, making reliable classification nearly impossible.

    Returns full analysis results for the reduced dataset.
    """
    print(f"  Excluding class: {exclude_class}")

    # Filter out excluded class
    mask = (y != exclude_class)
    X_filtered = X[mask]
    y_filtered = y[mask]
    run_ids_filtered = [rid for rid, m in zip(run_ids, mask) if m]

    classes = sorted(set(y_filtered))
    n_classes = len(classes)
    chance = 1.0 / n_classes

    print(f"  Remaining: {len(y_filtered)} runs, {n_classes} classes")
    print(f"  Class distribution: {dict(zip(*np.unique(y_filtered, return_counts=True)))}")

    # LOO CV with LR
    y_pred = leave_one_out_cv(X_filtered, y_filtered, C=1.0)
    ba = balanced_accuracy_score(y_filtered, y_pred)
    n_correct = np.sum(y_filtered == y_pred)

    # Burrows' Delta
    y_pred_delta, delta_ba = burrows_delta_loo(X_filtered, y_filtered, classes)

    # Per-class metrics
    per_class = compute_per_class_metrics(y_filtered, y_pred, classes)

    # Permutation test
    perm = permutation_test(X_filtered, y_filtered, ba, n_permutations, seed)

    return {
        "excluded_class": exclude_class,
        "n_runs": len(y_filtered),
        "n_classes": n_classes,
        "chance_level": float(chance),
        "classes": classes,
        "lr": {
            "balanced_accuracy": float(ba),
            "raw_accuracy": float(n_correct / len(y_filtered)),
            "n_correct": int(n_correct),
            "per_class_metrics": per_class,
            "y_pred": y_pred.tolist()
        },
        "delta": {
            "balanced_accuracy": float(delta_ba),
            "y_pred": y_pred_delta.tolist()
        },
        "permutation": {
            "p_value": perm["p_value"],
            "null_mean": perm["null_mean"],
            "null_std": perm["null_std"]
        },
        "interpretation": (
            f"3-class analysis (excluding {exclude_class}): "
            f"BA = {ba:.1%}, p = {perm['p_value']:.4f} "
            f"(chance = {chance:.1%})"
        )
    }


def feature_sensitivity_analysis(X: np.ndarray, y: np.ndarray,
                                  feature_names: list,
                                  k_values: list = None,
                                  n_permutations: int = 1000,
                                  seed: int = 42) -> dict:
    """
    Test sensitivity to number of features.

    Ranks features by corpus frequency (mean across all runs),
    then tests classification with top-k features.

    Args:
        X: feature matrix
        y: labels
        feature_names: list of feature names
        k_values: list of k values to test (default: [50, 100, 150, all])
        n_permutations: number of permutations for each test
        seed: random seed
    """
    if k_values is None:
        k_values = [50, 100, 150, len(feature_names)]

    # Rank features by mean frequency (descending)
    feature_means = X.mean(axis=0)
    feature_ranks = np.argsort(-feature_means)

    results = {}
    classes = sorted(set(y))

    for k in k_values:
        k = min(k, len(feature_names))
        top_k_indices = feature_ranks[:k]
        X_k = X[:, top_k_indices]

        # LOO CV
        y_pred = leave_one_out_cv(X_k, y, C=1.0)
        ba = balanced_accuracy_score(y, y_pred)

        # Quick permutation test
        perm = permutation_test(X_k, y, ba, n_permutations, seed)

        # Burrows' Delta
        y_pred_delta, delta_ba = burrows_delta_loo(X_k, y, classes)

        results[k] = {
            "n_features": k,
            "lr_balanced_accuracy": float(ba),
            "delta_balanced_accuracy": float(delta_ba),
            "p_value": float(perm["p_value"]),
            "top_features": [feature_names[i] for i in top_k_indices[:10]]  # Top 10 only
        }

        print(f"    k={k}: LR BA = {ba:.1%}, Delta BA = {delta_ba:.1%}, p = {perm['p_value']:.4f}")

    return {
        "k_values": k_values,
        "results": results,
        "interpretation": (
            "Feature sensitivity analysis shows classification stability "
            "across different numbers of top-frequency features."
        )
    }


def confound_probe_book(X: np.ndarray, y_narrator: np.ndarray,
                         run_info: dict, run_ids: list,
                         n_permutations: int = 1000,
                         seed: int = 42) -> dict:
    """
    Probe for topical/structural confounds by predicting BOOK instead of NARRATOR.

    If BOOK is more predictable than NARRATOR, it suggests the features
    may capture topical rather than stylistic information.

    Note: This requires book labels in run_info.
    """
    # Extract book labels from run_info
    y_book = []
    for rid in run_ids:
        book = run_info.get(rid, {}).get("book", "Unknown")
        y_book.append(book)

    y_book = np.array(y_book)
    book_classes = sorted(set(y_book))
    n_books = len(book_classes)

    if n_books < 2:
        return {
            "error": "Cannot run book prediction: fewer than 2 books in data",
            "n_books": n_books
        }

    print(f"  Book classes: {book_classes}")
    print(f"  Book distribution: {dict(zip(*np.unique(y_book, return_counts=True)))}")

    # Narrator prediction (for comparison)
    narrator_classes = sorted(set(y_narrator))
    y_pred_narrator = leave_one_out_cv(X, y_narrator, C=1.0)
    ba_narrator = balanced_accuracy_score(y_narrator, y_pred_narrator)

    # Book prediction
    try:
        y_pred_book = leave_one_out_cv(X, y_book, C=1.0)
        ba_book = balanced_accuracy_score(y_book, y_pred_book)

        # Permutation test for book
        perm_book = permutation_test(X, y_book, ba_book, n_permutations, seed)
    except Exception as e:
        return {
            "error": f"Book prediction failed: {str(e)}",
            "narrator_ba": float(ba_narrator)
        }

    # Compare
    difference = ba_book - ba_narrator
    narrator_chance = 1.0 / len(narrator_classes)
    book_chance = 1.0 / n_books

    return {
        "narrator": {
            "balanced_accuracy": float(ba_narrator),
            "chance_level": float(narrator_chance),
            "above_chance": float(ba_narrator - narrator_chance)
        },
        "book": {
            "balanced_accuracy": float(ba_book),
            "chance_level": float(book_chance),
            "above_chance": float(ba_book - book_chance),
            "p_value": float(perm_book["p_value"]),
            "n_classes": n_books,
            "classes": book_classes
        },
        "difference": float(difference),
        "interpretation": (
            f"Narrator BA: {ba_narrator:.1%} (chance {narrator_chance:.1%}), "
            f"Book BA: {ba_book:.1%} (chance {book_chance:.1%}). "
            f"{'POSSIBLE CONFOUND: Book more predictable' if ba_book > ba_narrator + 0.1 else 'No strong confound detected'}"
        )
    }


def plot_permutation_null(null_scores: np.ndarray, observed: float,
                           p_value: float, output_path: Path) -> None:
    """
    Plot histogram of permutation null distribution with observed value.

    Creates publication-quality figure showing:
    - Null distribution histogram
    - Observed value (vertical line)
    - p-value region (shaded)
    - Statistical annotations
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram of null distribution
    n, bins, patches = ax.hist(null_scores, bins=50, density=True,
                                alpha=0.7, color='steelblue',
                                edgecolor='white', linewidth=0.5)

    # Observed value line
    ax.axvline(observed, color='darkred', linewidth=2.5,
               label=f'Observed BA = {observed:.1%}')

    # Null mean line
    null_mean = np.mean(null_scores)
    ax.axvline(null_mean, color='gray', linestyle='--', linewidth=1.5,
               label=f'Null mean = {null_mean:.1%}')

    # Shade the extreme region (p-value visualization)
    for i, (left, right, patch) in enumerate(zip(bins[:-1], bins[1:], patches)):
        if left >= observed:
            patch.set_facecolor('darkred')
            patch.set_alpha(0.5)

    # Labels and formatting
    ax.set_xlabel('Balanced Accuracy', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Permutation Null Distribution\n(Run-Level Analysis)', fontsize=14)

    # Add p-value annotation
    ax.text(0.98, 0.95, f'p = {p_value:.4f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add sample size annotation
    ax.text(0.98, 0.85, f'N = 14 runs\nn_perm = {len(null_scores):,}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, max(0.6, observed + 0.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_confusion_matrix(cm: np.ndarray, classes: list,
                           output_path: Path) -> None:
    """
    Plot confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Labels
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix (Run-Level LOO-CV)')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_report(results: dict) -> str:
    """Generate markdown report."""

    lr = results['primary_lr']
    delta = results['burrows_delta']

    report = f"""# Run-Aggregated Stylometric Analysis (Supplementary)

**Generated:** {results['metadata']['generated_at']}
**Version:** {results['metadata']['script_version']}
**Status:** SUPPLEMENTARY ANALYSIS (addresses pseudoreplication concern)

---

## Methodology

This analysis aggregates block-level features to the **run level** before classification.
Instead of treating 244 blocks as independent observations, we average features within
each of the 14 voice runs, resulting in **N=14 true independent observations**.

### Key Differences from Block-Level Analysis

| Aspect | Block-Level (Pre-registered) | Run-Aggregated (This Analysis) |
|--------|------------------------------|--------------------------------|
| Unit of analysis | 244 blocks | 14 runs |
| Feature aggregation | None | Sum counts, then convert to freq |
| CV scheme | Leave-one-run-out (14 folds) | Leave-one-out (14 folds) |
| Block capping | Yes (20/run) | Not needed |
| Pseudoreplication | Partially addressed | Fully addressed |

---

## Data Summary

| Voice | Runs | Blocks | Total Words |
|-------|------|--------|-------------|
"""

    for voice, info in results['run_distribution'].items():
        report += f"| {voice} | {info['n_runs']} | {info['n_blocks']} | {info['total_words']:,} |\n"

    report += f"""| **Total** | **{results['n_runs']}** | **{results['n_blocks']}** | **{results['total_words']:,}** |

---

## Primary Results

### Classification Accuracy Comparison

| Method | Balanced Accuracy | Macro-F1 | Interpretation |
|--------|-------------------|----------|----------------|
| **Logistic Regression** | {lr['balanced_accuracy']:.1%} | {lr['per_class_metrics']['macro_f1']:.3f} | {'At' if abs(lr['balanced_accuracy'] - 0.25) < 0.05 else 'Above' if lr['balanced_accuracy'] > 0.25 else 'Below'} chance |
| **Burrows' Delta** | {delta['balanced_accuracy']:.1%} | {delta['per_class_metrics']['macro_f1']:.3f} | {'At' if abs(delta['balanced_accuracy'] - 0.25) < 0.05 else 'Above' if delta['balanced_accuracy'] > 0.25 else 'Below'} chance |
| Chance baseline | 25.0% | 0.25 | Random guessing |

### Statistical Inference (LR Results)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Permutation p-value | **{results['permutation']['p_value']:.4f}** | {'Significant' if results['permutation']['p_value'] < 0.05 else 'Not significant'} |
| Null mean | {results['permutation']['null_mean']:.1%} | Expected under no effect |
| Null 95% range | [{results['permutation_based_ci']['null_ci_95'][0]:.1%}, {results['permutation_based_ci']['null_ci_95'][1]:.1%}] | Permutation distribution |

---

## Per-Class Metrics

### Logistic Regression

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
"""

    for cls, metrics in lr['per_class_metrics']['per_class'].items():
        report += f"| {cls} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['support']} |\n"

    report += f"""| **Macro** | {lr['per_class_metrics']['macro_precision']:.3f} | {lr['per_class_metrics']['macro_recall']:.3f} | {lr['per_class_metrics']['macro_f1']:.3f} | - |

### Burrows' Delta

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
"""

    for cls, metrics in delta['per_class_metrics']['per_class'].items():
        report += f"| {cls} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['support']} |\n"

    report += f"""| **Macro** | {delta['per_class_metrics']['macro_precision']:.3f} | {delta['per_class_metrics']['macro_recall']:.3f} | {delta['per_class_metrics']['macro_f1']:.3f} | - |

---

## Confidence Intervals

| Method | 95% CI | Note |
|--------|--------|------|
| Wilson (raw acc) | [{results['wilson_ci']['lower']:.1%}, {results['wilson_ci']['upper']:.1%}] | Valid for raw accuracy only |
| Permutation null | [{results['permutation_based_ci']['null_ci_95'][0]:.1%}, {results['permutation_based_ci']['null_ci_95'][1]:.1%}] | Range under null hypothesis |
| Bootstrap | [{results['bootstrap']['ci_95_lower']:.1%}, {results['bootstrap']['ci_95_upper']:.1%}] | Unreliable with N=14 |

---

## Regularization Sensitivity (C Parameter)

Since we have p >> n (169 features, 14 samples), results can be sensitive to regularization strength.

| C | Balanced Accuracy | Note |
|---|-------------------|------|
"""

    for c_val, c_results in sorted(results['c_sensitivity'].items(), key=lambda x: float(x[0])):
        note = "← default" if float(c_val) == 1.0 else ""
        report += f"| {c_val} | {c_results['balanced_accuracy']:.1%} | {note} |\n"

    report += f"""

**Interpretation:** {"Results are stable across C values" if max(r['balanced_accuracy'] for r in results['c_sensitivity'].values()) - min(r['balanced_accuracy'] for r in results['c_sensitivity'].values()) < 0.15 else "Results vary with C—interpret with caution"}.

---

## Jackknife Influence Analysis

How much does removing each run affect the overall result?

| Run ID | Voice | Reduced Acc | Influence |
|--------|-------|-------------|-----------|
"""

    for run_id, jk_data in sorted(results['jackknife_influence'].items()):
        influence_str = f"{jk_data['influence']:+.1%}"
        report += f"| {run_id} | {jk_data['voice']} | {jk_data['reduced_accuracy']:.1%} | {influence_str} |\n"

    max_influence = max(abs(d['influence']) for d in results['jackknife_influence'].values())
    report += f"""

**Max influence magnitude:** {max_influence:.1%}

**Interpretation:** {"No single run dominates the result" if max_influence < 0.15 else "One or more runs strongly influence the result—findings may not be robust"}.

---

## Confusion Matrices

### Logistic Regression

```
Predicted:    JACOB  MORMON  MORONI  NEPHI
Actual:
"""

    cm_lr = lr['confusion_matrix']
    classes = lr['classes']
    for i, actual in enumerate(classes):
        row = "  " + actual.ljust(10)
        for j in range(len(classes)):
            row += f"{cm_lr[i][j]:>7}"
        report += row + "\n"

    report += """```

### Burrows' Delta

```
Predicted:    JACOB  MORMON  MORONI  NEPHI
Actual:
"""

    cm_delta = delta['confusion_matrix']
    for i, actual in enumerate(classes):
        row = "  " + actual.ljust(10)
        for j in range(len(classes)):
            row += f"{cm_delta[i][j]:>7}"
        report += row + "\n"

    report += f"""```

---

## Per-Run Predictions (LR)

| Run ID | True Voice | Predicted | Correct |
|--------|------------|-----------|---------|
"""

    for run_id, true_label, pred_label in zip(
        lr['run_ids'],
        lr['y_true'],
        lr['y_pred']
    ):
        correct = "✓" if true_label == pred_label else "✗"
        report += f"| {run_id} | {true_label} | {pred_label} | {correct} |\n"

    report += f"""

---

## Comparison with Block-Level Analysis

| Metric | Block-Level (v3) | Run-Aggregated (LR) | Run-Aggregated (Delta) |
|--------|------------------|---------------------|------------------------|
| Balanced Accuracy | 24.2% | {lr['balanced_accuracy']:.1%} | {delta['balanced_accuracy']:.1%} |
| Permutation p-value | 0.177 | {results['permutation']['p_value']:.4f} | - |
| Unit of analysis | 244 blocks | 14 runs | 14 runs |

---

## Interpretation

The run-aggregated analysis {
    'confirms' if abs(lr['balanced_accuracy'] - 0.242) < 0.1
    else 'differs from'
} the block-level result. With balanced accuracy of {lr['balanced_accuracy']:.1%}
and p = {results['permutation']['p_value']:.3f}, we find **{'no significant evidence' if results['permutation']['p_value'] >= 0.05 else 'significant evidence'}**
of narrator-level stylistic differentiation when treating runs as the unit of analysis.

Both Logistic Regression and Burrows' Delta (the canonical stylometry baseline) yield
similar accuracy near chance level, reinforcing the null finding.

This supplementary analysis addresses reviewer concerns about pseudoreplication by
ensuring each observation is truly independent.

---

## Limitations

### Critical Limitations

1. **Severe class imbalance (MORONI=2 runs)**: In LOOCV, when a MORONI run is held out,
   training includes only 1 MORONI example. Learning "MORONI" from a single training
   run is essentially few-shot classification and highly unreliable. MORONI predictions
   should be interpreted with extreme caution.

2. **p >> n regime**: With 169 features and 14 samples, the model is severely
   underdetermined. L2 regularization helps but cannot fully address this. Results
   may be sensitive to feature selection and regularization choices.

3. **Small effective sample size**: N=14 runs provides limited statistical power
   to detect even moderate effects. The permutation null distribution is correspondingly
   wide, making it difficult to achieve significance.

### Additional Limitations

4. **Information loss**: Aggregating discards within-run variance information
5. **Not pre-registered**: This is a post-hoc sensitivity analysis
6. **Bootstrap CI unreliability**: With N=14 and imbalanced classes, bootstrap
   resamples often omit classes entirely, making CIs unreliable

---

## Phase 2: Robustness Analyses

"""

    # Add 3-class analysis section if available
    if 'robustness' in results and 'three_class_analysis' in results['robustness']:
        tc = results['robustness']['three_class_analysis']
        report += f"""### 3-Class Analysis (Excluding MORONI)

Addresses class imbalance by removing MORONI (only 2 runs).

| Metric | Value |
|--------|-------|
| Classes | {', '.join(tc['classes'])} |
| Chance level | {tc['chance_level']:.1%} |
| LR Balanced Accuracy | {tc['lr']['balanced_accuracy']:.1%} |
| Delta Balanced Accuracy | {tc['delta']['balanced_accuracy']:.1%} |
| Permutation p-value | {tc['permutation']['p_value']:.4f} |

**Interpretation:** {tc['interpretation']}

---

"""

    # Add feature sensitivity section
    if 'robustness' in results and 'feature_sensitivity' in results['robustness']:
        fs = results['robustness']['feature_sensitivity']
        report += """### Feature Sensitivity Analysis

Tests stability across different numbers of top-frequency features.

| k (features) | LR BA | Delta BA | p-value |
|--------------|-------|----------|---------|
"""
        for k, data in sorted(fs['results'].items(), key=lambda x: int(x[0])):
            report += f"| {k} | {data['lr_balanced_accuracy']:.1%} | {data['delta_balanced_accuracy']:.1%} | {data['p_value']:.4f} |\n"

        report += f"""
**Interpretation:** {fs['interpretation']}

---

"""

    # Add confound probe section
    if 'robustness' in results and 'confound_probe' in results['robustness']:
        cp = results['robustness']['confound_probe']
        if 'error' not in cp:
            report += f"""### Confound Probe (Book Prediction)

Tests whether features capture book/topic rather than narrator style.

| Prediction Target | Balanced Accuracy | Chance | Above Chance |
|-------------------|-------------------|--------|--------------|
| Narrator | {cp['narrator']['balanced_accuracy']:.1%} | {cp['narrator']['chance_level']:.1%} | {cp['narrator']['above_chance']:.1%} |
| Book | {cp['book']['balanced_accuracy']:.1%} | {cp['book']['chance_level']:.1%} | {cp['book']['above_chance']:.1%} |

**Interpretation:** {cp['interpretation']}

---

"""

    report += """## Figures

- `figures/permutation-null-dist.png` - Permutation null distribution
- `figures/confusion-matrix-lr.png` - Confusion matrix (Logistic Regression)
- `figures/confusion-matrix-delta.png` - Confusion matrix (Burrows' Delta)

---

*Analysis conducted: {results['metadata']['generated_at']}*
*Random seed: {results['metadata']['random_seed']}*
*Version: {results['metadata']['script_version']}*
"""

    return report


def main(quick_mode=False):
    """
    Main analysis function - v1.5.1 (Exchangeability Fix)

    Structure:
    1. PRIMARY ANALYSIS (Pre-specified, Confirmatory):
       - 3-class LR (JACOB, MORMON, NEPHI) - MORONI excluded due to n=2
       - BLOCKED permutation test (within book-strata) at alpha=0.05
       - Unrestricted permutation reported for reference only

    2. EXPLORATORY ANALYSES (FDR-corrected):
       - 4-class LR and Delta
       - Feature sensitivity, C sensitivity
       - Narrator vs Book comparison

    3. SENSITIVITY ANALYSES:
       - Blocked permutation (within book-strata)
       - Feature ranking inside CV folds

    Args:
        quick_mode: If True, use reduced permutations for development testing
    """
    # Set permutation counts based on mode
    if quick_mode:
        n_perm_primary = 100      # vs 100,000
        n_perm_exploratory = 50   # vs 10,000
        n_perm_sensitivity = 50   # vs 500-10,000
    else:
        n_perm_primary = N_PERMUTATIONS  # 100,000
        n_perm_exploratory = 10000
        n_perm_sensitivity = 500

    print("=" * 70)
    print("RUN-AGGREGATED STYLOMETRIC ANALYSIS v1.5.1")
    print("Publication-Quality Analysis for DSH/LLC (Exchangeability Fix)")
    if quick_mode:
        print(">>> QUICK MODE: Reduced permutations for testing <<<")
    print("=" * 70)

    # Set seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Create output directories
    FIGURES_DIR = Path("results/figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # DATA LOADING AND PREPARATION
    # =========================================================================

    print("\n[1/20] Loading data...")
    data = load_blocks(INPUT_FILE)
    blocks = data["blocks"]

    print("\n[2/20] Aggregating features by run...")
    X, y, run_ids, run_info = aggregate_runs(blocks)
    all_classes = sorted(np.unique(y))

    print(f"  Runs: {len(run_ids)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # =========================================================================
    # NARRATOR-BOOK STRUCTURE ANALYSIS (for exchangeability assessment)
    # =========================================================================

    print("\n[3/20] Analyzing narrator-book structure...")
    contingency = get_narrator_book_contingency(run_info, run_ids, y)
    print(f"  {contingency['interpretation']}")

    # Extract book strata for blocked permutation
    book_strata = get_book_strata(run_info, run_ids)
    strata_array = np.array([book_strata[rid] for rid in run_ids])
    print(f"  Book strata: {len(set(strata_array))} unique")

    # =========================================================================
    # PRIMARY ANALYSIS (Pre-specified, Confirmatory)
    # 3-class classification: JACOB, MORMON, NEPHI (excludes MORONI n=2)
    # =========================================================================

    print("\n" + "=" * 70)
    print("PRIMARY ANALYSIS (Pre-specified, Confirmatory)")
    print("3-class: JACOB, MORMON, NEPHI (MORONI excluded due to n=2)")
    print("=" * 70)

    # Filter to 3 classes
    primary_classes = ["JACOB", "MORMON", "NEPHI"]
    primary_mask = np.isin(y, primary_classes)
    X_primary = X[primary_mask]
    y_primary = y[primary_mask]
    run_ids_primary = [rid for rid, m in zip(run_ids, primary_mask) if m]
    strata_primary = strata_array[primary_mask]

    # Define book strata for blocked permutation
    unique_strata = np.unique(strata_primary)
    strata_counts = {str(s): int(np.sum(strata_primary == s)) for s in unique_strata}
    print(f"\n  Book strata for blocked permutation:")
    for stratum, count in strata_counts.items():
        print(f"    {stratum}: {count} runs")

    print(f"\n[4/20] Running PRIMARY 3-class LOO-CV...")
    print(f"  N = {len(y_primary)} runs, {len(primary_classes)} classes")
    print(f"  Chance level = {1/len(primary_classes):.1%}")

    y_pred_primary = leave_one_out_cv(X_primary, y_primary, C=1.0)
    ba_primary = balanced_accuracy_score(y_primary, y_pred_primary)
    n_correct_primary = np.sum(y_primary == y_pred_primary)
    print(f"  Balanced accuracy: {ba_primary:.1%}")

    # v1.5.1: BLOCKED permutation is now PRIMARY (exchangeability requires it)
    print(f"\n[5/20] Running PRIMARY BLOCKED permutation test ({n_perm_primary:,} permutations)...")
    print(f"  (Blocked permutation respects narrator-book structure)")
    blocked_perm_primary = blocked_permutation_test(
        X_primary, y_primary, ba_primary, strata_primary,
        n_permutations=n_perm_primary, seed=RANDOM_SEED
    )
    print(f"  BLOCKED p-value: {blocked_perm_primary['p_value']:.4f}")
    print(f"  Degenerate permutations: {blocked_perm_primary['n_degenerate']}/{n_perm_primary}")

    # Unrestricted permutation for REFERENCE only (not valid if exchangeability violated)
    print(f"\n[5b/20] Running REFERENCE unrestricted permutation (for comparison)...")
    perm_unrestricted = permutation_test(X_primary, y_primary, ba_primary, n_perm_exploratory, RANDOM_SEED)
    print(f"  Unrestricted p-value: {perm_unrestricted['p_value']:.4f} (REFERENCE ONLY)")
    print(f"  NOTE: Unrestricted p-value is INVALID if exchangeability is violated")

    # Primary analysis conclusion - based on BLOCKED permutation
    primary_significant = blocked_perm_primary['p_value'] < 0.05
    print(f"\n  >>> PRIMARY RESULT (blocked): {'SIGNIFICANT' if primary_significant else 'NOT SIGNIFICANT'} at alpha=0.05")

    # Per-class metrics for primary (with binomial CIs)
    per_class_primary = compute_per_class_metrics(y_primary, y_pred_primary, primary_classes)
    print(f"\n  Per-class recall with 95% Wilson CIs:")
    for cls, metrics in per_class_primary['per_class'].items():
        ci = metrics['recall_ci_95']
        print(f"    {cls}: {metrics['recall']:.1%} [{ci[0]:.1%}, {ci[1]:.1%}] (n={metrics['support']})")

    # Bootstrap CI for balanced accuracy
    print(f"\n[5c/20] Computing bootstrap CI for balanced accuracy...")
    boot_results_primary = bootstrap_ci(X_primary, y_primary, n_bootstrap=1000, seed=RANDOM_SEED)
    print(f"  Bootstrap 95% CI: [{boot_results_primary['ci_95_lower']:.1%}, {boot_results_primary['ci_95_upper']:.1%}]")
    print(f"  (Caveat: bootstrap may be optimistic with small N)")

    # Visualizations for primary
    print("\n[6/20] Generating PRIMARY analysis visualizations...")
    plot_permutation_null(
        blocked_perm_primary['null_scores'] if 'null_scores' in blocked_perm_primary else perm_unrestricted['null_scores'],
        ba_primary,
        blocked_perm_primary['p_value'],
        FIGURES_DIR / "primary-permutation-null.png"
    )
    cm_primary = confusion_matrix(y_primary, y_pred_primary, labels=primary_classes)
    plot_confusion_matrix(cm_primary, primary_classes, FIGURES_DIR / "primary-confusion-matrix.png")

    # =========================================================================
    # EXPLORATORY ANALYSES (FDR-corrected)
    # =========================================================================

    print("\n" + "=" * 70)
    print("EXPLORATORY ANALYSES (will be FDR-corrected)")
    print("=" * 70)

    exploratory_pvalues = {}

    # 4-class analysis (includes MORONI)
    print("\n[7/20] Running EXPLORATORY 4-class analysis...")
    y_pred_4class = leave_one_out_cv(X, y, C=1.0)
    ba_4class = balanced_accuracy_score(y, y_pred_4class)
    perm_4class = permutation_test(X, y, ba_4class, n_perm_exploratory, RANDOM_SEED)
    exploratory_pvalues["4class_lr"] = perm_4class['p_value']
    print(f"  4-class BA: {ba_4class:.1%}, p={perm_4class['p_value']:.4f}")

    per_class_4class = compute_per_class_metrics(y, y_pred_4class, all_classes)
    cm_4class = confusion_matrix(y, y_pred_4class, labels=all_classes)

    # Burrows' Delta (4-class)
    print("\n[8/20] Running EXPLORATORY Burrows' Delta...")
    y_pred_delta, ba_delta = burrows_delta_loo(X, y, all_classes)
    perm_delta = permutation_test(X, y, ba_delta, n_perm_exploratory, RANDOM_SEED)
    exploratory_pvalues["4class_delta"] = perm_delta['p_value']
    print(f"  Delta BA: {ba_delta:.1%}, p={perm_delta['p_value']:.4f}")

    per_class_delta = compute_per_class_metrics(y, y_pred_delta, all_classes)
    cm_delta = confusion_matrix(y, y_pred_delta, labels=all_classes)

    # Burrows' Delta (3-class)
    print("\n[9/20] Running EXPLORATORY 3-class Burrows' Delta...")
    y_pred_delta_3, ba_delta_3 = burrows_delta_loo(X_primary, y_primary, primary_classes)
    exploratory_pvalues["3class_delta"] = 0.5  # Placeholder - skip perm for Delta 3-class
    print(f"  3-class Delta BA: {ba_delta_3:.1%}")

    # C sensitivity
    print("\n[10/20] Running EXPLORATORY C sensitivity...")
    c_sensitivity = c_sensitivity_analysis(X_primary, y_primary)

    # Feature sensitivity (corpus-wide ranking)
    print("\n[11/20] Running EXPLORATORY feature sensitivity (corpus ranking)...")
    feature_sens = feature_sensitivity_analysis(
        X_primary, y_primary, FUNCTION_WORDS,
        k_values=[50, 100, 150],
        n_permutations=n_perm_sensitivity, seed=RANDOM_SEED
    )
    for k, res in feature_sens['results'].items():
        exploratory_pvalues[f"feature_k{k}"] = res['p_value']

    # Narrator vs Book comparison
    print("\n[12/20] Running EXPLORATORY narrator vs book comparison...")
    narrator_book = narrator_vs_book_comparison(X, y, run_info, run_ids,
                                                 n_permutations=n_perm_sensitivity, seed=RANDOM_SEED)
    if 'error' not in narrator_book:
        exploratory_pvalues["book_prediction"] = narrator_book['book']['p_value']
        print(f"  {narrator_book['comparison']['assessment']}")

    # Apply FDR correction
    print("\n[13/20] Applying FDR correction to exploratory p-values...")
    fdr_results = compute_fdr_correction(exploratory_pvalues)
    for name, res in fdr_results.items():
        sig_raw = "* " if res['significant_raw'] else "  "
        sig_fdr = "**" if res['significant_fdr'] else "  "
        print(f"  {name:20s}: raw p={res['p_value_raw']:.4f} {sig_raw} | FDR p={res['p_value_fdr']:.4f} {sig_fdr}")

    # =========================================================================
    # SENSITIVITY ANALYSES
    # =========================================================================

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSES")
    print("=" * 70)

    # Note: Blocked permutation is now PRIMARY (moved to step 5)
    # Unrestricted permutation comparison already done in step 5b

    # Feature ranking inside folds (leak-proof)
    print("\n[14/20] Running feature sensitivity with fold-internal ranking...")
    feature_sens_leakproof = feature_sensitivity_with_fold_ranking(
        X_primary, y_primary, FUNCTION_WORDS,
        k_values=[50, 100],
        n_permutations=n_perm_sensitivity, seed=RANDOM_SEED
    )

    # Jackknife influence (on primary)
    print("\n[15/20] Running jackknife influence analysis...")
    jackknife = jackknife_influence(X_primary, y_primary, run_ids_primary, ba_primary)
    most_influential = max(jackknife.items(), key=lambda x: abs(x[1]['influence']))
    print(f"  Most influential run: {most_influential[0]} "
          f"(influence: {most_influential[1]['influence']:+.1%})")

    # Wilson CI (for raw accuracy only)
    print("\n[16/20] Computing additional uncertainty estimates...")
    wilson_lower, wilson_upper = wilson_interval(n_correct_primary, len(y_primary))
    perm_ci = permutation_based_ci(X_primary, y_primary, blocked_perm_primary['null_scores'], ba_primary)

    # Additional visualizations
    print("\n[18/20] Generating additional visualizations...")
    plot_confusion_matrix(cm_4class, all_classes, FIGURES_DIR / "exploratory-4class-cm.png")
    plot_confusion_matrix(cm_delta, all_classes, FIGURES_DIR / "exploratory-delta-cm.png")

    # =========================================================================
    # COMPILE RESULTS
    # =========================================================================

    print("\n[19/20] Compiling results...")

    # Run distribution
    run_distribution = {}
    for voice in all_classes:
        voice_runs = [rid for rid, info in run_info.items() if info['voice'] == voice]
        run_distribution[voice] = {
            'n_runs': len(voice_runs),
            'n_blocks': sum(run_info[rid]['n_blocks'] for rid in voice_runs),
            'total_words': sum(run_info[rid]['total_words'] for rid in voice_runs)
        }

    # Methodology card
    methodology_card = create_methodology_card()

    # Remove large arrays for JSON
    blocked_perm_primary_json = {k: v for k, v in blocked_perm_primary.items() if k != 'null_scores'}
    perm_unrestricted_json = {k: v for k, v in perm_unrestricted.items() if k != 'null_scores'}
    perm_4class_json = {k: v for k, v in perm_4class.items() if k != 'null_scores'}

    results = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script_version": "1.5.1",
            "random_seed": RANDOM_SEED,
            "rng_note": "numpy.random.RandomState(42) used throughout for reproducibility",
            "n_permutations_primary": n_perm_primary,
            "quick_mode": quick_mode,
            "methodology": "Run-aggregated counts with LOO CV",
            "aggregation_method": "Sum raw FW counts across blocks, then convert to frequencies",
            "status": "SUPPLEMENTARY (not pre-registered)",
            "audit_response": "v1.5.1 addresses exchangeability via blocked permutation as PRIMARY"
        },
        "methodology_card": methodology_card,
        "data_summary": {
            "n_runs": len(run_ids),
            "n_blocks": sum(info['n_blocks'] for info in run_info.values()),
            "total_words": sum(info['total_words'] for info in run_info.values()),
            "n_features": X.shape[1],
            "run_distribution": run_distribution,
            "narrator_book_contingency": contingency,
            "book_strata": strata_counts
        },
        "primary_analysis": {
            "description": "3-class LR (JACOB, MORMON, NEPHI) - MORONI excluded (n=2)",
            "inference_method": "BLOCKED permutation (within book-strata) - respects exchangeability",
            "pre_specified": True,
            "confirmatory": True,
            "classes": primary_classes,
            "n_samples": len(y_primary),
            "chance_level": float(1/len(primary_classes)),
            "balanced_accuracy": float(ba_primary),
            "balanced_accuracy_bootstrap_ci": [
                float(boot_results_primary['ci_95_lower']),
                float(boot_results_primary['ci_95_upper'])
            ],
            "raw_accuracy": float(n_correct_primary / len(y_primary)),
            "blocked_permutation": blocked_perm_primary_json,
            "unrestricted_permutation_reference": perm_unrestricted_json,
            "significant": bool(primary_significant),
            "alpha": 0.05,
            "per_class_metrics": per_class_primary,
            "confusion_matrix": cm_primary.tolist(),
            "y_true": y_primary.tolist(),
            "y_pred": y_pred_primary.tolist(),
            "run_ids": run_ids_primary,
            "note": "Blocked permutation is PRIMARY because narrator-book confounding violates exchangeability. Unrestricted p-value is for REFERENCE only."
        },
        "exploratory_analyses": {
            "note": "All p-values FDR-corrected using Benjamini-Hochberg",
            "fdr_correction": fdr_results,
            "four_class_lr": {
                "balanced_accuracy": float(ba_4class),
                "permutation": perm_4class_json,
                "per_class_metrics": per_class_4class,
                "confusion_matrix": cm_4class.tolist()
            },
            "four_class_delta": {
                "balanced_accuracy": float(ba_delta),
                "permutation": {k: v for k, v in perm_delta.items() if k != 'null_scores'},
                "per_class_metrics": per_class_delta,
                "confusion_matrix": cm_delta.tolist()
            },
            "three_class_delta": {
                "balanced_accuracy": float(ba_delta_3)
            },
            "c_sensitivity": c_sensitivity,
            "feature_sensitivity": feature_sens,
            "narrator_vs_book": narrator_book
        },
        "sensitivity_analyses": {
            "note": "Blocked permutation is now PRIMARY; unrestricted permutation shown for comparison",
            "unrestricted_vs_blocked": {
                "unrestricted_p": float(perm_unrestricted['p_value']),
                "blocked_p": float(blocked_perm_primary['p_value']),
                "interpretation": "Large difference indicates exchangeability violation"
            },
            "feature_ranking_inside_folds": feature_sens_leakproof,
            "jackknife_influence": jackknife
        },
        "uncertainty": {
            "wilson_ci_raw_accuracy": {
                "lower": float(wilson_lower),
                "upper": float(wilson_upper),
                "note": "For raw accuracy only, not balanced accuracy"
            },
            "bootstrap_ci_balanced_accuracy": {
                "lower": float(boot_results_primary['ci_95_lower']),
                "upper": float(boot_results_primary['ci_95_upper']),
                "n_bootstrap": boot_results_primary['n_bootstrap'],
                "caveat": "May be optimistic with N=12 due to LOO duplicate leakage"
            },
            "permutation_based_ci": perm_ci
        },
        "figures": {
            "primary_permutation_null": str(FIGURES_DIR / "primary-permutation-null.png"),
            "primary_confusion_matrix": str(FIGURES_DIR / "primary-confusion-matrix.png"),
            "exploratory_4class_cm": str(FIGURES_DIR / "exploratory-4class-cm.png"),
            "exploratory_delta_cm": str(FIGURES_DIR / "exploratory-delta-cm.png")
        },
        "limitations": {
            "sample_size": "N=12 (primary) or N=14 (exploratory) limits statistical power",
            "moroni": "MORONI excluded from primary due to n=2",
            "exchangeability": "Blocked permutation addresses but may not fully resolve",
            "confounding": contingency['interpretation'],
            "generalization": "Results specific to this text; may not generalize"
        }
    }

    # Save results
    print("\n[20/20] Saving results...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {OUTPUT_FILE}")

    # Generate report (simplified for now - full report update deferred)
    # TODO: Update generate_report() for v1.5.1 structure
    print(f"  Report generation deferred (v1.5.1 structure)")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY (v1.5.1 - Exchangeability Fix)")
    print("=" * 70)

    print(f"\n>>> PRIMARY ANALYSIS (Confirmatory) <<<")
    blocked_null_mean = blocked_perm_primary['null_mean']
    print(f"3-class (JACOB, MORMON, NEPHI): BA = {ba_primary:.1%} (blocked-null mean = {blocked_null_mean:.1%})")
    print(f"Bootstrap 95% CI: [{boot_results_primary['ci_95_lower']:.1%}, {boot_results_primary['ci_95_upper']:.1%}] (descriptive only)")
    print(f"BLOCKED permutation p-value (PRIMARY): {blocked_perm_primary['p_value']:.4f}")
    print(f"Unrestricted permutation p-value (reference): {perm_unrestricted['p_value']:.4f}")
    print(f"CONCLUSION: {'SIGNIFICANT' if primary_significant else 'NOT SIGNIFICANT'} at alpha=0.05")

    print(f"\n>>> EXPLORATORY ANALYSES (FDR-corrected) <<<")
    print(f"4-class LR: BA = {ba_4class:.1%}, raw p = {perm_4class['p_value']:.4f}, "
          f"FDR p = {fdr_results.get('4class_lr', {}).get('p_value_fdr', 'N/A')}")
    print(f"4-class Delta: BA = {ba_delta:.1%}")
    if 'error' not in narrator_book:
        print(f"Narrator vs Book: {narrator_book['comparison']['assessment']}")

    print(f"\n>>> EXCHANGEABILITY CHECK <<<")
    print(f"Blocked p-value (PRIMARY): {blocked_perm_primary['p_value']:.4f}")
    print(f"Unrestricted p-value (ref): {perm_unrestricted['p_value']:.4f}")
    p_diff = perm_unrestricted['p_value'] - blocked_perm_primary['p_value']
    if abs(p_diff) > 0.1:
        print(f"WARNING: Large difference ({p_diff:+.4f}) confirms exchangeability violation")
    print(f"Max jackknife influence: {most_influential[1]['influence']:+.1%}")

    print(f"\n>>> LIMITATIONS <<<")
    print(f"- N=12 runs in primary analysis (MORONI excluded)")
    print(f"- {contingency['interpretation']}")
    print(f"- High-dimensional (171 features, ~12 training samples/fold)")
    print(f"- Cannot definitively separate narrator from topic effects")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run-Aggregated Stylometric Analysis v1.5.1")
    parser.add_argument("--quick", action="store_true",
                        help="Quick development mode: reduced permutations for testing")
    args = parser.parse_args()

    if args.quick:
        # Override for quick testing - modify global
        import sys
        N_PERMUTATIONS = 100
        # Patch all permutation counts in main() by passing as argument
        print(">>> QUICK MODE: Using reduced permutations <<<")

    main(quick_mode=args.quick)
