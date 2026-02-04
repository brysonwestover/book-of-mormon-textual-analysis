#!/usr/bin/env python3
"""
Run stylometric classification experiments on Book of Mormon voice blocks.

Implements:
1. 4-class classification (MORMON, NEPHI, MORONI, JACOB)
2. Grouped cross-validation by run_id (prevents leakage)
3. Multiple classifiers (SVM, Random Forest)
4. Bootstrap confidence intervals
5. Confusion matrix analysis

See: docs/decisions/block-derivation-strategy.md

Version: 1.0.0
Date: 2026-02-01
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report
)

# Configuration
INPUT_FILE = Path("data/text/processed/bom-stylometric-features.json")
OUTPUT_FILE = Path("results/classification-results.json")
REPORT_FILE = Path("results/classification-report.md")

# Ensure output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42


def load_features(filepath: Path) -> dict:
    """Load the stylometric features JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_feature_matrix(data: dict) -> tuple:
    """
    Build feature matrix X, labels y, and groups for CV.

    Returns:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of labels
        groups: numpy array of run_ids for grouped CV
        feature_names: list of feature names
        block_ids: list of block identifiers
    """
    blocks = data["blocks"]
    feature_names = data["feature_names"]

    n_samples = len(blocks)
    n_features = len(feature_names)

    X = np.zeros((n_samples, n_features))
    y = []
    groups = []
    block_ids = []

    # Create feature name to index mapping
    feat_to_idx = {name: i for i, name in enumerate(feature_names)}

    for i, block in enumerate(blocks):
        # Extract features
        features = block["features"]
        for feat_name, value in features.items():
            if feat_name in feat_to_idx:
                X[i, feat_to_idx[feat_name]] = value

        # Label and group
        y.append(block["voice"])
        groups.append(block["run_id"])
        block_ids.append(block["block_id"])

    return X, np.array(y), np.array(groups), feature_names, block_ids


def run_grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   classifier, n_splits: int = 5) -> dict:
    """
    Run grouped cross-validation.

    Groups are run_ids, ensuring no text from the same contiguous passage
    appears in both train and test.
    """
    # Get unique groups
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # Adjust n_splits if we have fewer groups
    actual_splits = min(n_splits, n_groups)

    if actual_splits < 2:
        print(f"  Warning: Only {n_groups} groups, using leave-one-out")
        actual_splits = n_groups

    gkf = GroupKFold(n_splits=actual_splits)

    # Get cross-validated predictions
    y_pred = cross_val_predict(classifier, X, y, groups=groups, cv=gkf)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y, y_pred)

    # Get class labels
    classes = sorted(set(y))

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": conf_matrix.tolist(),
        "classes": classes,
        "y_true": y.tolist(),
        "y_pred": y_pred.tolist(),
        "n_splits": actual_splits,
        "n_groups": n_groups
    }


def bootstrap_confidence_interval(y_true: list, y_pred: list,
                                   metric_fn, n_bootstrap: int = 1000,
                                   confidence: float = 0.95) -> tuple:
    """
    Compute bootstrap confidence interval for a metric.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]

        try:
            score = metric_fn(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except:
            continue

    bootstrap_scores = np.array(bootstrap_scores)

    # Point estimate
    point = metric_fn(y_true, y_pred)

    # Confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

    return point, lower, upper


def analyze_feature_importance(X: np.ndarray, y: np.ndarray,
                                feature_names: list, top_k: int = 30) -> list:
    """
    Analyze feature importance using Random Forest.

    Returns list of (feature_name, importance) tuples.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]

    return [(feature_names[i], importances[i]) for i in indices]


def generate_report(results: dict, output_path: Path):
    """Generate a markdown report of the classification results."""
    lines = [
        "# Stylometric Classification Results",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Input:** {INPUT_FILE}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Total blocks:** {results['n_samples']}",
        f"- **Features:** {results['n_features']}",
        f"- **Classes:** {', '.join(results['classes'])}",
        f"- **Cross-validation:** {results['cv_splits']}-fold GroupKFold by run_id",
        "",
        "---",
        "",
        "## Classification Performance",
        "",
    ]

    for clf_name, clf_results in results["classifiers"].items():
        lines.extend([
            f"### {clf_name}",
            "",
            f"| Metric | Value | 95% CI |",
            f"|--------|-------|--------|",
            f"| Accuracy | {clf_results['accuracy']:.3f} | [{clf_results['accuracy_ci'][0]:.3f}, {clf_results['accuracy_ci'][1]:.3f}] |",
            f"| Macro F1 | {clf_results['f1_macro']:.3f} | [{clf_results['f1_macro_ci'][0]:.3f}, {clf_results['f1_macro_ci'][1]:.3f}] |",
            f"| Weighted F1 | {clf_results['f1_weighted']:.3f} | - |",
            "",
            "**Confusion Matrix:**",
            "",
            "```",
        ])

        # Format confusion matrix
        classes = clf_results["classes"]
        conf_matrix = np.array(clf_results["confusion_matrix"])

        # Header
        header = "Pred →   " + "  ".join(f"{c[:6]:>6}" for c in classes)
        lines.append(header)
        lines.append("True ↓")

        for i, true_class in enumerate(classes):
            row = f"{true_class[:6]:>6}   " + "  ".join(f"{conf_matrix[i,j]:>6}" for j in range(len(classes)))
            lines.append(row)

        lines.extend([
            "```",
            "",
        ])

    # Feature importance
    lines.extend([
        "---",
        "",
        "## Top Discriminating Features",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ])

    for i, (feat, imp) in enumerate(results["top_features"][:20], 1):
        # Clean up feature name for display
        feat_display = feat.replace("mfw_", "").replace("char2_", "2g:").replace("char3_", "3g:").replace("char4_", "4g:")
        lines.append(f"| {i} | {feat_display} | {imp:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## Class Distribution",
        "",
        "| Voice | Blocks | % |",
        "|-------|--------|---|",
    ])

    for voice, count in sorted(results["class_distribution"].items(),
                                key=lambda x: -x[1]):
        pct = count / results["n_samples"] * 100
        lines.append(f"| {voice} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "The classification results indicate whether the claimed narrators (Mormon, Nephi, Moroni, Jacob) ",
        "exhibit statistically distinguishable stylistic signatures.",
        "",
        "**Key observations:**",
        "",
    ])

    # Add interpretation based on results
    best_clf = max(results["classifiers"].items(),
                   key=lambda x: x[1]["accuracy"])
    best_acc = best_clf[1]["accuracy"]
    baseline = 1 / len(results["classes"])

    if best_acc > baseline + 0.3:
        lines.append(f"- Accuracy ({best_acc:.1%}) is substantially above chance ({baseline:.1%}), suggesting detectable stylistic differences")
    elif best_acc > baseline + 0.15:
        lines.append(f"- Accuracy ({best_acc:.1%}) is moderately above chance ({baseline:.1%}), suggesting some stylistic differentiation")
    else:
        lines.append(f"- Accuracy ({best_acc:.1%}) is only marginally above chance ({baseline:.1%}), suggesting limited stylistic differentiation")

    lines.extend([
        "",
        "**Limitations:**",
        "",
        "- Sample sizes are imbalanced (Mormon >> others)",
        "- Enos, Jarom, Omni excluded due to insufficient data",
        "- Results may reflect topic/genre differences, not just author",
        "- Translation layer effects cannot be fully separated",
        "",
        "---",
        "",
        "*See METHODOLOGY.md for full research framework and LIMITATIONS.md for scope boundaries.*",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    print("=" * 70)
    print("STYLOMETRIC CLASSIFICATION EXPERIMENTS")
    print("=" * 70)
    print()

    # Set random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    print(f"Loading features from {INPUT_FILE}...")
    data = load_features(INPUT_FILE)
    print(f"  Loaded {len(data['blocks'])} blocks with {len(data['feature_names'])} features")
    print()

    # Build feature matrix
    print("Building feature matrix...")
    X, y, groups, feature_names, block_ids = build_feature_matrix(data)
    print(f"  X shape: {X.shape}")
    print(f"  Classes: {sorted(set(y))}")
    print(f"  Unique groups (run_ids): {len(set(groups))}")
    print()

    # Class distribution
    class_dist = Counter(y)
    print("Class distribution:")
    for voice, count in sorted(class_dist.items(), key=lambda x: -x[1]):
        print(f"  {voice}: {count} ({count/len(y)*100:.1f}%)")
    print()

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print()

    # Define classifiers
    classifiers = {
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, random_state=RANDOM_SEED),
        "SVM (Linear)": SVC(kernel='linear', C=1.0, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    }

    # Run experiments
    results = {
        "metadata": {
            "source_file": str(INPUT_FILE),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED
        },
        "n_samples": len(y),
        "n_features": X.shape[1],
        "classes": sorted(set(y)),
        "class_distribution": dict(class_dist),
        "cv_splits": 5,
        "classifiers": {}
    }

    for clf_name, clf in classifiers.items():
        print(f"Running {clf_name}...")

        # Grouped CV
        cv_results = run_grouped_cv(X_scaled, y, groups, clf, n_splits=5)

        print(f"  Accuracy: {cv_results['accuracy']:.3f}")
        print(f"  Macro F1: {cv_results['f1_macro']:.3f}")

        # Bootstrap CIs
        print("  Computing bootstrap confidence intervals...")

        acc_point, acc_lower, acc_upper = bootstrap_confidence_interval(
            cv_results["y_true"], cv_results["y_pred"],
            accuracy_score
        )

        f1_point, f1_lower, f1_upper = bootstrap_confidence_interval(
            cv_results["y_true"], cv_results["y_pred"],
            lambda yt, yp: f1_score(yt, yp, average='macro')
        )

        print(f"  Accuracy 95% CI: [{acc_lower:.3f}, {acc_upper:.3f}]")
        print(f"  Macro F1 95% CI: [{f1_lower:.3f}, {f1_upper:.3f}]")
        print()

        # Store results
        results["classifiers"][clf_name] = {
            "accuracy": cv_results["accuracy"],
            "accuracy_ci": [acc_lower, acc_upper],
            "f1_macro": cv_results["f1_macro"],
            "f1_macro_ci": [f1_lower, f1_upper],
            "f1_weighted": cv_results["f1_weighted"],
            "confusion_matrix": cv_results["confusion_matrix"],
            "classes": cv_results["classes"],
            "n_splits": cv_results["n_splits"],
            "n_groups": cv_results["n_groups"]
        }

    # Feature importance analysis
    print("Analyzing feature importance...")
    top_features = analyze_feature_importance(X_scaled, y, feature_names)
    results["top_features"] = [(f, float(i)) for f, i in top_features]

    print("Top 10 discriminating features:")
    for feat, imp in top_features[:10]:
        print(f"  {feat}: {imp:.4f}")
    print()

    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Generate report
    print(f"Generating report at {REPORT_FILE}...")
    generate_report(results, REPORT_FILE)

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    best_clf = max(results["classifiers"].items(),
                   key=lambda x: x[1]["accuracy"])
    baseline = 1 / len(results["classes"])

    print(f"Best classifier: {best_clf[0]}")
    print(f"  Accuracy: {best_clf[1]['accuracy']:.1%} (baseline: {baseline:.1%})")
    print(f"  Macro F1: {best_clf[1]['f1_macro']:.3f}")
    print()
    print(f"Full report: {REPORT_FILE}")
    print(f"Raw results: {OUTPUT_FILE}")

    return results


if __name__ == "__main__":
    main()
