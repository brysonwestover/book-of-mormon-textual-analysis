#!/usr/bin/env python3
"""
Corrected stylometric classification experiments (v2).

Implements GPT-5.2 Pro methodology corrections:
1. Function words only (content-suppressed) as primary analysis
2. Class weights to handle imbalance
3. Macro-F1 and balanced accuracy as primary metrics
4. Correct baselines (majority class, not random)
5. Permutation test for significance
6. Downsampling experiments for apples-to-apples comparison
7. Proper pipeline to prevent leakage

See: docs/decisions/classification-methodology-corrections.md

Version: 2.0.0
Date: 2026-02-01
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import re

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GroupKFold, cross_val_predict, permutation_test_score
)
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report, precision_recall_fscore_support
)

# Configuration
INPUT_FILE = Path("data/text/processed/bom-voice-blocks.json")
OUTPUT_FILE = Path("results/classification-results-v2.json")
REPORT_FILE = Path("results/classification-report-v2.md")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Function words (content-light, style-heavy)
# Excluding theological/content terms
FUNCTION_WORDS = [
    # Articles
    "a", "an", "the",
    # Pronouns
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "what", "that", "this", "these", "those",
    "one", "ones",
    # Prepositions
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "off", "over", "under",
    "of", "unto", "upon",
    # Conjunctions
    "and", "but", "or", "nor", "yet", "so",
    "if", "then", "because", "although", "while", "whereas", "when", "where",
    # Auxiliary verbs
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    # Negation
    "not", "no", "never", "neither", "nor",
    # Common adverbs (non-content)
    "now", "then", "here", "there", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "any", "such", "only", "also", "very", "just", "even",
    "again", "ever", "still", "already",
    # Archaic function words (BoM-relevant but not content)
    "ye", "thee", "thou", "thy", "thine", "thyself",
    "hath", "doth", "didst", "hast", "art", "wilt", "shalt",
    "wherefore", "therefore", "thus", "hence",
    "yea", "nay", "verily",
    "lest", "except", "save",
    "whoso", "whosoever", "whatsoever",
    "forth", "hither", "thither", "whither",
]

# Words to EXCLUDE (content/theological)
CONTENT_WORDS_TO_EXCLUDE = [
    "god", "lord", "christ", "jesus", "messiah", "holy", "spirit",
    "prophet", "prophets", "angel", "angels",
    "nephi", "mormon", "moroni", "jacob", "alma", "lehi",
    "nephite", "nephites", "lamanite", "lamanites",
    "israel", "jerusalem", "zion",
    "commandment", "commandments", "covenant", "covenants",
    "sin", "sins", "repent", "repentance",
    "faith", "believe", "prayer", "pray",
    "church", "priest", "priests", "king", "kings",
    "war", "battle", "sword", "army",
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


def extract_function_word_features(text: str) -> dict:
    """Extract function word frequencies only (content-suppressed)."""
    tokens = tokenize(text)
    total_words = len(tokens)

    if total_words == 0:
        return {f"fw_{w}": 0.0 for w in FUNCTION_WORDS}

    # Count only function words, exclude content words
    word_counts = Counter(tokens)

    features = {}
    for word in FUNCTION_WORDS:
        if word not in CONTENT_WORDS_TO_EXCLUDE:
            count = word_counts.get(word, 0)
            features[f"fw_{word}"] = (count / total_words) * 1000  # per 1000 words

    return features


def build_feature_matrix(blocks: list, target_size: int = 1000,
                         voices: list = None) -> tuple:
    """
    Build feature matrix using function words only.

    Returns:
        X: numpy array (n_samples, n_features)
        y: numpy array of labels
        groups: numpy array of run_ids
        feature_names: list
        metadata: list of block metadata dicts
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

    # Extract features for first block to get feature names
    sample_features = extract_function_word_features(filtered_blocks[0]["text"])
    feature_names = sorted(sample_features.keys())
    n_features = len(feature_names)

    # Build matrices
    n_samples = len(filtered_blocks)
    X = np.zeros((n_samples, n_features))
    y = []
    groups = []
    metadata = []

    feat_to_idx = {name: i for i, name in enumerate(feature_names)}

    for i, block in enumerate(filtered_blocks):
        features = extract_function_word_features(block["text"])

        for feat_name, value in features.items():
            if feat_name in feat_to_idx:
                X[i, feat_to_idx[feat_name]] = value

        y.append(block["voice"])
        groups.append(block["run_id"])
        metadata.append({
            "block_id": block["block_id"],
            "run_id": block["run_id"],
            "voice": block["voice"],
            "book": block["book"],
            "start_ref": block["start_ref"],
            "end_ref": block["end_ref"],
            "word_count": block["word_count"]
        })

    return X, np.array(y), np.array(groups), feature_names, metadata


def compute_baselines(y: np.ndarray) -> dict:
    """Compute proper baselines for imbalanced data."""
    class_counts = Counter(y)
    n_total = len(y)
    n_classes = len(class_counts)

    # Majority class baseline
    majority_class = max(class_counts, key=class_counts.get)
    majority_count = class_counts[majority_class]
    majority_accuracy = majority_count / n_total

    # Random baseline (uniform)
    random_accuracy = 1 / n_classes

    # Stratified random baseline (predict according to class distribution)
    stratified_accuracy = sum((count / n_total) ** 2 for count in class_counts.values())

    # Macro-F1 for trivial predictors
    # Always predict majority: only one class has recall=1, others=0
    # F1 for majority = 2 * precision * recall / (precision + recall)
    # precision = majority_count / n_total, recall = 1
    majority_precision = majority_count / n_total
    majority_f1_for_class = 2 * majority_precision * 1 / (majority_precision + 1)
    # Macro-F1 = average across classes = majority_f1 / n_classes (others are 0)
    trivial_macro_f1 = majority_f1_for_class / n_classes

    return {
        "majority_class": majority_class,
        "majority_accuracy": majority_accuracy,
        "random_accuracy": random_accuracy,
        "stratified_accuracy": stratified_accuracy,
        "trivial_macro_f1": trivial_macro_f1,
        "class_distribution": dict(class_counts)
    }


def run_classification_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          n_splits: int = 5) -> dict:
    """
    Run classification with proper methodology.

    Uses:
    - Pipeline to prevent leakage
    - Class weights for imbalance
    - GroupKFold for proper CV
    - Multiple metrics
    """
    # Adjust splits for number of groups
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)

    # Create pipeline (scaler + classifier)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver='lbfgs'
        ))
    ])

    # GroupKFold
    gkf = GroupKFold(n_splits=actual_splits)

    # Get predictions
    y_pred = cross_val_predict(pipeline, X, y, groups=groups, cv=gkf)

    # Compute metrics
    classes = sorted(set(y))

    accuracy = accuracy_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')
    weighted_f1 = f1_score(y, y_pred, average='weighted')

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=classes, zero_division=0
    )

    per_class = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": int(support[i])
        }

    conf_matrix = confusion_matrix(y, y_pred, labels=classes)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": conf_matrix.tolist(),
        "classes": classes,
        "y_true": y.tolist(),
        "y_pred": y_pred.tolist(),
        "n_splits": actual_splits,
        "n_groups": n_groups
    }


def run_permutation_test(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                         n_permutations: int = 100) -> dict:
    """
    Run permutation test for statistical significance.

    Permutes labels at the group level.
    """
    print(f"  Running permutation test ({n_permutations} permutations)...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver='lbfgs'
        ))
    ])

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    # Use balanced accuracy as the scoring metric
    score, perm_scores, p_value = permutation_test_score(
        pipeline, X, y, groups=groups, cv=gkf,
        n_permutations=n_permutations,
        scoring='balanced_accuracy',
        random_state=RANDOM_SEED
    )

    return {
        "observed_score": score,
        "p_value": p_value,
        "null_mean": np.mean(perm_scores),
        "null_std": np.std(perm_scores),
        "null_95_percentile": np.percentile(perm_scores, 95),
        "n_permutations": n_permutations
    }


def run_downsampling_experiment(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                                 target_per_class: int = 12,
                                 n_resamples: int = 100) -> dict:
    """
    Run downsampled experiments for apples-to-apples comparison.

    Repeatedly sample each class to target_per_class blocks.
    """
    print(f"  Running downsampling experiment ({n_resamples} resamples, n={target_per_class}/class)...")

    classes = sorted(set(y))
    class_indices = {cls: np.where(y == cls)[0] for cls in classes}

    # Check if all classes have enough samples
    min_class_size = min(len(idx) for idx in class_indices.values())
    actual_target = min(target_per_class, min_class_size)

    if actual_target < target_per_class:
        print(f"    Warning: Reducing target to {actual_target} (smallest class size)")

    macro_f1_scores = []
    balanced_acc_scores = []

    for i in range(n_resamples):
        # Sample indices
        sampled_indices = []
        for cls in classes:
            cls_idx = class_indices[cls]
            np.random.seed(RANDOM_SEED + i)
            sampled = np.random.choice(cls_idx, size=actual_target, replace=False)
            sampled_indices.extend(sampled)

        sampled_indices = np.array(sampled_indices)

        X_sample = X[sampled_indices]
        y_sample = y[sampled_indices]
        groups_sample = groups[sampled_indices]

        # Run CV on sampled data
        try:
            results = run_classification_cv(X_sample, y_sample, groups_sample, n_splits=3)
            macro_f1_scores.append(results["macro_f1"])
            balanced_acc_scores.append(results["balanced_accuracy"])
        except Exception as e:
            continue

    return {
        "target_per_class": actual_target,
        "n_resamples": len(macro_f1_scores),
        "macro_f1_mean": np.mean(macro_f1_scores),
        "macro_f1_std": np.std(macro_f1_scores),
        "macro_f1_95ci": [np.percentile(macro_f1_scores, 2.5),
                          np.percentile(macro_f1_scores, 97.5)],
        "balanced_acc_mean": np.mean(balanced_acc_scores),
        "balanced_acc_std": np.std(balanced_acc_scores),
        "balanced_acc_95ci": [np.percentile(balanced_acc_scores, 2.5),
                              np.percentile(balanced_acc_scores, 97.5)]
    }


def run_feature_importance(X: np.ndarray, y: np.ndarray,
                            feature_names: list) -> list:
    """Compute feature importance using logistic regression coefficients."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver='lbfgs'
        ))
    ])

    pipeline.fit(X, y)

    # Get coefficients (shape: n_classes x n_features)
    coefs = pipeline.named_steps['clf'].coef_

    # Average absolute coefficient across classes
    importance = np.mean(np.abs(coefs), axis=0)

    # Sort by importance
    indices = np.argsort(importance)[::-1]

    return [(feature_names[i], importance[i]) for i in indices[:30]]


def generate_report_v2(results: dict, output_path: Path):
    """Generate corrected methodology report."""
    lines = [
        "# Stylometric Classification Results (v2 - Corrected Methodology)",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "---",
        "",
        "## Methodology Corrections Applied",
        "",
        "This analysis addresses issues identified in GPT-5.2 Pro review:",
        "",
        "1. **Content-suppressed features:** Function words only (no theological terms)",
        "2. **Proper baselines:** Majority class (not random)",
        "3. **Imbalance handling:** Class weights + balanced accuracy",
        "4. **Statistical significance:** Permutation test",
        "5. **Apples-to-apples:** Downsampled experiments",
        "",
        "---",
        "",
        "## Data Summary",
        "",
        f"- **Total blocks:** {results['n_samples']}",
        f"- **Features:** {results['n_features']} (function words only)",
        f"- **Classes:** {', '.join(results['classes'])}",
        "",
        "### Class Distribution",
        "",
        "| Voice | Blocks | % |",
        "|-------|--------|---|",
    ]

    for voice, count in sorted(results["baselines"]["class_distribution"].items(),
                                key=lambda x: -x[1]):
        pct = count / results["n_samples"] * 100
        lines.append(f"| {voice} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## Baselines",
        "",
        "| Baseline | Value |",
        "|----------|-------|",
        f"| Majority class ({results['baselines']['majority_class']}) | {results['baselines']['majority_accuracy']:.1%} |",
        f"| Random (uniform) | {results['baselines']['random_accuracy']:.1%} |",
        f"| Trivial macro-F1 | {results['baselines']['trivial_macro_f1']:.3f} |",
        "",
        "---",
        "",
        "## Primary Results (Function Words Only)",
        "",
        "| Metric | Value | vs Baseline |",
        "|--------|-------|-------------|",
        f"| Accuracy | {results['primary']['accuracy']:.1%} | vs {results['baselines']['majority_accuracy']:.1%} majority |",
        f"| **Balanced Accuracy** | **{results['primary']['balanced_accuracy']:.1%}** | vs 25% random |",
        f"| **Macro F1** | **{results['primary']['macro_f1']:.3f}** | vs {results['baselines']['trivial_macro_f1']:.3f} trivial |",
        "",
        "### Per-Class Performance",
        "",
        "| Voice | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------| ",
    ])

    for cls in results['classes']:
        pc = results['primary']['per_class'][cls]
        lines.append(f"| {cls} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} | {pc['support']} |")

    lines.extend([
        "",
        "### Confusion Matrix",
        "",
        "```",
        "Pred →   " + "  ".join(f"{c[:6]:>6}" for c in results['classes']),
        "True ↓",
    ])

    conf_matrix = np.array(results['primary']['confusion_matrix'])
    for i, cls in enumerate(results['classes']):
        row = f"{cls[:6]:>6}   " + "  ".join(f"{conf_matrix[i,j]:>6}" for j in range(len(results['classes'])))
        lines.append(row)

    lines.extend([
        "```",
        "",
        "---",
        "",
        "## Statistical Significance (Permutation Test)",
        "",
        f"- **Observed balanced accuracy:** {results['permutation']['observed_score']:.3f}",
        f"- **Null distribution mean:** {results['permutation']['null_mean']:.3f} ± {results['permutation']['null_std']:.3f}",
        f"- **p-value:** {results['permutation']['p_value']:.4f}",
        "",
    ])

    if results['permutation']['p_value'] < 0.05:
        lines.append("**Interpretation:** The classification performance is statistically significant (p < 0.05).")
    else:
        lines.append("**Interpretation:** The classification performance is NOT statistically significant (p ≥ 0.05).")

    lines.extend([
        "",
        "---",
        "",
        "## Downsampled Experiment (Apples-to-Apples)",
        "",
        f"To address class imbalance, we repeatedly sampled {results['downsampling']['target_per_class']} blocks per class.",
        "",
        f"- **Resamples:** {results['downsampling']['n_resamples']}",
        f"- **Macro F1:** {results['downsampling']['macro_f1_mean']:.3f} (95% CI: [{results['downsampling']['macro_f1_95ci'][0]:.3f}, {results['downsampling']['macro_f1_95ci'][1]:.3f}])",
        f"- **Balanced Accuracy:** {results['downsampling']['balanced_acc_mean']:.1%} (95% CI: [{results['downsampling']['balanced_acc_95ci'][0]:.1%}, {results['downsampling']['balanced_acc_95ci'][1]:.1%}])",
        "",
        "---",
        "",
        "## Top Discriminating Features (Function Words)",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ])

    for i, (feat, imp) in enumerate(results["top_features"][:15], 1):
        feat_display = feat.replace("fw_", "")
        lines.append(f"| {i} | {feat_display} | {imp:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ])

    # Interpretation based on results
    ba = results['primary']['balanced_accuracy']
    p_val = results['permutation']['p_value']
    ds_f1 = results['downsampling']['macro_f1_mean']

    if p_val < 0.05 and ba > 0.35:
        lines.extend([
            "**Finding:** There is statistically significant evidence of stylistic differentiation between claimed narrators,",
            "even when using only function words (content-suppressed analysis).",
            "",
        ])
    elif p_val < 0.05:
        lines.extend([
            "**Finding:** While statistically significant, the effect size is modest.",
            "The stylistic differences are subtle and may not strongly support distinct authorship claims.",
            "",
        ])
    else:
        lines.extend([
            "**Finding:** The classification performance is not statistically significant.",
            "There is insufficient evidence of stylistic differentiation using function words alone.",
            "",
        ])

    lines.extend([
        "**Limitations:**",
        "",
        "- Jacob (n=12) treated as exploratory due to small sample",
        "- Cannot distinguish genuine multi-authorship from skilled mimicry",
        "- Translation layer effects cannot be separated",
        "- Topic/genre confounds partially controlled but not eliminated",
        "",
        "---",
        "",
        "*Methodology per GPT-5.2 Pro consultation. See docs/decisions/classification-methodology-corrections.md*",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    print("=" * 70)
    print("STYLOMETRIC CLASSIFICATION v2 (CORRECTED METHODOLOGY)")
    print("=" * 70)
    print()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    print(f"Loading blocks from {INPUT_FILE}...")
    data = load_blocks(INPUT_FILE)
    blocks = data["blocks"]
    print(f"  Loaded {len(blocks)} total blocks")

    # Build feature matrix (function words only)
    print()
    print("Building feature matrix (function words only)...")
    X, y, groups, feature_names, metadata = build_feature_matrix(blocks)
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]} (function words)")
    print(f"  Groups: {len(np.unique(groups))}")

    # Compute baselines
    print()
    print("Computing baselines...")
    baselines = compute_baselines(y)
    print(f"  Majority class: {baselines['majority_class']} ({baselines['majority_accuracy']:.1%})")
    print(f"  Random baseline: {baselines['random_accuracy']:.1%}")

    # Class distribution
    print()
    print("Class distribution:")
    for voice, count in sorted(baselines['class_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {voice}: {count} ({count/len(y)*100:.1f}%)")

    # Primary classification
    print()
    print("Running primary classification (with class weights)...")
    primary_results = run_classification_cv(X, y, groups)
    print(f"  Accuracy: {primary_results['accuracy']:.1%}")
    print(f"  Balanced Accuracy: {primary_results['balanced_accuracy']:.1%}")
    print(f"  Macro F1: {primary_results['macro_f1']:.3f}")

    # Permutation test
    print()
    perm_results = run_permutation_test(X, y, groups, n_permutations=100)
    print(f"  p-value: {perm_results['p_value']:.4f}")

    # Downsampling experiment
    print()
    downsample_results = run_downsampling_experiment(X, y, groups, target_per_class=12, n_resamples=100)
    print(f"  Downsampled Macro F1: {downsample_results['macro_f1_mean']:.3f} ± {downsample_results['macro_f1_std']:.3f}")

    # Feature importance
    print()
    print("Computing feature importance...")
    top_features = run_feature_importance(X, y, feature_names)
    print("Top 10 features:")
    for feat, imp in top_features[:10]:
        print(f"  {feat}: {imp:.4f}")

    # Compile results
    results = {
        "metadata": {
            "source_file": str(INPUT_FILE),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script_version": "2.0.0",
            "methodology": "GPT-5.2 Pro corrected (function words, class weights, permutation test)",
            "random_seed": RANDOM_SEED
        },
        "n_samples": len(y),
        "n_features": X.shape[1],
        "classes": sorted(set(y)),
        "baselines": baselines,
        "primary": primary_results,
        "permutation": perm_results,
        "downsampling": downsample_results,
        "top_features": [(f, float(i)) for f, i in top_features]
    }

    # Save
    print()
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Generating report at {REPORT_FILE}...")
    generate_report_v2(results, REPORT_FILE)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Balanced Accuracy: {primary_results['balanced_accuracy']:.1%} (vs 25% chance)")
    print(f"Macro F1: {primary_results['macro_f1']:.3f} (vs {baselines['trivial_macro_f1']:.3f} trivial)")
    print(f"Permutation p-value: {perm_results['p_value']:.4f}")
    print(f"Downsampled Macro F1: {downsample_results['macro_f1_mean']:.3f} [{downsample_results['macro_f1_95ci'][0]:.3f}, {downsample_results['macro_f1_95ci'][1]:.3f}]")
    print()

    if perm_results['p_value'] < 0.05:
        print("CONCLUSION: Statistically significant stylistic differentiation detected.")
    else:
        print("CONCLUSION: No statistically significant stylistic differentiation detected.")

    print()
    print(f"Full report: {REPORT_FILE}")

    return results


if __name__ == "__main__":
    main()
