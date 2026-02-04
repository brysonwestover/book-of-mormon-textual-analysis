#!/usr/bin/env python3
"""
Additional stylometric analyses to investigate the null result.

Implements:
1. Unmasking analysis - Does separation collapse quickly or gradually?
2. Pairwise comparisons - Mormon vs Nephi (largest samples)
3. Same-author verification - Are cross-narrator pairs as different as expected?
4. Within-class consistency - How consistent is each narrator with itself?

Version: 1.0.0
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score
)
from scipy.spatial.distance import cosine, cdist

INPUT_FILE = Path("data/text/processed/bom-voice-blocks.json")
OUTPUT_FILE = Path("results/additional-analyses.json")
REPORT_FILE = Path("results/additional-analyses-report.md")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Function words list (same as v2)
FUNCTION_WORDS = [
    "a", "an", "the", "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "what", "that", "this", "these", "those",
    "one", "ones", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "off", "over", "under", "of", "unto", "upon",
    "and", "but", "or", "nor", "yet", "so", "if", "then", "because", "although",
    "while", "whereas", "when", "where", "be", "is", "am", "are", "was", "were",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "not", "no", "never", "neither", "nor", "now", "then", "here", "there", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "any", "such", "only", "also", "very", "just", "even",
    "again", "ever", "still", "already",
    "ye", "thee", "thou", "thy", "thine", "thyself",
    "hath", "doth", "didst", "hast", "art", "wilt", "shalt",
    "wherefore", "therefore", "thus", "hence", "yea", "nay", "verily",
    "lest", "except", "save", "whoso", "whosoever", "whatsoever",
    "forth", "hither", "thither", "whither",
]


def load_blocks(filepath: Path) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def tokenize(text: str) -> list:
    text = text.lower()
    return re.findall(r'\b[a-z]+\b', text)


def extract_function_word_features(text: str) -> np.ndarray:
    """Extract function word frequency vector."""
    tokens = tokenize(text)
    total = len(tokens)
    if total == 0:
        return np.zeros(len(FUNCTION_WORDS))

    word_counts = Counter(tokens)
    features = np.array([
        (word_counts.get(w, 0) / total) * 1000
        for w in FUNCTION_WORDS
    ])
    return features


def build_feature_matrix(blocks: list, target_size: int = 1000,
                         voices: list = None) -> tuple:
    """Build feature matrix for specified voices."""
    if voices is None:
        voices = ["MORMON", "NEPHI", "MORONI", "JACOB"]

    filtered = [
        b for b in blocks
        if b["target_size"] == target_size
        and b["quote_status"] == "original"
        and b["voice"] in voices
    ]

    X = np.array([extract_function_word_features(b["text"]) for b in filtered])
    y = np.array([b["voice"] for b in filtered])
    groups = np.array([b["run_id"] for b in filtered])

    return X, y, groups, filtered


# ============================================================
# ANALYSIS 1: UNMASKING
# ============================================================

def run_unmasking_analysis(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           n_iterations: int = 10, features_to_remove: int = 5) -> dict:
    """
    Unmasking analysis: iteratively remove top features and track accuracy degradation.

    Interpretation:
    - Rapid collapse → differences are shallow (topic/genre)
    - Gradual degradation → differences are deep (authorship)
    """
    print("Running unmasking analysis...")

    results = []
    feature_mask = np.ones(X.shape[1], dtype=bool)

    for iteration in range(n_iterations):
        X_masked = X[:, feature_mask]

        if X_masked.shape[1] < features_to_remove:
            break

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_masked)

        # Train classifier
        clf = LogisticRegression(class_weight='balanced', max_iter=1000,
                                  random_state=RANDOM_SEED)

        # CV
        n_groups = len(np.unique(groups))
        n_splits = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_splits)

        try:
            y_pred = cross_val_predict(clf, X_scaled, y, groups=groups, cv=gkf)
            bal_acc = balanced_accuracy_score(y, y_pred)
            macro_f1 = f1_score(y, y_pred, average='macro')
        except:
            bal_acc = 0.25
            macro_f1 = 0.0

        results.append({
            "iteration": iteration,
            "features_remaining": X_masked.shape[1],
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1
        })

        print(f"  Iter {iteration}: {X_masked.shape[1]} features, bal_acc={bal_acc:.3f}")

        # Find and remove top discriminating features
        clf.fit(X_scaled, y)
        importance = np.mean(np.abs(clf.coef_), axis=0)

        # Get indices of top features (in masked space)
        top_indices_masked = np.argsort(importance)[-features_to_remove:]

        # Map back to original feature indices
        original_indices = np.where(feature_mask)[0]
        for idx in top_indices_masked:
            if idx < len(original_indices):
                feature_mask[original_indices[idx]] = False

    return {
        "iterations": results,
        "interpretation": interpret_unmasking(results)
    }


def interpret_unmasking(results: list) -> str:
    """Interpret unmasking curve."""
    if len(results) < 3:
        return "Insufficient iterations for interpretation"

    initial_acc = results[0]["balanced_accuracy"]

    # Check if it starts above chance
    if initial_acc <= 0.30:
        return "No initial separation to unmask (started at/below chance)"

    # Check degradation rate
    mid_idx = len(results) // 2
    mid_acc = results[mid_idx]["balanced_accuracy"]

    drop = initial_acc - mid_acc

    if drop > 0.15:
        return "RAPID collapse - suggests shallow (topic/genre) differences"
    elif drop > 0.05:
        return "MODERATE degradation - mixed signal"
    else:
        return "GRADUAL degradation - suggests deep (authorial) differences"


# ============================================================
# ANALYSIS 2: PAIRWISE COMPARISONS
# ============================================================

def run_pairwise_analysis(blocks: list, target_size: int = 1000) -> dict:
    """
    Run pairwise comparisons between each pair of narrators.
    Focuses on Mormon vs Nephi (largest samples).
    """
    print("Running pairwise comparisons...")

    pairs = [
        ("MORMON", "NEPHI"),
        ("MORMON", "MORONI"),
        ("MORMON", "JACOB"),
        ("NEPHI", "MORONI"),
        ("NEPHI", "JACOB"),
        ("MORONI", "JACOB")
    ]

    results = {}

    for voice1, voice2 in pairs:
        print(f"  {voice1} vs {voice2}...")

        X, y, groups, _ = build_feature_matrix(blocks, target_size, [voice1, voice2])

        if len(X) < 10:
            results[f"{voice1}_vs_{voice2}"] = {
                "n_samples": len(X),
                "error": "Too few samples"
            }
            continue

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Classify
        clf = LogisticRegression(class_weight='balanced', max_iter=1000,
                                  random_state=RANDOM_SEED)

        n_groups = len(np.unique(groups))
        n_splits = min(5, n_groups)

        if n_splits < 2:
            results[f"{voice1}_vs_{voice2}"] = {
                "n_samples": len(X),
                "error": "Too few groups for CV"
            }
            continue

        gkf = GroupKFold(n_splits=n_splits)

        try:
            y_pred = cross_val_predict(clf, X_scaled, y, groups=groups, cv=gkf)
            bal_acc = balanced_accuracy_score(y, y_pred)
            accuracy = accuracy_score(y, y_pred)

            # Class breakdown
            class_counts = Counter(y)

            results[f"{voice1}_vs_{voice2}"] = {
                "n_samples": len(X),
                "n_voice1": class_counts[voice1],
                "n_voice2": class_counts[voice2],
                "accuracy": accuracy,
                "balanced_accuracy": bal_acc,
                "baseline": 0.5  # Binary classification
            }
        except Exception as e:
            results[f"{voice1}_vs_{voice2}"] = {
                "n_samples": len(X),
                "error": str(e)
            }

    return results


# ============================================================
# ANALYSIS 3: SAME-AUTHOR VERIFICATION
# ============================================================

def run_same_author_verification(X: np.ndarray, y: np.ndarray,
                                  n_pairs: int = 500) -> dict:
    """
    Same-author verification: compare distances between same-narrator pairs
    vs different-narrator pairs.

    If narrators are truly different, different-narrator pairs should have
    larger distances than same-narrator pairs.
    """
    print("Running same-author verification...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Get indices by class
    classes = sorted(set(y))
    class_indices = {c: np.where(y == c)[0] for c in classes}

    # Sample same-narrator pairs
    same_distances = []
    for cls in classes:
        indices = class_indices[cls]
        if len(indices) < 2:
            continue
        for _ in range(min(n_pairs // len(classes), len(indices) * (len(indices) - 1) // 2)):
            i, j = random.sample(list(indices), 2)
            dist = cosine(X_scaled[i], X_scaled[j])
            if not np.isnan(dist):
                same_distances.append(dist)

    # Sample different-narrator pairs
    diff_distances = []
    for _ in range(n_pairs):
        cls1, cls2 = random.sample(classes, 2)
        if len(class_indices[cls1]) == 0 or len(class_indices[cls2]) == 0:
            continue
        i = random.choice(class_indices[cls1])
        j = random.choice(class_indices[cls2])
        dist = cosine(X_scaled[i], X_scaled[j])
        if not np.isnan(dist):
            diff_distances.append(dist)

    same_distances = np.array(same_distances)
    diff_distances = np.array(diff_distances)

    # Statistics
    same_mean = np.mean(same_distances) if len(same_distances) > 0 else 0
    same_std = np.std(same_distances) if len(same_distances) > 0 else 0
    diff_mean = np.mean(diff_distances) if len(diff_distances) > 0 else 0
    diff_std = np.std(diff_distances) if len(diff_distances) > 0 else 0

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((same_std**2 + diff_std**2) / 2)
    cohens_d = (diff_mean - same_mean) / pooled_std if pooled_std > 0 else 0

    # Interpretation
    if cohens_d > 0.8:
        interpretation = "LARGE effect - different narrators are substantially more distant"
    elif cohens_d > 0.5:
        interpretation = "MEDIUM effect - some narrator differentiation"
    elif cohens_d > 0.2:
        interpretation = "SMALL effect - weak narrator differentiation"
    else:
        interpretation = "NEGLIGIBLE effect - narrators are NOT more distant than same-narrator pairs"

    return {
        "same_narrator_distance_mean": same_mean,
        "same_narrator_distance_std": same_std,
        "diff_narrator_distance_mean": diff_mean,
        "diff_narrator_distance_std": diff_std,
        "cohens_d": cohens_d,
        "n_same_pairs": len(same_distances),
        "n_diff_pairs": len(diff_distances),
        "interpretation": interpretation
    }


# ============================================================
# ANALYSIS 4: WITHIN-CLASS CONSISTENCY
# ============================================================

def run_within_class_consistency(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Check how consistent each narrator is with itself.

    High within-class variance suggests the "narrator" label doesn't
    correspond to a stable stylistic generator.
    """
    print("Running within-class consistency analysis...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classes = sorted(set(y))
    results = {}

    for cls in classes:
        indices = np.where(y == cls)[0]
        if len(indices) < 2:
            results[cls] = {"error": "Too few samples"}
            continue

        X_cls = X_scaled[indices]

        # Compute pairwise distances within class
        distances = cdist(X_cls, X_cls, metric='cosine')
        # Get upper triangle (excluding diagonal)
        upper_tri = distances[np.triu_indices(len(X_cls), k=1)]

        results[cls] = {
            "n_samples": len(indices),
            "mean_within_distance": np.mean(upper_tri),
            "std_within_distance": np.std(upper_tri),
            "min_within_distance": np.min(upper_tri),
            "max_within_distance": np.max(upper_tri)
        }

    # Compare: which narrator is most/least consistent?
    consistencies = {
        cls: r["mean_within_distance"]
        for cls, r in results.items()
        if "mean_within_distance" in r
    }

    if consistencies:
        most_consistent = min(consistencies, key=consistencies.get)
        least_consistent = max(consistencies, key=consistencies.get)
    else:
        most_consistent = least_consistent = None

    return {
        "by_narrator": results,
        "most_consistent": most_consistent,
        "least_consistent": least_consistent
    }


# ============================================================
# MAIN
# ============================================================

def generate_report(results: dict, output_path: Path):
    """Generate markdown report."""
    lines = [
        "# Additional Stylometric Analyses",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "These analyses investigate the null result from the primary classification.",
        "",
        "---",
        "",
        "## 1. Unmasking Analysis",
        "",
        "**Question:** Do differences collapse quickly (shallow/topic) or gradually (deep/authorial)?",
        "",
        f"**Interpretation:** {results['unmasking']['interpretation']}",
        "",
        "| Iteration | Features | Balanced Acc |",
        "|-----------|----------|--------------|",
    ]

    for r in results['unmasking']['iterations']:
        lines.append(f"| {r['iteration']} | {r['features_remaining']} | {r['balanced_accuracy']:.3f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Pairwise Comparisons",
        "",
        "**Question:** Can we distinguish pairs of narrators (simpler than 4-way)?",
        "",
        "| Comparison | N | Balanced Acc | vs 50% Baseline |",
        "|------------|---|--------------|-----------------|",
    ])

    for pair, r in results['pairwise'].items():
        if "error" in r:
            lines.append(f"| {pair.replace('_', ' ')} | {r['n_samples']} | ERROR | {r['error']} |")
        else:
            diff = r['balanced_accuracy'] - 0.5
            sign = "+" if diff > 0 else ""
            lines.append(f"| {pair.replace('_', ' ')} | {r['n_samples']} | {r['balanced_accuracy']:.1%} | {sign}{diff:.1%} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Same-Author Verification",
        "",
        "**Question:** Are cross-narrator pairs more distant than same-narrator pairs?",
        "",
        f"- Same-narrator distance: {results['verification']['same_narrator_distance_mean']:.4f} ± {results['verification']['same_narrator_distance_std']:.4f}",
        f"- Different-narrator distance: {results['verification']['diff_narrator_distance_mean']:.4f} ± {results['verification']['diff_narrator_distance_std']:.4f}",
        f"- **Cohen's d:** {results['verification']['cohens_d']:.3f}",
        f"- **Interpretation:** {results['verification']['interpretation']}",
        "",
        "---",
        "",
        "## 4. Within-Class Consistency",
        "",
        "**Question:** How internally consistent is each narrator?",
        "",
        "| Narrator | N | Mean Distance | Std |",
        "|----------|---|---------------|-----|",
    ])

    for narrator, r in results['consistency']['by_narrator'].items():
        if "error" in r:
            lines.append(f"| {narrator} | - | ERROR | - |")
        else:
            lines.append(f"| {narrator} | {r['n_samples']} | {r['mean_within_distance']:.4f} | {r['std_within_distance']:.4f} |")

    lines.extend([
        "",
        f"- **Most consistent:** {results['consistency']['most_consistent']}",
        f"- **Least consistent:** {results['consistency']['least_consistent']}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ])

    # Summary interpretation
    unmask_interp = results['unmasking']['interpretation']
    verif_d = results['verification']['cohens_d']

    if "No initial separation" in unmask_interp and verif_d < 0.3:
        lines.append("**Conclusion:** Multiple analyses confirm the null result. There is no detectable stylistic differentiation between claimed narrators using function words.")
    elif verif_d > 0.5:
        lines.append("**Conclusion:** While classification failed, distance-based verification shows SOME narrator differentiation. Signal may be present but weak.")
    else:
        lines.append("**Conclusion:** Results are mixed. Further investigation with control corpora needed.")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    print("=" * 70)
    print("ADDITIONAL STYLOMETRIC ANALYSES")
    print("=" * 70)
    print()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    print(f"Loading from {INPUT_FILE}...")
    data = load_blocks(INPUT_FILE)
    blocks = data["blocks"]

    # Build feature matrix
    X, y, groups, _ = build_feature_matrix(blocks)
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print()

    results = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(X),
            "n_features": X.shape[1]
        }
    }

    # Analysis 1: Unmasking
    print()
    results['unmasking'] = run_unmasking_analysis(X, y, groups)

    # Analysis 2: Pairwise
    print()
    results['pairwise'] = run_pairwise_analysis(blocks)

    # Analysis 3: Same-author verification
    print()
    results['verification'] = run_same_author_verification(X, y)

    # Analysis 4: Within-class consistency
    print()
    results['consistency'] = run_within_class_consistency(X, y)

    # Save
    print()
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Generating report at {REPORT_FILE}...")
    generate_report(results, REPORT_FILE)

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Unmasking: {results['unmasking']['interpretation']}")
    print(f"Verification Cohen's d: {results['verification']['cohens_d']:.3f}")
    print(f"  → {results['verification']['interpretation']}")
    print()
    print(f"Full report: {REPORT_FILE}")

    return results


if __name__ == "__main__":
    main()
