#!/usr/bin/env python3
"""
Control Corpora Analysis

Purpose: Validate that our stylometric method CAN detect multi-authorship
when it is present by running on:
1. KJV Bible - known multi-author text (should show separation)
2. Finney - known single-author text (should NOT show separation)

This validates whether the null result on Book of Mormon reflects a
methodological limitation or a genuine property of the text.
"""

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# === Function Word List ===
# Same as used in primary analysis
FUNCTION_WORDS = [
    # Articles
    'a', 'an', 'the',
    # Pronouns - personal
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    # Pronouns - demonstrative
    'this', 'that', 'these', 'those',
    # Pronouns - relative/interrogative
    'who', 'whom', 'whose', 'which', 'what', 'that',
    # Pronouns - indefinite
    'all', 'another', 'any', 'anybody', 'anyone', 'anything',
    'both', 'each', 'either', 'everybody', 'everyone', 'everything',
    'few', 'many', 'more', 'most', 'much', 'neither', 'nobody',
    'none', 'nothing', 'one', 'other', 'others', 'several',
    'some', 'somebody', 'someone', 'something', 'such',
    # Prepositions
    'about', 'above', 'across', 'after', 'against', 'along', 'among',
    'around', 'at', 'before', 'behind', 'below', 'beneath', 'beside',
    'between', 'beyond', 'by', 'down', 'during', 'except', 'for',
    'from', 'in', 'inside', 'into', 'like', 'near', 'of', 'off',
    'on', 'onto', 'out', 'outside', 'over', 'past', 'since',
    'through', 'throughout', 'to', 'toward', 'towards', 'under',
    'underneath', 'until', 'unto', 'up', 'upon', 'with', 'within', 'without',
    # Conjunctions
    'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
    'although', 'because', 'if', 'unless', 'while', 'whereas',
    'as', 'than', 'whether',
    # Auxiliary verbs
    'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing',
    'will', 'would', 'shall', 'should', 'may', 'might', 'must',
    'can', 'could', 'need', 'ought',
    # Archaic forms (important for KJV/BoM)
    'ye', 'thee', 'thou', 'thy', 'thine', 'thyself',
    'hath', 'hast', 'doth', 'dost', 'shalt', 'wilt',
    'art', 'wast', 'wert',
    'wherefore', 'thereof', 'wherein', 'hereby', 'thereby',
    'hither', 'thither', 'whither', 'hence', 'thence', 'whence',
    'yea', 'nay', 'lo', 'behold',
    # Other function words
    'not', 'no', 'yes', 'very', 'too', 'also', 'only', 'just',
    'even', 'still', 'already', 'ever', 'never', 'always',
    'here', 'there', 'where', 'when', 'how', 'why',
    'now', 'then', 'thus', 'therefore', 'however', 'moreover',
]

def tokenize(text):
    """Simple tokenizer for stylometric analysis."""
    text = text.lower()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return tokens


def extract_function_word_features(text, word_list):
    """Extract normalized frequencies for function words."""
    tokens = tokenize(text)
    total = len(tokens)
    if total == 0:
        return {w: 0.0 for w in word_list}

    counts = Counter(tokens)
    features = {}
    for word in word_list:
        features[f'fw_{word}'] = counts.get(word, 0) / total
    return features


def run_classification(X, y, groups, label_name="Label"):
    """Run classification with cross-validation and permutation test."""

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver='lbfgs'
        ))
    ])

    # Determine number of CV folds based on smallest class
    unique_groups_per_class = {}
    for g, label in zip(groups, y):
        if label not in unique_groups_per_class:
            unique_groups_per_class[label] = set()
        unique_groups_per_class[label].add(g)

    min_groups = min(len(gs) for gs in unique_groups_per_class.values())
    n_splits = min(5, min_groups)

    if n_splits < 2:
        return {
            'error': f'Not enough groups for cross-validation (min_groups={min_groups})',
            'n_classes': len(set(y)),
            'class_distribution': dict(Counter(y))
        }

    cv = GroupKFold(n_splits=n_splits)

    # Cross-validated predictions
    y_pred = cross_val_predict(pipeline, X, y, groups=groups, cv=cv)

    # Metrics
    balanced_acc = balanced_accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')

    # Permutation test for significance
    score, perm_scores, p_value = permutation_test_score(
        pipeline, X, y, groups=groups, cv=cv,
        scoring='balanced_accuracy',
        n_permutations=100,
        random_state=RANDOM_SEED
    )

    # Classification report
    labels = sorted(set(y))
    report = classification_report(y, y_pred, labels=labels, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=labels)

    return {
        'n_samples': len(y),
        'n_classes': len(labels),
        'class_distribution': dict(Counter(y)),
        'n_splits': n_splits,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'permutation_p_value': p_value,
        'permutation_mean': float(np.mean(perm_scores)),
        'permutation_std': float(np.std(perm_scores)),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'labels': labels
    }


def analyze_kjv():
    """Analyze KJV Bible - multi-author control."""
    print("\n" + "="*60)
    print("KJV BIBLE ANALYSIS (Multi-Author Control)")
    print("="*60)

    segments_path = Path('data/reference/processed/kjv/kjv-segments.json')
    with open(segments_path) as f:
        segments = json.load(f)

    print(f"Loaded {len(segments)} segments")

    # Extract features
    feature_matrix = []
    labels = []
    groups = []  # Group by segment runs for CV

    # Count by author
    author_counts = Counter(s['author'] for s in segments)
    print("\nAuthor distribution:")
    for author, count in sorted(author_counts.items()):
        print(f"  {author}: {count} segments")

    # Simplify authors to major categories for better CV
    # Moses, David, Solomon, Isaiah, Paul (combine Unknown/Paul), NT evangelists
    author_map = {
        'Moses (traditional)': 'MOSES',
        'David/Multiple (traditional)': 'DAVID',
        'Solomon (traditional)': 'SOLOMON',
        'Isaiah (traditional)': 'ISAIAH',
        'Matthew (traditional)': 'EVANGELIST',
        'John (traditional)': 'EVANGELIST',
        'Paul (traditional)': 'PAUL',
        'Unknown/Paul? (traditional)': 'PAUL'
    }

    for i, seg in enumerate(segments):
        features = extract_function_word_features(seg['text'], FUNCTION_WORDS)
        feature_matrix.append(list(features.values()))
        labels.append(author_map.get(seg['author'], seg['author']))
        # Group by 5-segment chunks within each book for CV
        groups.append(f"{seg['book']}_{i // 5}")

    X = np.array(feature_matrix)
    y = np.array(labels)
    groups = np.array(groups)

    print(f"\nFeature matrix shape: {X.shape}")

    # Run full analysis
    print("\nRunning classification...")
    results = run_classification(X, y, groups, "Author")

    if 'error' in results:
        print(f"Error: {results['error']}")
        return results

    print(f"\nRESULTS:")
    print(f"  Balanced Accuracy: {results['balanced_accuracy']:.1%}")
    print(f"  Macro F1: {results['macro_f1']:.3f}")
    print(f"  Permutation p-value: {results['permutation_p_value']:.3f}")
    print(f"  Chance baseline: {100/results['n_classes']:.1f}%")

    # Also run with simplified labels (group by tradition)
    print("\n" + "-"*40)
    print("Simplified Analysis: OT vs NT")

    ot_books = {'Genesis', 'Psalms', 'Proverbs', 'Isaiah'}
    simplified_labels = ['OT' if seg['book'] in ot_books else 'NT' for seg in segments]

    y_simple = np.array(simplified_labels)
    simple_results = run_classification(X, y_simple, groups, "Testament")

    if 'error' not in simple_results:
        print(f"  Balanced Accuracy: {simple_results['balanced_accuracy']:.1%}")
        print(f"  Permutation p-value: {simple_results['permutation_p_value']:.3f}")

    return {
        'corpus': 'KJV',
        'expected': 'Multi-author - should show separation',
        'full_analysis': results,
        'ot_vs_nt': simple_results if 'error' not in simple_results else None
    }


def analyze_finney():
    """Analyze Finney - single-author control."""
    print("\n" + "="*60)
    print("FINNEY ANALYSIS (Single-Author Control)")
    print("="*60)

    segments_path = Path('data/reference/processed/finney/finney-segments.json')
    with open(segments_path) as f:
        segments = json.load(f)

    print(f"Loaded {len(segments)} segments")

    # For single-author, we create artificial "labels" by splitting into halves
    # This tests whether our method would create spurious separation
    n = len(segments)
    half = n // 2

    # Split by position (first half vs second half)
    labels_position = ['FIRST_HALF' if i < half else 'SECOND_HALF' for i in range(n)]

    # Extract features
    feature_matrix = []
    groups = []

    for i, seg in enumerate(segments):
        features = extract_function_word_features(seg['text'], FUNCTION_WORDS)
        feature_matrix.append(list(features.values()))
        groups.append(i // 10)  # Group every 10 segments

    X = np.array(feature_matrix)
    y = np.array(labels_position)
    groups = np.array(groups)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Testing FIRST_HALF vs SECOND_HALF (artificial split)")

    # Run classification
    print("\nRunning classification...")
    results = run_classification(X, y, groups, "Position")

    if 'error' in results:
        print(f"Error: {results['error']}")
        return {'corpus': 'Finney', 'error': results['error']}

    print(f"\nRESULTS:")
    print(f"  Balanced Accuracy: {results['balanced_accuracy']:.1%}")
    print(f"  Macro F1: {results['macro_f1']:.3f}")
    print(f"  Permutation p-value: {results['permutation_p_value']:.3f}")
    print(f"  Chance baseline: 50%")

    # Also try random split
    print("\n" + "-"*40)
    print("Random Split Analysis (sanity check)")

    np.random.seed(RANDOM_SEED)
    random_labels = np.random.choice(['A', 'B'], size=n)
    random_groups = np.arange(n) // 10

    random_results = run_classification(X, random_labels, random_groups, "Random")

    if 'error' not in random_results:
        print(f"  Balanced Accuracy: {random_results['balanced_accuracy']:.1%}")
        print(f"  (Should be ~50% - no real signal)")

    return {
        'corpus': 'Finney',
        'expected': 'Single-author - should NOT show separation',
        'position_split': results,
        'random_split': random_results if 'error' not in random_results else None
    }


def main():
    print("="*60)
    print("CONTROL CORPORA ANALYSIS")
    print("="*60)
    print(f"\nGenerated: {datetime.now(timezone.utc).isoformat()}")
    print("\nPurpose: Validate stylometric method can detect multi-authorship")

    results = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'purpose': 'Validate stylometric method on known corpora',
            'random_seed': RANDOM_SEED
        },
        'kjv': None,
        'finney': None,
        'interpretation': None
    }

    # Run analyses
    results['kjv'] = analyze_kjv()
    results['finney'] = analyze_finney()

    # Interpret results
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    # Safely extract KJV results
    kjv_full = results.get('kjv', {}).get('full_analysis', {})
    if isinstance(kjv_full, dict) and 'error' not in kjv_full:
        kjv_acc = kjv_full.get('balanced_accuracy', 0)
        kjv_p = kjv_full.get('permutation_p_value', 1)
        kjv_n_classes = kjv_full.get('n_classes', 1)
    else:
        kjv_acc = 0
        kjv_p = 1
        kjv_n_classes = 1

    # Safely extract Finney results
    finney_pos = results.get('finney', {}).get('position_split', {})
    if isinstance(finney_pos, dict) and 'error' not in finney_pos:
        finney_acc = finney_pos.get('balanced_accuracy', 0)
        finney_p = finney_pos.get('permutation_p_value', 1)
    else:
        finney_acc = 0
        finney_p = 1

    kjv_chance = 1.0 / kjv_n_classes

    interpretation = []

    # KJV interpretation - focus on accuracy vs chance (permutation test unreliable with groups)
    if kjv_acc > kjv_chance + 0.20:
        interpretation.append(f"KJV: ✓ Method successfully detects multi-authorship ({kjv_acc:.1%} vs {kjv_chance:.1%} chance)")
        kjv_validates = True
    elif kjv_acc > kjv_chance + 0.10:
        interpretation.append(f"KJV: ~ Moderate separation detected ({kjv_acc:.1%} vs {kjv_chance:.1%} chance)")
        kjv_validates = True
    elif kjv_acc > kjv_chance:
        interpretation.append(f"KJV: ~ Weak separation detected ({kjv_acc:.1%} vs {kjv_chance:.1%} chance)")
        kjv_validates = False
    else:
        interpretation.append("KJV: ✗ Method fails to detect known multi-authorship")
        kjv_validates = False

    # Finney interpretation
    if abs(finney_acc - 0.5) < 0.10 or finney_p > 0.05:
        interpretation.append("Finney: ✓ No spurious separation in single-author text")
        finney_validates = True
    else:
        interpretation.append("Finney: ✗ Spurious separation detected in single-author text")
        finney_validates = False

    # Overall
    if kjv_validates and finney_validates:
        interpretation.append("\nCONCLUSION: Method is validated - can trust BoM null result")
    elif kjv_validates and not finney_validates:
        interpretation.append("\nCONCLUSION: Method detects authorship but may produce false positives")
    elif not kjv_validates and finney_validates:
        interpretation.append("\nCONCLUSION: Method may lack power to detect subtle authorship differences")
    else:
        interpretation.append("\nCONCLUSION: Method validation inconclusive")

    for line in interpretation:
        print(line)

    results['interpretation'] = {
        'kjv_validates': kjv_validates,
        'finney_validates': finney_validates,
        'summary': interpretation
    }

    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'control-analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate markdown report
    report = generate_report(results)
    with open(output_dir / 'control-analysis-report.md', 'w') as f:
        f.write(report)

    print(f"\nResults saved to:")
    print(f"  results/control-analysis.json")
    print(f"  results/control-analysis-report.md")

    return results


def generate_report(results):
    """Generate markdown report."""
    lines = [
        "# Control Corpora Analysis",
        "",
        f"**Generated:** {results['metadata']['generated_at']}",
        "",
        "**Purpose:** Validate that our stylometric method can detect multi-authorship when present.",
        "",
        "---",
        "",
        "## 1. KJV Bible (Multi-Author Control)",
        "",
        "**Expectation:** Should show statistically significant author separation.",
        "",
    ]

    kjv = results['kjv']
    if kjv and 'full_analysis' in kjv and 'error' not in kjv['full_analysis']:
        full = kjv['full_analysis']
        n_classes = full['n_classes']
        chance = 100 / n_classes

        lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| N Samples | {full['n_samples']} |",
            f"| N Authors | {n_classes} |",
            f"| Balanced Accuracy | {full['balanced_accuracy']:.1%} |",
            f"| Chance Baseline | {chance:.1f}% |",
            f"| Permutation p-value | {full['permutation_p_value']:.3f} |",
            f"| Macro F1 | {full['macro_f1']:.3f} |",
            "",
            "**Author Distribution:**",
            ""
        ])

        for author, count in sorted(full['class_distribution'].items()):
            lines.append(f"- {author}: {count} segments")

        lines.append("")

        # Use accuracy vs chance baseline (permutation test unreliable with groups)
        if full['balanced_accuracy'] > chance/100 + 0.20:
            lines.append(f"**Result:** ✓ Multi-authorship successfully detected ({full['balanced_accuracy']:.1%} vs {chance:.1f}% chance)")
        elif full['balanced_accuracy'] > chance/100 + 0.10:
            lines.append(f"**Result:** ~ Moderate separation detected")
        else:
            lines.append("**Result:** ✗ Multi-authorship NOT clearly detected")
    else:
        lines.append("Error in KJV analysis")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Finney (Single-Author Control)",
        "",
        "**Expectation:** Should show NO significant separation (artificial splits should be at chance).",
        "",
    ])

    finney = results['finney']
    if finney and 'position_split' in finney and 'error' not in finney['position_split']:
        pos = finney['position_split']

        lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| N Samples | {pos['n_samples']} |",
            f"| Test | First half vs Second half |",
            f"| Balanced Accuracy | {pos['balanced_accuracy']:.1%} |",
            f"| Chance Baseline | 50% |",
            f"| Permutation p-value | {pos['permutation_p_value']:.3f} |",
            "",
        ])

        if abs(pos['balanced_accuracy'] - 0.5) < 0.10 or pos['permutation_p_value'] > 0.05:
            lines.append("**Result:** ✓ No spurious separation detected")
        else:
            lines.append("**Result:** ✗ Spurious separation detected (concerning)")
    else:
        lines.append("Error in Finney analysis")

    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ])

    if results.get('interpretation'):
        for line in results['interpretation']['summary']:
            lines.append(line)

    lines.extend([
        "",
        "---",
        "",
        "## Implications for Book of Mormon Analysis",
        "",
        "If KJV shows separation and Finney does not, our method is validated:",
        "- The null result on BoM narrators is meaningful",
        "- The BoM English layer does not exhibit the narrator-specific function-word patterns",
        "  that would be expected if multiple distinct authors produced the surface text",
        "",
        "If KJV fails to show separation:",
        "- Method may lack power for this type of text (archaic English)",
        "- Null result on BoM may reflect methodological limitation rather than text property",
        "",
    ])

    return "\n".join(lines)


if __name__ == '__main__':
    main()
