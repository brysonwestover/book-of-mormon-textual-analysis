#!/usr/bin/env python3
"""
Phase 2.D: Translation-Layer Calibration (Garnett Corpus)

This analysis tests whether stylometry can recover authorial signal through
a single translator's voice, using Constance Garnett's translations of
Russian literature.

Research Question:
Can function-word stylometry distinguish between 4 Russian authors
(Dostoevsky, Tolstoy, Chekhov, Turgenev) when all works are translated
by the same person (Constance Garnett)?

Predictions (from pre-registration):
- D1: If accuracy >> chance → Translation does NOT erase authorial signal
      → BoM null result is informative
- D2: If accuracy ≈ chance → Translation DOES erase signal
      → BoM null may be expected under H4 (ancient multi-author)

Methodology (matched to BoM analysis):
- Same block size: 1000 words
- Same features: Function words (adapted for non-archaic English)
- Same model: Logistic Regression with balanced weights
- Grouping: By work (novel/story collection)
- CV: Leave-one-work-out
- Metric: Work-weighted balanced accuracy

Version: 1.0.0
Date: 2026-02-05
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
import warnings

# Configuration
GARNETT_DIR = Path("data/reference/garnett/raw")
MANIFEST_FILE = GARNETT_DIR / "manifest.json"
OUTPUT_FILE = Path("results/garnett-analysis-results.json")
REPORT_FILE = Path("results/garnett-analysis-report.md")

RANDOM_SEED = 42
BLOCK_SIZE = 1000  # Words per block (matched to BoM primary analysis)
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 1000

# Function words for modern English (Garnett's translation style)
# Adapted from BoM list - removed archaic forms, kept core function words
FUNCTION_WORDS = [
    # Articles
    "a", "an", "the",
    # Personal pronouns
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    # Relative/interrogative pronouns
    "who", "whom", "whose", "which", "what", "that",
    # Demonstratives
    "this", "these", "those",
    # Quantifiers
    "all", "any", "both", "each", "every", "few", "many", "more", "most",
    "much", "neither", "none", "one", "other", "several", "some", "such",
    # Coordinating conjunctions
    "and", "but", "or", "nor", "for", "yet", "so",
    # Subordinating conjunctions
    "if", "then", "because", "although", "though", "while", "unless",
    "when", "where", "before", "after", "since", "until", "as",
    # Prepositions
    "in", "on", "at", "by", "with", "about", "against", "between",
    "into", "through", "during", "above", "below", "to", "from", "up", "down",
    "out", "off", "over", "under", "again", "further",
    "of", "upon", "among", "concerning", "according",
    # Be verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    # Have verbs
    "have", "has", "had", "having",
    # Do verbs
    "do", "does", "did", "doing",
    # Modal verbs
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    # Negation
    "not", "no", "never", "nothing", "nobody", "nowhere",
    # Adverbs
    "very", "too", "also", "only", "just", "even", "still", "already",
    "now", "here", "there", "therefore", "thus", "hence",
    "how", "why",
    # Additional common function words
    "than", "whether", "either", "nor", "once", "whether",
    "however", "moreover", "nevertheless", "meanwhile",
    "perhaps", "quite", "rather", "almost", "always", "often", "sometimes",
]

WORD_TO_IDX = {w: i for i, w in enumerate(FUNCTION_WORDS)}


def clean_text(text: str) -> str:
    """Clean and normalize text for analysis."""
    # Remove Project Gutenberg headers/footers
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    for marker in start_markers:
        if marker in text:
            idx = text.find(marker)
            # Find the next double newline after the marker
            next_para = text.find("\n\n", idx)
            if next_para != -1:
                text = text[next_para:]

    for marker in end_markers:
        if marker in text:
            idx = text.find(marker)
            text = text[:idx]

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_function_word_features(text: str) -> np.ndarray:
    """Extract function word frequency features."""
    words = text.lower().split()
    total = len(words)
    if total == 0:
        return np.zeros(len(FUNCTION_WORDS))

    counts = np.zeros(len(FUNCTION_WORDS))
    for word in words:
        word_clean = re.sub(r'[^a-z]', '', word)
        if word_clean in WORD_TO_IDX:
            counts[WORD_TO_IDX[word_clean]] += 1

    return (counts / total) * 1000  # Per 1000 words


def create_blocks(text: str, block_size: int = BLOCK_SIZE) -> list:
    """Split text into non-overlapping blocks of approximately block_size words."""
    words = text.split()
    blocks = []

    for i in range(0, len(words), block_size):
        block_words = words[i:i + block_size]
        if len(block_words) >= block_size * 0.8:  # Keep blocks at least 80% full
            blocks.append(' '.join(block_words))

    return blocks


def load_garnett_corpus(novels_only: bool = False) -> dict:
    """Load all Garnett translations and organize by author/work.

    Args:
        novels_only: If True, exclude short story collections (Chekhov)
    """
    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    corpus = {}

    for work_info in manifest:
        author = work_info['author']
        title = work_info['title']
        filename = work_info['filename']
        genre = work_info.get('genre', 'unknown')

        # Skip story collections if novels_only
        if novels_only and genre == 'stories':
            continue

        filepath = GARNETT_DIR / filename
        if not filepath.exists():
            print(f"  Warning: {filename} not found, skipping")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        clean = clean_text(raw_text)
        word_count = len(clean.split())

        if author not in corpus:
            corpus[author] = []

        corpus[author].append({
            'title': title,
            'text': clean,
            'word_count': word_count,
            'filename': filename,
            'genre': genre,
        })

    return corpus


def build_dataset(corpus: dict) -> tuple:
    """Build feature matrix and labels from corpus.

    Returns:
        X: Feature matrix
        y: Author labels
        work_ids: Work ID for each block
        sample_weights: Weight for each block (1 / blocks_in_work)
        work_metadata: Dict mapping work_id to metadata
    """
    X = []
    y = []
    work_ids = []
    sample_weights = []
    work_metadata = {}

    work_id = 0
    for author, works in sorted(corpus.items()):
        for work in works:
            blocks = create_blocks(work['text'])
            n_blocks = len(blocks)

            work_metadata[work_id] = {
                'author': author,
                'title': work['title'],
                'genre': work.get('genre', 'unknown'),
                'n_blocks': n_blocks,
                'word_count': work['word_count'],
            }

            # Weight each block so each work contributes equally
            block_weight = 1.0 / n_blocks if n_blocks > 0 else 0

            for block in blocks:
                features = extract_function_word_features(block)
                X.append(features)
                y.append(author)
                work_ids.append(work_id)
                sample_weights.append(block_weight)

            work_id += 1

    return (np.array(X), np.array(y), np.array(work_ids),
            np.array(sample_weights), work_metadata)


def run_leave_one_work_out_cv(X: np.ndarray, y: np.ndarray,
                               work_ids: np.ndarray,
                               sample_weights: np.ndarray = None) -> dict:
    """Run leave-one-work-out cross-validation."""
    unique_works = np.unique(work_ids)

    y_true_all = []
    y_pred_all = []
    work_results = {}

    for held_out_work in unique_works:
        train_mask = work_ids != held_out_work
        test_mask = work_ids == held_out_work

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train classifier with sample weights if provided
        clf = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sample_weights is not None:
                train_weights = sample_weights[train_mask]
                clf.fit(X_train_scaled, y_train, sample_weight=train_weights)
            else:
                clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        # Work-level accuracy
        work_acc = np.mean(y_test == y_pred)
        work_author = y_test[0]
        work_results[int(held_out_work)] = {
            'author': work_author,
            'accuracy': float(work_acc),
            'n_blocks': int(len(y_test))
        }

    # Compute work-weighted balanced accuracy
    author_work_accs = defaultdict(list)
    for work_id, result in work_results.items():
        author_work_accs[result['author']].append(result['accuracy'])

    class_means = [np.mean(accs) for accs in author_work_accs.values()]
    work_weighted_balanced_acc = np.mean(class_means)

    # Block-level balanced accuracy
    block_balanced_acc = balanced_accuracy_score(y_true_all, y_pred_all)

    return {
        'work_weighted_balanced_accuracy': float(work_weighted_balanced_acc),
        'block_balanced_accuracy': float(block_balanced_acc),
        'work_results': work_results,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'author_work_accuracies': {k: [float(v) for v in vals]
                                    for k, vals in author_work_accs.items()}
    }


def run_permutation_test(X: np.ndarray, y: np.ndarray,
                         work_ids: np.ndarray, observed_score: float,
                         n_permutations: int = N_PERMUTATIONS) -> dict:
    """Run permutation test at the work level."""
    # Get work-level labels
    work_to_author = {}
    for i, (work_id, author) in enumerate(zip(work_ids, y)):
        work_to_author[work_id] = author

    unique_works = list(work_to_author.keys())
    work_authors = [work_to_author[w] for w in unique_works]

    rng = np.random.RandomState(RANDOM_SEED)
    null_scores = []

    print(f"  Running {n_permutations} permutations...")

    for perm_i in range(n_permutations):
        if (perm_i + 1) % 1000 == 0:
            print(f"    Permutation {perm_i + 1}/{n_permutations}")

        # Permute work-level labels
        permuted_authors = work_authors.copy()
        rng.shuffle(permuted_authors)

        # Create new block-level labels
        work_to_perm_author = dict(zip(unique_works, permuted_authors))
        y_permuted = np.array([work_to_perm_author[w] for w in work_ids])

        # Run CV with permuted labels
        result = run_leave_one_work_out_cv(X, y_permuted, work_ids)
        null_scores.append(result['work_weighted_balanced_accuracy'])

    null_scores = np.array(null_scores)
    p_value = (1 + np.sum(null_scores >= observed_score)) / (1 + n_permutations)

    return {
        'observed_score': float(observed_score),
        'p_value': float(p_value),
        'null_mean': float(np.mean(null_scores)),
        'null_std': float(np.std(null_scores)),
        'null_95_percentile': float(np.percentile(null_scores, 95)),
        'n_permutations': n_permutations
    }


def run_bootstrap_ci(X: np.ndarray, y: np.ndarray,
                     work_ids: np.ndarray,
                     n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """Compute bootstrap confidence interval at work level."""
    unique_works = np.unique(work_ids)
    work_to_author = {}
    for i, (work_id, author) in enumerate(zip(work_ids, y)):
        work_to_author[work_id] = author

    # Get author for each work
    author_works = defaultdict(list)
    for work_id in unique_works:
        author_works[work_to_author[work_id]].append(work_id)

    rng = np.random.RandomState(RANDOM_SEED)
    bootstrap_scores = []

    print(f"  Running {n_bootstrap} bootstrap samples...")

    for boot_i in range(n_bootstrap):
        if (boot_i + 1) % 200 == 0:
            print(f"    Bootstrap {boot_i + 1}/{n_bootstrap}")

        # Stratified bootstrap at work level
        sampled_works = []
        for author, works in author_works.items():
            sampled = rng.choice(works, size=len(works), replace=True)
            sampled_works.extend(sampled)

        # Build bootstrap dataset
        boot_mask = np.isin(work_ids, sampled_works)
        X_boot = X[boot_mask]
        y_boot = y[boot_mask]
        work_ids_boot = work_ids[boot_mask]

        result = run_leave_one_work_out_cv(X_boot, y_boot, work_ids_boot)
        bootstrap_scores.append(result['work_weighted_balanced_accuracy'])

    bootstrap_scores = np.array(bootstrap_scores)

    return {
        'mean': float(np.mean(bootstrap_scores)),
        'std': float(np.std(bootstrap_scores)),
        'ci_95_lower': float(np.percentile(bootstrap_scores, 2.5)),
        'ci_95_upper': float(np.percentile(bootstrap_scores, 97.5)),
        'n_bootstrap': n_bootstrap
    }


def generate_report(results: dict) -> str:
    """Generate markdown report."""
    cv = results['cross_validation']
    perm = results['permutation']
    boot = results['bootstrap']
    corpus = results['corpus_stats']

    lines = [
        "# Phase 2.D: Translation-Layer Calibration (Garnett Corpus)",
        "",
        "## Research Question",
        "",
        "Can function-word stylometry distinguish between Russian authors",
        "(Dostoevsky, Tolstoy, Chekhov, Turgenev) when all works are translated",
        "by a single translator (Constance Garnett)?",
        "",
        "## Corpus Statistics",
        "",
        f"- **Total works**: {corpus['total_works']}",
        f"- **Total words**: {corpus['total_words']:,}",
        f"- **Total blocks**: {corpus['total_blocks']}",
        f"- **Block size**: {BLOCK_SIZE} words",
        "",
        "### By Author",
        "",
        "| Author | Works | Words | Blocks |",
        "|--------|-------|-------|--------|",
    ]

    for author, stats in sorted(corpus['by_author'].items()):
        lines.append(f"| {author} | {stats['works']} | {stats['words']:,} | {stats['blocks']} |")

    lines.extend([
        "",
        "## Primary Results",
        "",
        f"- **Work-weighted balanced accuracy**: {cv['work_weighted_balanced_accuracy']:.1%}",
        f"- **Block-level balanced accuracy**: {cv['block_balanced_accuracy']:.1%}",
        f"- **Chance level**: 25% (4 classes)",
        "",
        "### Statistical Significance",
        "",
        f"- **Permutation p-value**: {perm['p_value']:.4f}",
        f"- **Null distribution mean**: {perm['null_mean']:.1%}",
        f"- **Null 95th percentile**: {perm['null_95_percentile']:.1%}",
        "",
        "### Confidence Interval",
        "",
        f"- **Bootstrap 95% CI**: [{boot['ci_95_lower']:.1%}, {boot['ci_95_upper']:.1%}]",
        "",
        "## Per-Author Performance",
        "",
        "| Author | Works | Mean Work Accuracy |",
        "|--------|-------|--------------------|",
    ])

    for author, accs in sorted(cv['author_work_accuracies'].items()):
        mean_acc = np.mean(accs)
        lines.append(f"| {author} | {len(accs)} | {mean_acc:.1%} |")

    # Interpretation
    acc = cv['work_weighted_balanced_accuracy']
    p = perm['p_value']

    if acc > 0.50 and p < 0.05:
        interpretation = "STRONG"
        detail = "Translation does NOT erase authorial signal. The BoM null result is highly informative."
    elif acc > 0.35 and p < 0.05:
        interpretation = "MODERATE"
        detail = "Translation preserves detectable authorial signal. The BoM null result is informative."
    elif acc > 0.30 and p < 0.05:
        interpretation = "WEAK"
        detail = "Translation partially preserves authorial signal. BoM interpretation requires caution."
    else:
        interpretation = "NONE"
        detail = "Translation appears to erase authorial signal. BoM null may be expected under H4."

    lines.extend([
        "",
        "## Interpretation",
        "",
        f"**Signal strength**: {interpretation}",
        "",
        detail,
        "",
        "### Implications for BoM Analysis",
        "",
    ])

    if interpretation in ["STRONG", "MODERATE"]:
        lines.extend([
            "Since stylometry CAN recover authorial signal through Garnett's translation,",
            "the BoM null result suggests that the claimed narrators do not exhibit",
            "distinguishable function-word patterns in the English text.",
            "",
            "This is consistent with:",
            "- H1: Single modern author",
            "- H3: Single author mimicking multiple voices",
            "",
            "And less consistent with:",
            "- H4: Multiple ancient authors through single translator (if translation",
            "  preserved style as it does for Garnett)",
        ])
    else:
        lines.extend([
            "Since stylometry CANNOT reliably recover authorial signal through Garnett's",
            "translation, the BoM null result is ambiguous. Translation homogenization",
            "may explain the lack of narrator-specific patterns.",
            "",
            "The BoM null result remains consistent with all hypotheses (H1-H5).",
        ])

    lines.extend([
        "",
        f"*Generated: {results['metadata']['generated_at']}*"
    ])

    return '\n'.join(lines)


def run_analysis(corpus: dict, analysis_name: str) -> dict:
    """Run full analysis pipeline on a corpus."""
    print(f"\n{'='*60}")
    print(f"Analysis: {analysis_name}")
    print(f"{'='*60}")

    # Compute corpus stats
    corpus_stats = {
        'total_works': 0,
        'total_words': 0,
        'total_blocks': 0,
        'by_author': {}
    }

    for author, works in sorted(corpus.items()):
        author_words = sum(w['word_count'] for w in works)
        author_blocks = sum(len(create_blocks(w['text'])) for w in works)

        corpus_stats['total_works'] += len(works)
        corpus_stats['total_words'] += author_words
        corpus_stats['total_blocks'] += author_blocks
        corpus_stats['by_author'][author] = {
            'works': len(works),
            'words': author_words,
            'blocks': author_blocks
        }

        print(f"  {author}: {len(works)} works, {author_words:,} words, {author_blocks} blocks")

    print(f"\n  Total: {corpus_stats['total_works']} works, "
          f"{corpus_stats['total_words']:,} words, {corpus_stats['total_blocks']} blocks")

    # Build dataset
    print("\nBuilding feature matrix...")
    X, y, work_ids, sample_weights, work_metadata = build_dataset(corpus)
    print(f"  {X.shape[0]} blocks, {X.shape[1]} features")

    # Run cross-validation with sample weights
    print("\nRunning leave-one-work-out cross-validation (with block weighting)...")
    cv_results = run_leave_one_work_out_cv(X, y, work_ids, sample_weights)
    print(f"  Work-weighted balanced accuracy: {cv_results['work_weighted_balanced_accuracy']:.1%}")
    print(f"  Block-level balanced accuracy: {cv_results['block_balanced_accuracy']:.1%}")

    # Permutation test
    print("\nRunning permutation test...")
    perm_results = run_permutation_test(
        X, y, work_ids,
        cv_results['work_weighted_balanced_accuracy'],
        N_PERMUTATIONS
    )
    print(f"  p-value: {perm_results['p_value']:.4f}")

    # Bootstrap CI
    print("\nRunning bootstrap confidence interval...")
    boot_results = run_bootstrap_ci(X, y, work_ids, N_BOOTSTRAP)
    print(f"  95% CI: [{boot_results['ci_95_lower']:.1%}, {boot_results['ci_95_upper']:.1%}]")

    return {
        'analysis_name': analysis_name,
        'corpus_stats': corpus_stats,
        'work_metadata': {str(k): v for k, v in work_metadata.items()},
        'cross_validation': cv_results,
        'permutation': perm_results,
        'bootstrap': boot_results,
    }


def main():
    print("=" * 70)
    print("PHASE 2.D: TRANSLATION-LAYER CALIBRATION (GARNETT CORPUS)")
    print("=" * 70)

    results = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'script_version': '2.0.0',
            'random_seed': RANDOM_SEED,
            'block_size': BLOCK_SIZE,
            'n_features': len(FUNCTION_WORDS),
            'methodology_notes': [
                'Block weighting: each work contributes equally regardless of length',
                'Leave-one-work-out CV prevents train/test leakage',
                'Work-level permutation respects non-independence of blocks',
            ]
        },
        'analyses': {}
    }

    # PRIMARY ANALYSIS: Novels only (excludes Chekhov story collections)
    print("\n" + "=" * 70)
    print("PRIMARY ANALYSIS: NOVELS ONLY")
    print("(Excludes Chekhov story collections to control for genre)")
    print("=" * 70)

    corpus_novels = load_garnett_corpus(novels_only=True)
    novels_results = run_analysis(corpus_novels, "Novels Only (Primary)")
    results['analyses']['novels_only'] = novels_results

    # SECONDARY ANALYSIS: Full corpus
    print("\n" + "=" * 70)
    print("SECONDARY ANALYSIS: FULL CORPUS")
    print("(All works including Chekhov story collections)")
    print("=" * 70)

    corpus_full = load_garnett_corpus(novels_only=False)
    full_results = run_analysis(corpus_full, "Full Corpus (Secondary)")
    results['analyses']['full_corpus'] = full_results

    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Generate report
    report = generate_report_v2(results)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {REPORT_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, analysis in results['analyses'].items():
        acc = analysis['cross_validation']['work_weighted_balanced_accuracy']
        p = analysis['permutation']['p_value']
        n_works = analysis['corpus_stats']['total_works']
        n_authors = len(analysis['corpus_stats']['by_author'])

        print(f"\n{analysis['analysis_name']}:")
        print(f"  Works: {n_works}, Authors: {n_authors}")
        print(f"  Accuracy: {acc:.1%} (chance = {100/n_authors:.0f}%)")
        print(f"  p-value: {p:.4f}")

        chance = 1.0 / n_authors
        if acc > 0.50 and p < 0.05:
            print(f"  → STRONG signal detected")
        elif acc > chance + 0.15 and p < 0.05:
            print(f"  → MODERATE signal detected")
        elif p < 0.05:
            print(f"  → WEAK signal detected")
        else:
            print(f"  → NO significant signal")


def generate_report_v2(results: dict) -> str:
    """Generate comprehensive markdown report."""
    lines = [
        "# Phase 2.D: Translation-Layer Calibration (Garnett Corpus)",
        "",
        "## Research Question",
        "",
        "Can function-word stylometry distinguish between Russian authors",
        "(Dostoevsky, Tolstoy, Chekhov, Turgenev) when all works are translated",
        "by a single translator (Constance Garnett)?",
        "",
        "## Methodology",
        "",
        f"- **Block size**: {BLOCK_SIZE} words",
        f"- **Features**: {len(FUNCTION_WORDS)} function words",
        "- **Classifier**: Logistic Regression (balanced class weights)",
        "- **CV**: Leave-one-work-out",
        "- **Block weighting**: Each work contributes equally",
        "- **Inference**: Work-level permutation test",
        "",
    ]

    for name, analysis in results['analyses'].items():
        cv = analysis['cross_validation']
        perm = analysis['permutation']
        boot = analysis['bootstrap']
        corpus = analysis['corpus_stats']
        n_authors = len(corpus['by_author'])
        chance = 100.0 / n_authors

        lines.extend([
            f"## {analysis['analysis_name']}",
            "",
            "### Corpus",
            "",
            f"- **Works**: {corpus['total_works']}",
            f"- **Words**: {corpus['total_words']:,}",
            f"- **Blocks**: {corpus['total_blocks']}",
            f"- **Authors**: {n_authors}",
            "",
            "| Author | Works | Words | Blocks |",
            "|--------|-------|-------|--------|",
        ])

        for author, stats in sorted(corpus['by_author'].items()):
            lines.append(f"| {author} | {stats['works']} | {stats['words']:,} | {stats['blocks']} |")

        lines.extend([
            "",
            "### Results",
            "",
            f"- **Work-weighted balanced accuracy**: {cv['work_weighted_balanced_accuracy']:.1%}",
            f"- **Chance level**: {chance:.0f}%",
            f"- **Permutation p-value**: {perm['p_value']:.4f}",
            f"- **Bootstrap 95% CI**: [{boot['ci_95_lower']:.1%}, {boot['ci_95_upper']:.1%}]",
            "",
        ])

        acc = cv['work_weighted_balanced_accuracy']
        p = perm['p_value']

        if acc > 0.50 and p < 0.05:
            strength = "STRONG"
        elif acc > (1/n_authors) + 0.15 and p < 0.05:
            strength = "MODERATE"
        elif p < 0.05:
            strength = "WEAK"
        else:
            strength = "NONE"

        lines.extend([
            f"**Signal strength**: {strength}",
            "",
        ])

    # Interpretation
    novels = results['analyses'].get('novels_only', {})
    novels_acc = novels.get('cross_validation', {}).get('work_weighted_balanced_accuracy', 0)
    novels_p = novels.get('permutation', {}).get('p_value', 1)

    lines.extend([
        "## Interpretation for BoM Analysis",
        "",
    ])

    if novels_acc > 0.40 and novels_p < 0.05:
        lines.extend([
            "**Authorial signal SURVIVES translation through a single translator.**",
            "",
            "Since stylometry can recover author identity through Garnett's translation,",
            "the BoM null result (inability to distinguish narrators) is **informative**.",
            "",
            "This suggests the Book of Mormon narrators do not exhibit distinguishable",
            "function-word patterns, consistent with:",
            "- H1: Single modern author",
            "- H3: Single author mimicking multiple voices",
            "",
        ])
    else:
        lines.extend([
            "**Authorial signal does NOT reliably survive translation.**",
            "",
            "The BoM null result remains **ambiguous** - translation homogenization",
            "could explain the lack of narrator-specific patterns even if multiple",
            "ancient authors existed.",
            "",
        ])

    lines.append(f"*Generated: {results['metadata']['generated_at']}*")

    return '\n'.join(lines)


if __name__ == "__main__":
    main()
