#!/usr/bin/env python3
"""
Phase 2.D: Translation-Layer Calibration (Garnett Corpus) - OPTIMIZED

This is the optimized version for AWS deployment with parallelization.

Optimizations:
1. Pre-compute all feature matrices once
2. Parallelize permutation test with joblib
3. Parallelize bootstrap with joblib
4. Checkpoint saving for long runs

Pre-Registered Confound Controls (Section 6.3):
5. Character/place name masking before feature extraction
6. Translation period stratification (early/mid/late Garnett career)

Version: 2.1.0
Date: 2026-02-05
"""

import argparse
import json
import re
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# Configuration
GARNETT_DIR = Path("data/reference/garnett/raw")
MANIFEST_FILE = GARNETT_DIR / "manifest.json"
OUTPUT_FILE = Path("results/garnett-analysis-results.json")
REPORT_FILE = Path("results/garnett-analysis-report.md")
CHECKPOINT_FILE = Path("results/garnett-checkpoint.json")

RANDOM_SEED = 42
BLOCK_SIZE = 1000
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 1000

# Function words for modern English
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
    "than", "whether", "either", "once",
    "however", "moreover", "nevertheless", "meanwhile",
    "perhaps", "quite", "rather", "almost", "always", "often", "sometimes",
]

WORD_TO_IDX = {w: i for i, w in enumerate(FUNCTION_WORDS)}

# ============================================================================
# PRE-REGISTERED CONFOUND CONTROL 1: Character/Place Name Masking (Section 6.3)
# ============================================================================

# Common Russian character names from Garnett translations
# These are masked to prevent content leakage into stylometric analysis
RUSSIAN_NAMES = {
    # Dostoevsky - Crime and Punishment
    "raskolnikov", "rodion", "rodya", "razumihin", "razumikhin", "dounia",
    "dunya", "pulcheria", "sonia", "sonya", "marmeladov", "porfiry", "luzhin",
    "svidrigailov", "lebezyatnikov", "katerina", "ivanovna",
    # Dostoevsky - Brothers Karamazov
    "karamazov", "fyodor", "dmitri", "mitya", "ivan", "alyosha", "alexey",
    "smerdyakov", "grushenka", "katerina", "zossima", "rakitin", "kolya",
    # Dostoevsky - The Idiot
    "myshkin", "nastasya", "filippovna", "rogozhin", "ganya", "ivolgin",
    "totsky", "yepanchin", "aglaya", "lebedev", "ippolit",
    # Dostoevsky - The Possessed/Demons
    "stavrogin", "nikolay", "verkhovensky", "pyotr", "stepan", "shatov",
    "kirillov", "liza", "lebyadkin", "virginsky", "liputin",
    # Tolstoy - Anna Karenina
    "anna", "karenina", "karenin", "vronsky", "levin", "kitty", "stiva",
    "oblonsky", "dolly", "darya", "sergey", "koznyshev", "varenka",
    # Tolstoy - War and Peace
    "pierre", "bezukhov", "natasha", "rostov", "andrei", "bolkonsky",
    "marya", "nikolai", "helene", "kutuzov", "napoleon", "anatole",
    "denisov", "dolokhov", "petya", "sonya", "boris",
    # Chekhov - common characters
    "gurov", "ionych", "startsev", "ranevskaya", "lopakhin", "trofimov",
    # Turgenev - Fathers and Sons
    "bazarov", "arkady", "kirsanov", "nikolai", "pavel", "odintsova",
    "fenichka", "sitnikov", "kukshina",
    # Turgenev - other works
    "lavretsky", "liza", "kalitina", "insarov", "elena", "litvinov",
    "nezhdanov", "marianna", "sipyagin",
    # Common Russian names/patronymics
    "ivan", "pyotr", "pavel", "sergei", "mikhail", "andrei", "nikolai",
    "alexei", "dmitri", "vladimir", "boris", "konstantin", "yevgeny",
    "alexandr", "alexander", "vasily", "grigory", "ilya", "fyodor",
    "petrovich", "ivanovich", "nikolaevich", "alexandrovich", "pavlovich",
    "mikhailovich", "sergeyevich", "dmitrievich", "fyodorovich",
}

# Common Russian place names
RUSSIAN_PLACES = {
    "moscow", "petersburg", "russia", "russian", "siberia", "siberian",
    "ukraine", "caucasus", "crimea", "volga", "neva", "nevsky",
    "arbat", "tverskaya", "kremlin", "skotoprigonyevsk",  # fictional town
}

ALL_NAMES_TO_MASK = RUSSIAN_NAMES | RUSSIAN_PLACES


def mask_names(text: str) -> str:
    """Mask character and place names to prevent content leakage.

    Pre-registered confound control (Section 6.3):
    'Mask character names and place names before feature extraction'
    """
    words = text.split()
    masked_words = []
    for word in words:
        word_lower = re.sub(r'[^a-z]', '', word.lower())
        if word_lower in ALL_NAMES_TO_MASK:
            # Replace with placeholder that won't affect function word counts
            masked_words.append("[NAME]")
        else:
            masked_words.append(word)
    return ' '.join(masked_words)


# ============================================================================
# PRE-REGISTERED CONFOUND CONTROL 2: Translation Period (Section 6.3)
# ============================================================================

# Garnett's translation periods based on publication year
# Early: 1894-1904, Late: 1912-1918
def get_translation_period(publication_year: int) -> str:
    """Categorize work by Garnett's translation period.

    Pre-registered confound control (Section 6.3):
    'Report results stratified by early/mid/late Garnett career'

    Periods:
    - Early: 1894-1904 (Turgenev, early Tolstoy)
    - Late: 1912-1918 (Dostoevsky, Chekhov, late Tolstoy)
    """
    if publication_year <= 1904:
        return "early"
    else:
        return "late"


def clean_text(text: str) -> str:
    """Clean and normalize text for analysis."""
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
            next_para = text.find("\n\n", idx)
            if next_para != -1:
                text = text[next_para:]

    for marker in end_markers:
        if marker in text:
            idx = text.find(marker)
            text = text[:idx]

    return re.sub(r'\s+', ' ', text).strip()


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

    return (counts / total) * 1000


def create_blocks(text: str, block_size: int = BLOCK_SIZE) -> list:
    """Split text into non-overlapping blocks."""
    words = text.split()
    blocks = []
    for i in range(0, len(words), block_size):
        block_words = words[i:i + block_size]
        if len(block_words) >= block_size * 0.8:
            blocks.append(' '.join(block_words))
    return blocks


def load_garnett_corpus(novels_only: bool = False, period_filter: str = None) -> dict:
    """Load all Garnett translations.

    Args:
        novels_only: If True, exclude short story collections (Chekhov)
        period_filter: If set to 'early' or 'late', filter by translation period
    """
    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    corpus = {}
    for work_info in manifest:
        author = work_info['author']
        title = work_info['title']
        filename = work_info['filename']
        genre = work_info.get('genre', 'unknown')
        publication_year = work_info.get('publication_year', 1900)

        # Genre filter
        if novels_only and genre == 'stories':
            continue

        # Period filter (Pre-registered confound control)
        period = get_translation_period(publication_year)
        if period_filter and period != period_filter:
            continue

        filepath = GARNETT_DIR / filename
        if not filepath.exists():
            print(f"  Warning: {filename} not found, skipping")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Clean text (remove Gutenberg headers/footers)
        clean = clean_text(raw_text)

        # Apply name masking (Pre-registered confound control Section 6.3)
        masked = mask_names(clean)

        word_count = len(masked.split())

        if author not in corpus:
            corpus[author] = []

        corpus[author].append({
            'title': title,
            'text': masked,  # Use masked text for analysis
            'word_count': word_count,
            'filename': filename,
            'genre': genre,
            'publication_year': publication_year,
            'translation_period': period,
        })

    return corpus


def build_dataset(corpus: dict) -> tuple:
    """Build feature matrix and labels from corpus."""
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


def run_single_cv(X: np.ndarray, y: np.ndarray, work_ids: np.ndarray,
                  sample_weights: np.ndarray = None) -> dict:
    """Run leave-one-work-out CV and return results."""
    unique_works = np.unique(work_ids)

    y_true_all = []
    y_pred_all = []
    work_results = {}
    skipped_folds = []

    for held_out_work in unique_works:
        train_mask = work_ids != held_out_work
        test_mask = work_ids == held_out_work

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Check if training set has at least 2 classes
        unique_train_classes = np.unique(y_train)
        if len(unique_train_classes) < 2:
            # Skip this fold - not enough classes for classification
            skipped_folds.append({
                'work_id': int(held_out_work),
                'reason': f'Training set has only 1 class: {unique_train_classes[0]}',
                'test_author': y_test[0]
            })
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

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

        work_acc = np.mean(y_test == y_pred)
        work_author = y_test[0]
        work_results[int(held_out_work)] = {
            'author': work_author,
            'accuracy': float(work_acc),
            'n_blocks': int(len(y_test))
        }

    # Report skipped folds
    if skipped_folds:
        print(f"  WARNING: Skipped {len(skipped_folds)} folds due to single-class training sets")
        for sf in skipped_folds:
            print(f"    - Work {sf['work_id']} ({sf['test_author']}): {sf['reason']}")

    # Check if we have enough valid folds
    if len(work_results) == 0:
        raise ValueError("No valid CV folds - all folds had single-class training sets")

    # Compute work-weighted balanced accuracy
    author_work_accs = defaultdict(list)
    for work_id, result in work_results.items():
        author_work_accs[result['author']].append(result['accuracy'])

    class_means = [np.mean(accs) for accs in author_work_accs.values()]
    work_weighted_balanced_acc = np.mean(class_means)
    block_balanced_acc = balanced_accuracy_score(y_true_all, y_pred_all)

    return {
        'work_weighted_balanced_accuracy': float(work_weighted_balanced_acc),
        'block_balanced_accuracy': float(block_balanced_acc),
        'work_results': work_results,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'author_work_accuracies': {k: [float(v) for v in vals]
                                    for k, vals in author_work_accs.items()},
        'skipped_folds': skipped_folds,
        'n_valid_folds': len(work_results),
        'n_total_folds': len(unique_works)
    }


def run_single_permutation(X: np.ndarray, y: np.ndarray, work_ids: np.ndarray,
                           sample_weights: np.ndarray, work_to_author: dict,
                           unique_works: list, perm_seed: int) -> float:
    """Run a single permutation and return the score."""
    rng = np.random.RandomState(perm_seed)

    # Permute work-level labels
    work_authors = [work_to_author[w] for w in unique_works]
    permuted_authors = work_authors.copy()
    rng.shuffle(permuted_authors)

    # Create new block-level labels
    work_to_perm_author = dict(zip(unique_works, permuted_authors))
    y_permuted = np.array([work_to_perm_author[w] for w in work_ids])

    # Run CV with permuted labels (include sample_weights for consistency)
    try:
        result = run_single_cv(X, y_permuted, work_ids, sample_weights)
        return result['work_weighted_balanced_accuracy']
    except ValueError:
        # If all folds failed due to single-class training sets, return NaN
        return np.nan


def run_permutation_test_parallel(X: np.ndarray, y: np.ndarray,
                                   work_ids: np.ndarray, sample_weights: np.ndarray,
                                   observed_score: float,
                                   n_permutations: int, n_jobs: int = -1) -> dict:
    """Run permutation test with parallel processing."""
    # Get work-level labels
    work_to_author = {}
    for i, (work_id, author) in enumerate(zip(work_ids, y)):
        work_to_author[work_id] = author

    unique_works = list(work_to_author.keys())

    print(f"  Running {n_permutations} permutations with {n_jobs} jobs...")

    # Run permutations in parallel
    null_scores = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_permutation)(
            X, y, work_ids, sample_weights, work_to_author, unique_works, RANDOM_SEED + i
        )
        for i in range(n_permutations)
    )

    null_scores = np.array(null_scores)

    # Filter out NaN values (from failed permutations)
    valid_scores = null_scores[~np.isnan(null_scores)]
    n_failed = np.sum(np.isnan(null_scores))

    if n_failed > 0:
        print(f"  WARNING: {n_failed} permutations failed due to single-class folds")

    if len(valid_scores) == 0:
        raise ValueError("All permutations failed - analysis not viable")

    p_value = (1 + np.sum(valid_scores >= observed_score)) / (1 + len(valid_scores))

    return {
        'observed_score': float(observed_score),
        'p_value': float(p_value),
        'null_mean': float(np.mean(valid_scores)),
        'null_std': float(np.std(valid_scores)),
        'null_95_percentile': float(np.percentile(valid_scores, 95)),
        'n_permutations': n_permutations,
        'n_valid_permutations': len(valid_scores),
        'n_failed_permutations': int(n_failed)
    }


def run_single_bootstrap(X: np.ndarray, y: np.ndarray, work_ids: np.ndarray,
                          sample_weights: np.ndarray, author_works: dict,
                          unique_works: list, boot_seed: int) -> float:
    """Run a single bootstrap sample and return the score."""
    rng = np.random.RandomState(boot_seed)

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
    weights_boot = sample_weights[boot_mask] if sample_weights is not None else None

    try:
        result = run_single_cv(X_boot, y_boot, work_ids_boot, weights_boot)
        return result['work_weighted_balanced_accuracy']
    except ValueError:
        # If all folds failed, return NaN
        return np.nan


def run_bootstrap_ci_parallel(X: np.ndarray, y: np.ndarray,
                               work_ids: np.ndarray, sample_weights: np.ndarray,
                               n_bootstrap: int, n_jobs: int = -1) -> dict:
    """Compute bootstrap CI with parallel processing."""
    unique_works = np.unique(work_ids)
    work_to_author = {}
    for i, (work_id, author) in enumerate(zip(work_ids, y)):
        work_to_author[work_id] = author

    author_works = defaultdict(list)
    for work_id in unique_works:
        author_works[work_to_author[work_id]].append(work_id)
    author_works = dict(author_works)

    print(f"  Running {n_bootstrap} bootstrap samples with {n_jobs} jobs...")

    bootstrap_scores = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_bootstrap)(
            X, y, work_ids, sample_weights, author_works, unique_works, RANDOM_SEED + 50000 + i
        )
        for i in range(n_bootstrap)
    )

    bootstrap_scores = np.array(bootstrap_scores)

    # Filter out NaN values (from failed bootstraps)
    valid_scores = bootstrap_scores[~np.isnan(bootstrap_scores)]
    n_failed = np.sum(np.isnan(bootstrap_scores))

    if n_failed > 0:
        print(f"  WARNING: {n_failed} bootstrap samples failed due to single-class folds")

    if len(valid_scores) == 0:
        raise ValueError("All bootstrap samples failed - analysis not viable")

    return {
        'mean': float(np.mean(valid_scores)),
        'std': float(np.std(valid_scores)),
        'ci_95_lower': float(np.percentile(valid_scores, 2.5)),
        'ci_95_upper': float(np.percentile(valid_scores, 97.5)),
        'n_bootstrap': n_bootstrap,
        'n_valid_bootstrap': len(valid_scores),
        'n_failed_bootstrap': int(n_failed)
    }


def run_analysis(corpus: dict, analysis_name: str,
                 n_permutations: int, n_bootstrap: int, n_jobs: int) -> dict:
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

    # Run cross-validation
    print("\nRunning leave-one-work-out cross-validation...")
    cv_results = run_single_cv(X, y, work_ids, sample_weights)
    print(f"  Work-weighted balanced accuracy: {cv_results['work_weighted_balanced_accuracy']:.1%}")
    print(f"  Block-level balanced accuracy: {cv_results['block_balanced_accuracy']:.1%}")

    # Permutation test
    print("\nRunning permutation test...")
    perm_results = run_permutation_test_parallel(
        X, y, work_ids, sample_weights,
        cv_results['work_weighted_balanced_accuracy'],
        n_permutations,
        n_jobs
    )
    print(f"  p-value: {perm_results['p_value']:.4f}")

    # Bootstrap CI
    print("\nRunning bootstrap confidence interval...")
    boot_results = run_bootstrap_ci_parallel(X, y, work_ids, sample_weights, n_bootstrap, n_jobs)
    print(f"  95% CI: [{boot_results['ci_95_lower']:.1%}, {boot_results['ci_95_upper']:.1%}]")

    return {
        'analysis_name': analysis_name,
        'corpus_stats': corpus_stats,
        'work_metadata': {str(k): v for k, v in work_metadata.items()},
        'cross_validation': cv_results,
        'permutation': perm_results,
        'bootstrap': boot_results,
    }


def generate_report(results: dict) -> str:
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
        # Handle skipped analyses
        if analysis.get('skipped'):
            lines.extend([
                f"## {name}",
                "",
                f"**SKIPPED**: {analysis.get('reason', 'unknown reason')}",
                "",
            ])
            if 'note' in analysis:
                lines.append(f"_{analysis['note']}_")
                lines.append("")
            continue

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

        # Note skipped folds if any
        skipped_folds = cv.get('skipped_folds', [])
        if skipped_folds:
            lines.extend([
                f"_Note: {len(skipped_folds)} CV folds skipped due to single-class training sets_",
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


def main():
    parser = argparse.ArgumentParser(description="Run Garnett translation-layer calibration (optimized)")
    parser.add_argument("--permutations", "-p", type=int, default=N_PERMUTATIONS,
                        help=f"Number of permutations (default: {N_PERMUTATIONS})")
    parser.add_argument("--bootstrap", "-b", type=int, default=N_BOOTSTRAP,
                        help=f"Number of bootstrap samples (default: {N_BOOTSTRAP})")
    parser.add_argument("--jobs", "-j", type=int, default=-1,
                        help="Number of parallel jobs (default: -1 = all cores)")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 2.D: TRANSLATION-LAYER CALIBRATION (GARNETT CORPUS)")
    print("OPTIMIZED VERSION")
    print("=" * 70)
    print(f"Configuration: {args.permutations} permutations, {args.bootstrap} bootstrap, {args.jobs} jobs")

    results = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'script_version': '2.0.0',
            'random_seed': RANDOM_SEED,
            'block_size': BLOCK_SIZE,
            'n_features': len(FUNCTION_WORDS),
            'n_permutations': args.permutations,
            'n_bootstrap': args.bootstrap,
            'methodology_notes': [
                'Block weighting: each work contributes equally regardless of length',
                'Leave-one-work-out CV prevents train/test leakage',
                'Work-level permutation respects non-independence of blocks',
                'Parallelized with joblib for efficiency',
                'CONFOUND CONTROL: Character/place names masked before feature extraction (Section 6.3)',
                'CONFOUND CONTROL: Period stratification (early/late Garnett career) (Section 6.3)',
            ]
        },
        'analyses': {}
    }

    # PRIMARY ANALYSIS: Novels only
    print("\n" + "=" * 70)
    print("PRIMARY ANALYSIS: NOVELS ONLY")
    print("(Excludes Chekhov story collections to control for genre)")
    print("=" * 70)

    corpus_novels = load_garnett_corpus(novels_only=True)
    novels_results = run_analysis(
        corpus_novels, "Novels Only (Primary)",
        args.permutations, args.bootstrap, args.jobs
    )
    results['analyses']['novels_only'] = novels_results

    # Save checkpoint
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nCheckpoint saved to {CHECKPOINT_FILE}")

    # SECONDARY ANALYSIS: Full corpus
    print("\n" + "=" * 70)
    print("SECONDARY ANALYSIS: FULL CORPUS")
    print("(All works including Chekhov story collections)")
    print("=" * 70)

    corpus_full = load_garnett_corpus(novels_only=False)
    full_results = run_analysis(
        corpus_full, "Full Corpus (Secondary)",
        args.permutations, args.bootstrap, args.jobs
    )
    results['analyses']['full_corpus'] = full_results

    # Save checkpoint
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nCheckpoint saved to {CHECKPOINT_FILE}")

    # ========================================================================
    # PERIOD-STRATIFIED ANALYSES (Pre-registered confound control Section 6.3)
    # ========================================================================

    # EARLY PERIOD: 1894-1904 (Turgenev, early Tolstoy)
    print("\n" + "=" * 70)
    print("PERIOD STRATIFICATION: EARLY (1894-1904)")
    print("(Turgenev and early Tolstoy translations)")
    print("=" * 70)

    corpus_early = load_garnett_corpus(novels_only=False, period_filter='early')
    if len(corpus_early) >= 2:  # Need at least 2 authors
        try:
            early_results = run_analysis(
                corpus_early, "Early Period (1894-1904)",
                args.permutations, args.bootstrap, args.jobs
            )
            results['analyses']['period_early'] = early_results
        except ValueError as e:
            print(f"  Analysis failed: {e}")
            print("  NOTE: Early period has only 2 authors; leave-one-work-out CV")
            print("        can create single-class training sets when one author has few works.")
            results['analyses']['period_early'] = {
                'skipped': True,
                'reason': 'cv_single_class_folds',
                'error': str(e),
                'note': 'Insufficient works per author for robust leave-one-work-out CV'
            }
    else:
        print("  Skipping: insufficient authors in early period")
        results['analyses']['period_early'] = {'skipped': True, 'reason': 'insufficient_authors'}

    # LATE PERIOD: 1912-1918 (Dostoevsky, Chekhov, late Tolstoy)
    print("\n" + "=" * 70)
    print("PERIOD STRATIFICATION: LATE (1912-1918)")
    print("(Dostoevsky, Chekhov, and late Tolstoy translations)")
    print("=" * 70)

    corpus_late = load_garnett_corpus(novels_only=False, period_filter='late')
    if len(corpus_late) >= 2:  # Need at least 2 authors
        try:
            late_results = run_analysis(
                corpus_late, "Late Period (1912-1918)",
                args.permutations, args.bootstrap, args.jobs
            )
            results['analyses']['period_late'] = late_results
        except ValueError as e:
            print(f"  Analysis failed: {e}")
            results['analyses']['period_late'] = {
                'skipped': True,
                'reason': 'cv_single_class_folds',
                'error': str(e)
            }
    else:
        print("  Skipping: insufficient authors in late period")
        results['analyses']['period_late'] = {'skipped': True, 'reason': 'insufficient_authors'}

    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Generate report
    report = generate_report(results)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {REPORT_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, analysis in results['analyses'].items():
        # Handle skipped analyses
        if analysis.get('skipped'):
            print(f"\n{name}:")
            print(f"  SKIPPED: {analysis.get('reason', 'unknown reason')}")
            if 'note' in analysis:
                print(f"  Note: {analysis['note']}")
            continue

        acc = analysis['cross_validation']['work_weighted_balanced_accuracy']
        p = analysis['permutation']['p_value']
        n_works = analysis['corpus_stats']['total_works']
        n_authors = len(analysis['corpus_stats']['by_author'])

        print(f"\n{analysis['analysis_name']}:")
        print(f"  Works: {n_works}, Authors: {n_authors}")
        print(f"  Accuracy: {acc:.1%} (chance = {100/n_authors:.0f}%)")
        print(f"  p-value: {p:.4f}")

        # Report skipped folds if any
        skipped_folds = analysis['cross_validation'].get('skipped_folds', [])
        if skipped_folds:
            print(f"  Note: {len(skipped_folds)} CV folds skipped (single-class training)")

        chance = 1.0 / n_authors
        if acc > 0.50 and p < 0.05:
            print(f"  → STRONG signal detected")
        elif acc > chance + 0.15 and p < 0.05:
            print(f"  → MODERATE signal detected")
        elif p < 0.05:
            print(f"  → WEAK signal detected")
        else:
            print(f"  → NO significant signal")


if __name__ == "__main__":
    main()
