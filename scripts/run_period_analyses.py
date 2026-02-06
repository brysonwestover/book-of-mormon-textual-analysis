#!/usr/bin/env python3
"""
Run period-stratified analyses for Garnett corpus.

Per Amendment #2, bootstrap CIs are NOT ESTIMABLE for period analyses
due to Tolstoy having only 2 works in each period (49% failure rate).

This script runs:
- Point estimates (LOWO CV accuracy)
- Permutation p-values (feasible, 0% failure rate)
- NO bootstrap CIs

Version: 1.0.0
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
from sklearn.preprocessing import StandardScaler

# Import from main script
import sys
sys.path.insert(0, str(Path(__file__).parent))

from run_garnett_analysis_optimized import (
    GARNETT_DIR, MANIFEST_FILE, FUNCTION_WORDS, WORD_TO_IDX,
    RANDOM_SEED, BLOCK_SIZE,
    mask_names, create_blocks, extract_features, build_dataset,
    load_garnett_corpus, get_translation_period
)

OUTPUT_FILE = Path("results/period-analysis-results.json")

def run_single_cv_safe(X, y, work_ids, sample_weights=None):
    """Run LOWO CV with safe handling of single-class folds."""
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

        # Check for single-class training
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            skipped_folds.append({
                'work_id': int(held_out_work),
                'reason': f'Single class in training: {unique_classes[0]}'
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
                clf.fit(X_train_scaled, y_train, sample_weight=sample_weights[train_mask])
            else:
                clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        work_acc = np.mean(y_test == y_pred)
        work_results[int(held_out_work)] = {
            'author': y_test[0],
            'accuracy': float(work_acc),
            'n_blocks': int(len(y_test))
        }

    if len(work_results) == 0:
        raise ValueError("All CV folds failed")

    # Compute work-weighted balanced accuracy
    author_work_accs = defaultdict(list)
    for work_id, result in work_results.items():
        author_work_accs[result['author']].append(result['accuracy'])

    class_means = [np.mean(accs) for accs in author_work_accs.values()]
    work_weighted_balanced_acc = np.mean(class_means)

    return {
        'work_weighted_balanced_accuracy': float(work_weighted_balanced_acc),
        'work_results': work_results,
        'skipped_folds': skipped_folds,
        'n_valid_folds': len(work_results),
        'n_total_folds': len(unique_works)
    }


def run_single_permutation(X, y, work_ids, sample_weights, work_to_author, unique_works, seed):
    """Run single permutation."""
    rng = np.random.RandomState(seed)

    work_authors = [work_to_author[w] for w in unique_works]
    permuted = work_authors.copy()
    rng.shuffle(permuted)

    work_to_perm = dict(zip(unique_works, permuted))
    y_perm = np.array([work_to_perm[w] for w in work_ids])

    try:
        result = run_single_cv_safe(X, y_perm, work_ids, sample_weights)
        return result['work_weighted_balanced_accuracy']
    except ValueError:
        return np.nan


def run_permutation_test(X, y, work_ids, sample_weights, observed_score, n_perms, n_jobs):
    """Run permutation test."""
    work_to_author = {}
    for work_id, author in zip(work_ids, y):
        work_to_author[work_id] = author
    unique_works = list(work_to_author.keys())

    print(f"  Running {n_perms} permutations...")

    null_scores = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_single_permutation)(
            X, y, work_ids, sample_weights, work_to_author, unique_works, RANDOM_SEED + i
        )
        for i in range(n_perms)
    )

    null_scores = np.array(null_scores)
    valid = null_scores[~np.isnan(null_scores)]
    n_failed = np.sum(np.isnan(null_scores))

    if n_failed > 0:
        print(f"  WARNING: {n_failed} permutations failed")

    p_value = (1 + np.sum(valid >= observed_score)) / (1 + len(valid))

    return {
        'observed_score': float(observed_score),
        'p_value': float(p_value),
        'null_mean': float(np.mean(valid)),
        'null_std': float(np.std(valid)),
        'n_permutations': n_perms,
        'n_valid': len(valid),
        'n_failed': int(n_failed)
    }


def run_period_analysis(period_name, period_filter, n_perms, n_jobs):
    """Run analysis for a single period."""
    print(f"\n{'='*60}")
    print(f"PERIOD: {period_name}")
    print(f"{'='*60}")

    corpus = load_garnett_corpus(novels_only=False, period_filter=period_filter)

    if len(corpus) < 2:
        return {'skipped': True, 'reason': 'insufficient_authors'}

    # Print corpus stats
    total_works = sum(len(works) for works in corpus.values())
    print(f"Authors: {list(corpus.keys())}")
    for author, works in corpus.items():
        print(f"  {author}: {len(works)} works")
    print(f"Total: {total_works} works")

    # Build dataset
    print("\nBuilding feature matrix...")
    X, y, work_ids, sample_weights, work_metadata = build_dataset(corpus)
    print(f"  {X.shape[0]} blocks, {X.shape[1]} features")

    # Run CV
    print("\nRunning LOWO cross-validation...")
    try:
        cv_results = run_single_cv_safe(X, y, work_ids, sample_weights)
        acc = cv_results['work_weighted_balanced_accuracy']
        print(f"  Work-weighted balanced accuracy: {acc:.1%}")

        if cv_results['skipped_folds']:
            print(f"  Skipped folds: {len(cv_results['skipped_folds'])}")
    except ValueError as e:
        print(f"  CV FAILED: {e}")
        return {'skipped': True, 'reason': 'cv_failed', 'error': str(e)}

    # Run permutation test
    print("\nRunning permutation test...")
    perm_results = run_permutation_test(
        X, y, work_ids, sample_weights, acc, n_perms, n_jobs
    )
    print(f"  p-value: {perm_results['p_value']:.4f}")

    # Compute corpus stats
    corpus_stats = {
        'total_works': total_works,
        'by_author': {author: len(works) for author, works in corpus.items()}
    }

    return {
        'period': period_name,
        'corpus_stats': corpus_stats,
        'cross_validation': cv_results,
        'permutation': perm_results,
        'bootstrap': {
            'status': 'NOT_ESTIMABLE',
            'reason': 'High failure rate (30-49%) due to Tolstoy having only 2 works',
            'reference': 'OSF Amendment #2, Deviation 6'
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--permutations', type=int, default=10000)
    parser.add_argument('--jobs', type=int, default=-1)
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 2.D: PERIOD-STRATIFIED ANALYSES")
    print("(Bootstrap CI not estimable - see Amendment #2)")
    print("=" * 70)

    results = {
        'metadata': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'n_permutations': args.permutations,
            'note': 'Bootstrap CI not estimable due to Amendment #2 feasibility criterion'
        },
        'analyses': {}
    }

    # Early period
    early_results = run_period_analysis(
        "Early Period (1894-1904)", "early", args.permutations, args.jobs
    )
    results['analyses']['period_early'] = early_results

    # Late period
    late_results = run_period_analysis(
        "Late Period (1912-1918)", "late", args.permutations, args.jobs
    )
    results['analyses']['period_late'] = late_results

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, analysis in results['analyses'].items():
        if analysis.get('skipped'):
            print(f"\n{name}: SKIPPED - {analysis.get('reason')}")
            continue

        acc = analysis['cross_validation']['work_weighted_balanced_accuracy']
        p = analysis['permutation']['p_value']
        n_authors = len(analysis['corpus_stats']['by_author'])
        chance = 1.0 / n_authors

        print(f"\n{analysis['period']}:")
        print(f"  Accuracy: {acc:.1%} (chance = {chance:.0%})")
        print(f"  p-value: {p:.4f}")
        print(f"  Bootstrap CI: NOT ESTIMABLE (Amendment #2)")

        if acc > chance + 0.10 and p < 0.05:
            print(f"  → Signal detected above chance")
        else:
            print(f"  → No significant signal")


if __name__ == "__main__":
    main()
