#!/usr/bin/env python3
"""
Corrected stylometric classification (v3) with proper group-level inference.

Implements GPT-5.2 Pro methodology corrections (Phase 2.0):
1. Run-weighted balanced accuracy (effective N = number of runs, not blocks)
2. Restricted group-level permutation (preserving class run-counts)
3. Cap blocks per run to prevent training domination
4. Bootstrap CI at run level
5. One-sided p-value for above-chance classification

Key insight: With only 14 runs and severe imbalance, we must treat runs as the
unit of analysis, not blocks. The 244 blocks are pseudoreplicated.

See: docs/decisions/phase2-execution-plan.md

Version: 3.0.0
Date: 2026-02-03
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict
from itertools import permutations
import re

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Configuration
INPUT_FILE = Path("data/text/processed/bom-voice-blocks.json")
OUTPUT_FILE = Path("results/classification-results-v3.json")
REPORT_FILE = Path("results/classification-report-v3.md")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MAX_BLOCKS_PER_RUN = 20  # Cap to prevent run_0015 domination
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 100000  # Monte Carlo permutations (exact has ~2.5M)

# Function words (same as v2)
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


def extract_function_word_features(text: str) -> dict:
    """Extract function word frequencies (content-suppressed)."""
    tokens = tokenize(text)
    total_words = len(tokens)

    if total_words == 0:
        return {f"fw_{w}": 0.0 for w in FUNCTION_WORDS}

    word_counts = Counter(tokens)
    features = {}
    for word in FUNCTION_WORDS:
        count = word_counts.get(word, 0)
        features[f"fw_{word}"] = (count / total_words) * 1000

    return features


def build_run_data(blocks: list, target_size: int = 1000,
                   voices: list = None, max_blocks_per_run: int = None) -> dict:
    """
    Build data structure organized by runs.

    Returns dict with:
        runs: list of run dicts, each containing:
            - run_id
            - voice
            - blocks: list of block dicts with features
        run_voices: dict mapping run_id -> voice
        voice_runs: dict mapping voice -> list of run_ids
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

    # Build run data with optional capping
    runs = []
    run_voices = {}
    voice_runs = defaultdict(list)

    for run_id, run_blocks in sorted(runs_dict.items()):
        voice = run_blocks[0]["voice"]

        # Cap blocks per run if specified
        if max_blocks_per_run and len(run_blocks) > max_blocks_per_run:
            # Sample deterministically
            np.random.seed(hash(run_id) % (2**32))
            indices = np.random.choice(len(run_blocks), max_blocks_per_run, replace=False)
            run_blocks = [run_blocks[i] for i in sorted(indices)]

        # Extract features for each block
        block_data = []
        for block in run_blocks:
            features = extract_function_word_features(block["text"])
            block_data.append({
                "block_id": block["block_id"],
                "features": features,
                "text_length": block["word_count"]
            })

        runs.append({
            "run_id": run_id,
            "voice": voice,
            "blocks": block_data
        })

        run_voices[run_id] = voice
        voice_runs[voice].append(run_id)

    return {
        "runs": runs,
        "run_voices": run_voices,
        "voice_runs": dict(voice_runs),
        "feature_names": sorted(extract_function_word_features("").keys())
    }


def runs_to_matrices(runs: list, feature_names: list) -> tuple:
    """Convert run data to X, y, groups matrices."""
    X_list = []
    y_list = []
    groups_list = []

    feat_to_idx = {name: i for i, name in enumerate(feature_names)}
    n_features = len(feature_names)

    for run in runs:
        for block in run["blocks"]:
            row = np.zeros(n_features)
            for feat_name, value in block["features"].items():
                if feat_name in feat_to_idx:
                    row[feat_to_idx[feat_name]] = value
            X_list.append(row)
            y_list.append(run["voice"])
            groups_list.append(run["run_id"])

    return np.array(X_list), np.array(y_list), np.array(groups_list)


def compute_run_weighted_balanced_accuracy(y_true_by_run: dict, y_pred_by_run: dict,
                                            run_voices: dict) -> float:
    """
    Compute run-weighted balanced accuracy.

    1. For each run, compute run-level accuracy (proportion correct)
    2. For each class, average run-level accuracies across runs of that class
    3. Average across classes (balanced)

    This treats runs as the unit of analysis, not blocks.
    """
    # Group runs by true voice
    voice_run_accuracies = defaultdict(list)

    for run_id in y_true_by_run:
        true_labels = y_true_by_run[run_id]
        pred_labels = y_pred_by_run[run_id]

        # Run-level accuracy
        run_acc = np.mean(np.array(true_labels) == np.array(pred_labels))

        # Group by voice
        voice = run_voices[run_id]
        voice_run_accuracies[voice].append(run_acc)

    # Balanced accuracy: average of per-class mean accuracies
    class_means = []
    for voice in sorted(voice_run_accuracies.keys()):
        class_means.append(np.mean(voice_run_accuracies[voice]))

    return np.mean(class_means)


def leave_one_run_out_cv(runs: list, feature_names: list) -> dict:
    """
    Leave-one-run-out cross-validation with run-weighted metrics.

    Returns predictions organized by run for proper metric computation.
    """
    n_runs = len(runs)

    # Storage for predictions
    y_true_by_run = {}
    y_pred_by_run = {}
    run_voices = {r["run_id"]: r["voice"] for r in runs}

    # Build full matrices once
    X_full, y_full, groups_full = runs_to_matrices(runs, feature_names)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver='lbfgs'
        ))
    ])

    for i, held_out_run in enumerate(runs):
        held_out_id = held_out_run["run_id"]

        # Split
        train_mask = groups_full != held_out_id
        test_mask = groups_full == held_out_id

        X_train, y_train = X_full[train_mask], y_full[train_mask]
        X_test, y_test = X_full[test_mask], y_full[test_mask]

        # Train and predict
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Store
        y_true_by_run[held_out_id] = y_test.tolist()
        y_pred_by_run[held_out_id] = y_pred.tolist()

    # Compute run-weighted balanced accuracy
    rwba = compute_run_weighted_balanced_accuracy(y_true_by_run, y_pred_by_run, run_voices)

    # Also compute block-level metrics for comparison
    all_true = []
    all_pred = []
    for run_id in y_true_by_run:
        all_true.extend(y_true_by_run[run_id])
        all_pred.extend(y_pred_by_run[run_id])

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Per-run accuracy
    run_accuracies = {}
    for run_id in y_true_by_run:
        true = np.array(y_true_by_run[run_id])
        pred = np.array(y_pred_by_run[run_id])
        run_accuracies[run_id] = float(np.mean(true == pred))

    return {
        "run_weighted_balanced_accuracy": rwba,
        "block_level_accuracy": float(np.mean(all_true == all_pred)),
        "y_true_by_run": y_true_by_run,
        "y_pred_by_run": y_pred_by_run,
        "run_voices": run_voices,
        "run_accuracies": run_accuracies
    }


def generate_restricted_permutations(voice_runs: dict, n_samples: int = None) -> list:
    """
    Generate restricted permutations that preserve class run-counts.

    We permute which voice label each run gets, but maintain the same
    number of runs per class (e.g., 4 MORMON runs, 5 NEPHI runs, etc.)

    Total valid permutations = 14! / (4! * 5! * 2! * 3!) = 2,522,520

    If n_samples is None, attempts exact enumeration.
    Otherwise, uses Monte Carlo sampling.
    """
    # Get run counts per voice
    run_counts = {voice: len(runs) for voice, runs in voice_runs.items()}
    all_runs = []
    for voice, runs in voice_runs.items():
        all_runs.extend(runs)

    n_runs = len(all_runs)
    voices = sorted(voice_runs.keys())

    # Calculate total permutations
    from math import factorial
    total_perms = factorial(n_runs)
    for count in run_counts.values():
        total_perms //= factorial(count)

    print(f"  Total valid permutations: {total_perms:,}")

    if n_samples is None and total_perms <= 100000:
        # Exact enumeration
        print(f"  Using exact enumeration...")
        return generate_all_restricted_permutations(all_runs, voice_runs)
    else:
        # Monte Carlo
        n = n_samples or N_PERMUTATIONS
        print(f"  Using Monte Carlo with {n:,} samples...")
        return sample_restricted_permutations(all_runs, voice_runs, n)


def generate_all_restricted_permutations(all_runs: list, voice_runs: dict) -> list:
    """Generate all valid permutations (for small enough space)."""
    # This is complex - use sampling instead for now
    # Exact enumeration requires generating all ways to assign runs to classes
    # while preserving counts
    raise NotImplementedError("Use Monte Carlo sampling for this case")


def sample_restricted_permutations(all_runs: list, voice_runs: dict,
                                    n_samples: int) -> list:
    """
    Sample restricted permutations via random shuffling.

    Each permutation is a mapping from run_id -> voice that preserves counts.
    """
    # Build the base assignment
    base_assignment = []
    for voice, runs in sorted(voice_runs.items()):
        for run_id in runs:
            base_assignment.append((run_id, voice))

    # The voices list (preserving counts)
    voices_list = [voice for _, voice in base_assignment]
    run_ids = [run_id for run_id, _ in base_assignment]

    permutations_list = []
    seen = set()

    np.random.seed(RANDOM_SEED)

    attempts = 0
    max_attempts = n_samples * 10

    while len(permutations_list) < n_samples and attempts < max_attempts:
        # Shuffle voices
        shuffled_voices = voices_list.copy()
        np.random.shuffle(shuffled_voices)

        # Create mapping
        perm = tuple(zip(run_ids, shuffled_voices))
        perm_key = tuple(sorted(perm))

        if perm_key not in seen:
            seen.add(perm_key)
            permutations_list.append(dict(perm))

        attempts += 1

    print(f"  Generated {len(permutations_list):,} unique permutations")
    return permutations_list


def run_permutation_test(runs: list, feature_names: list,
                          observed_score: float, voice_runs: dict,
                          n_permutations: int = N_PERMUTATIONS) -> dict:
    """
    Run restricted group-level permutation test.

    Permutes voice labels at the run level while preserving class counts.
    Tests whether observed score is significantly above chance.
    """
    print(f"  Running permutation test...")

    # Generate permutations
    perms = sample_restricted_permutations(
        [r["run_id"] for r in runs],
        voice_runs,
        n_permutations
    )

    # Build matrices once
    X_full, _, groups_full = runs_to_matrices(runs, feature_names)

    perm_scores = []

    for i, perm_mapping in enumerate(perms):
        if (i + 1) % 10000 == 0:
            print(f"    Permutation {i+1:,}/{len(perms):,}...")

        # Create permuted runs
        perm_runs = []
        for run in runs:
            perm_run = run.copy()
            perm_run["voice"] = perm_mapping[run["run_id"]]
            perm_runs.append(perm_run)

        # Compute permuted score using same CV
        try:
            perm_result = leave_one_run_out_cv(perm_runs, feature_names)
            perm_scores.append(perm_result["run_weighted_balanced_accuracy"])
        except Exception as e:
            continue

    perm_scores = np.array(perm_scores)

    # One-sided p-value: proportion of permutations >= observed
    p_value = np.mean(perm_scores >= observed_score)

    return {
        "observed_score": observed_score,
        "p_value": p_value,
        "null_mean": float(np.mean(perm_scores)),
        "null_std": float(np.std(perm_scores)),
        "null_min": float(np.min(perm_scores)),
        "null_max": float(np.max(perm_scores)),
        "null_5_percentile": float(np.percentile(perm_scores, 5)),
        "null_95_percentile": float(np.percentile(perm_scores, 95)),
        "n_permutations": len(perm_scores),
        "perm_scores": perm_scores.tolist()  # For diagnostics
    }


def run_bootstrap_ci(runs: list, feature_names: list,
                      n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """
    Bootstrap confidence interval at run level.

    Resamples runs with replacement (stratified by voice if possible).
    """
    print(f"  Running bootstrap ({n_bootstrap} resamples)...")

    voice_runs = defaultdict(list)
    for run in runs:
        voice_runs[run["voice"]].append(run)

    bootstrap_scores = []
    np.random.seed(RANDOM_SEED)

    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"    Bootstrap {i+1}/{n_bootstrap}...")

        # Stratified resample: sample with replacement within each voice
        resampled_runs = []
        for voice, voice_run_list in voice_runs.items():
            n_runs_voice = len(voice_run_list)
            indices = np.random.choice(n_runs_voice, n_runs_voice, replace=True)
            for idx in indices:
                # Create new run with unique ID to avoid CV issues
                new_run = voice_run_list[idx].copy()
                new_run["run_id"] = f"{new_run['run_id']}_bs{i}_{idx}"
                resampled_runs.append(new_run)

        try:
            result = leave_one_run_out_cv(resampled_runs, feature_names)
            bootstrap_scores.append(result["run_weighted_balanced_accuracy"])
        except Exception:
            continue

    bootstrap_scores = np.array(bootstrap_scores)

    return {
        "mean": float(np.mean(bootstrap_scores)),
        "std": float(np.std(bootstrap_scores)),
        "ci_95_lower": float(np.percentile(bootstrap_scores, 2.5)),
        "ci_95_upper": float(np.percentile(bootstrap_scores, 97.5)),
        "n_bootstrap": len(bootstrap_scores)
    }


def generate_report_v3(results: dict, output_path: Path):
    """Generate corrected methodology report."""
    lines = [
        "# Stylometric Classification Results (v3 - Run-Level Inference)",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "---",
        "",
        "## Methodology Corrections (Phase 2.0)",
        "",
        "This analysis addresses critical issues identified in GPT-5.2 Pro review:",
        "",
        "1. **Run-weighted metrics:** Effective N = 14 runs, not 244 blocks",
        "2. **Group-level permutation:** Permute voice labels at run level",
        "3. **Restricted permutations:** Preserve class run-counts (4/5/2/3)",
        "4. **Capped blocks per run:** Max 20 to prevent run_0015 domination",
        "5. **Bootstrap CI:** Stratified resampling at run level",
        "",
        "---",
        "",
        "## Data Summary",
        "",
        f"- **Total runs:** {results['n_runs']}",
        f"- **Total blocks:** {results['n_blocks']} (before capping)",
        f"- **Blocks after capping:** {results['n_blocks_capped']} (max {MAX_BLOCKS_PER_RUN}/run)",
        f"- **Features:** {results['n_features']}",
        "",
        "### Run Distribution by Voice",
        "",
        "| Voice | Runs | Blocks (orig) | Blocks (capped) |",
        "|-------|------|---------------|-----------------|",
    ]

    for voice in sorted(results["run_distribution"].keys()):
        dist = results["run_distribution"][voice]
        lines.append(f"| {voice} | {dist['n_runs']} | {dist['n_blocks_orig']} | {dist['n_blocks_capped']} |")

    lines.extend([
        "",
        "---",
        "",
        "## Primary Results",
        "",
        "### Run-Weighted Balanced Accuracy",
        "",
        f"- **Observed:** {results['primary']['run_weighted_balanced_accuracy']:.1%}",
        f"- **Chance baseline:** 25.0% (4 classes)",
        f"- **vs Chance:** {(results['primary']['run_weighted_balanced_accuracy'] - 0.25) * 100:+.1f} percentage points",
        "",
        "### Bootstrap Confidence Interval",
        "",
        f"- **95% CI:** [{results['bootstrap']['ci_95_lower']:.1%}, {results['bootstrap']['ci_95_upper']:.1%}]",
        f"- **Bootstrap mean:** {results['bootstrap']['mean']:.1%}",
        "",
        "### Permutation Test (Group-Level)",
        "",
        f"- **Observed score:** {results['permutation']['observed_score']:.3f}",
        f"- **Null distribution:** {results['permutation']['null_mean']:.3f} ± {results['permutation']['null_std']:.3f}",
        f"- **Null range:** [{results['permutation']['null_min']:.3f}, {results['permutation']['null_max']:.3f}]",
        f"- **p-value (one-sided):** {results['permutation']['p_value']:.4f}",
        f"- **Permutations:** {results['permutation']['n_permutations']:,}",
        "",
    ])

    if results['permutation']['p_value'] < 0.05:
        lines.append("**Interpretation:** Classification performance is **statistically significant** (p < 0.05).")
        lines.append("There is evidence of stylistic differentiation between narrators.")
    else:
        lines.append("**Interpretation:** Classification performance is **NOT statistically significant** (p ≥ 0.05).")
        lines.append("There is insufficient evidence of stylistic differentiation using function words.")

    lines.extend([
        "",
        "---",
        "",
        "## Per-Run Performance",
        "",
        "| Run ID | Voice | Blocks | Accuracy |",
        "|--------|-------|--------|----------|",
    ])

    for run_id in sorted(results['primary']['run_accuracies'].keys()):
        acc = results['primary']['run_accuracies'][run_id]
        voice = results['primary']['run_voices'][run_id]
        n_blocks = len(results['primary']['y_true_by_run'][run_id])
        lines.append(f"| {run_id} | {voice} | {n_blocks} | {acc:.1%} |")

    lines.extend([
        "",
        "---",
        "",
        "## Comparison with v2 Results",
        "",
        "| Metric | v2 (Block-Level) | v3 (Run-Level) |",
        "|--------|------------------|----------------|",
        f"| Balanced Accuracy | 21.6% | {results['primary']['run_weighted_balanced_accuracy']:.1%} |",
        f"| Permutation p-value | 1.0 (INVALID) | {results['permutation']['p_value']:.4f} |",
        f"| Null distribution std | ~0 (BUG) | {results['permutation']['null_std']:.4f} |",
        "",
        "---",
        "",
        "## Key Structural Insight",
        "",
        "The effective sample size is **14 runs**, not 244 blocks.",
        "",
        "Run_0015 alone contains 146 of 244 blocks (60%). Any analysis that treats",
        "blocks as independent observations is pseudoreplicated. The v2 permutation",
        "test failed because it permuted at the wrong level.",
        "",
        "---",
        "",
        "## Implications for Phase 2",
        "",
    ])

    if results['permutation']['p_value'] < 0.05:
        lines.extend([
            "**Signal detected.** Proceed to investigate source:",
            "- Is it genuine narrator style?",
            "- Is it genre/topic confound?",
            "- Is it an artifact of the grouping structure?",
        ])
    else:
        lines.extend([
            "**No significant signal.** Proceed to:",
            "- Phase 2.A: Robustness testing (confirm null is stable)",
            "- Phase 2.D: Garnett calibration (context for translation layer)",
        ])

    lines.extend([
        "",
        "---",
        "",
        "*Methodology per GPT-5.2 Pro consultation. See docs/decisions/phase2-execution-plan.md*",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    print("=" * 70)
    print("STYLOMETRIC CLASSIFICATION v3 (RUN-LEVEL INFERENCE)")
    print("=" * 70)
    print()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    print(f"Loading blocks from {INPUT_FILE}...")
    data = load_blocks(INPUT_FILE)
    blocks = data["blocks"]
    print(f"  Loaded {len(blocks)} total blocks")

    # Build run-level data
    print()
    print("Building run-level data structure...")
    print(f"  Capping at {MAX_BLOCKS_PER_RUN} blocks per run")

    # First without capping to get original counts
    run_data_orig = build_run_data(blocks, max_blocks_per_run=None)

    # Then with capping
    run_data = build_run_data(blocks, max_blocks_per_run=MAX_BLOCKS_PER_RUN)

    runs = run_data["runs"]
    voice_runs = run_data["voice_runs"]
    feature_names = run_data["feature_names"]

    n_runs = len(runs)
    n_blocks_orig = sum(len(r["blocks"]) for r in run_data_orig["runs"])
    n_blocks_capped = sum(len(r["blocks"]) for r in runs)

    print(f"  Runs: {n_runs}")
    print(f"  Blocks (original): {n_blocks_orig}")
    print(f"  Blocks (capped): {n_blocks_capped}")
    print(f"  Features: {len(feature_names)}")

    # Run distribution
    run_distribution = {}
    for voice in sorted(voice_runs.keys()):
        voice_run_ids = voice_runs[voice]
        n_blocks_voice_orig = sum(
            len(r["blocks"]) for r in run_data_orig["runs"]
            if r["run_id"] in voice_run_ids
        )
        n_blocks_voice_capped = sum(
            len(r["blocks"]) for r in runs
            if r["run_id"] in voice_run_ids
        )
        run_distribution[voice] = {
            "n_runs": len(voice_run_ids),
            "n_blocks_orig": n_blocks_voice_orig,
            "n_blocks_capped": n_blocks_voice_capped
        }
        print(f"  {voice}: {len(voice_run_ids)} runs, {n_blocks_voice_orig} blocks (capped: {n_blocks_voice_capped})")

    # Primary classification with leave-one-run-out CV
    print()
    print("Running leave-one-run-out cross-validation...")
    primary_results = leave_one_run_out_cv(runs, feature_names)

    print(f"  Run-weighted balanced accuracy: {primary_results['run_weighted_balanced_accuracy']:.1%}")
    print(f"  Block-level accuracy (for comparison): {primary_results['block_level_accuracy']:.1%}")

    # Bootstrap CI
    print()
    bootstrap_results = run_bootstrap_ci(runs, feature_names, n_bootstrap=N_BOOTSTRAP)
    print(f"  Bootstrap 95% CI: [{bootstrap_results['ci_95_lower']:.1%}, {bootstrap_results['ci_95_upper']:.1%}]")

    # Permutation test
    print()
    perm_results = run_permutation_test(
        runs, feature_names,
        primary_results["run_weighted_balanced_accuracy"],
        voice_runs,
        n_permutations=N_PERMUTATIONS
    )
    print(f"  Null distribution: {perm_results['null_mean']:.3f} ± {perm_results['null_std']:.3f}")
    print(f"  p-value: {perm_results['p_value']:.4f}")

    # Compile results
    results = {
        "metadata": {
            "source_file": str(INPUT_FILE),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script_version": "3.0.0",
            "methodology": "Run-level inference with restricted group permutation",
            "random_seed": RANDOM_SEED,
            "max_blocks_per_run": MAX_BLOCKS_PER_RUN,
            "n_bootstrap": N_BOOTSTRAP,
            "n_permutations": N_PERMUTATIONS
        },
        "n_runs": n_runs,
        "n_blocks": n_blocks_orig,
        "n_blocks_capped": n_blocks_capped,
        "n_features": len(feature_names),
        "run_distribution": run_distribution,
        "primary": primary_results,
        "bootstrap": bootstrap_results,
        "permutation": {k: v for k, v in perm_results.items() if k != "perm_scores"}  # Exclude large array
    }

    # Save
    print()
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Generating report at {REPORT_FILE}...")
    generate_report_v3(results, REPORT_FILE)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Run-weighted balanced accuracy: {primary_results['run_weighted_balanced_accuracy']:.1%}")
    print(f"95% Bootstrap CI: [{bootstrap_results['ci_95_lower']:.1%}, {bootstrap_results['ci_95_upper']:.1%}]")
    print(f"Permutation p-value: {perm_results['p_value']:.4f}")
    print(f"Null distribution: {perm_results['null_mean']:.3f} ± {perm_results['null_std']:.3f}")
    print()

    if perm_results['p_value'] < 0.05:
        print("CONCLUSION: Statistically significant stylistic differentiation detected.")
        print("  → Proceed to investigate source (narrator vs confound)")
    else:
        print("CONCLUSION: No statistically significant stylistic differentiation detected.")
        print("  → Proceed to robustness testing (Phase 2.A)")

    print()
    print(f"Full report: {REPORT_FILE}")

    return results


if __name__ == "__main__":
    main()
