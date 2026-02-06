#!/usr/bin/env python3
"""
Phase 2.A: Robustness Testing - OPTIMIZED VERSION

Key optimizations over v1.6.0:
1. Pre-compute feature matrices ONCE per variant (not per permutation)
2. Keep n-gram matrices SPARSE (no .toarray())
3. Parallelize permutation loop with joblib
4. Use StandardScaler(with_mean=False) for sparse compatibility

Expected speedup: 10-50x depending on hardware

Version: 2.0.0
Date: 2026-02-05
"""

import json
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import warnings

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import HashingVectorizer
from joblib import Parallel, delayed
import os

# Prevent nested parallelism issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Configuration
INPUT_FILE = Path("data/text/processed/bom-voice-blocks.json")
OUTPUT_FILE = Path("results/robustness-results.json")
REPORT_FILE = Path("results/robustness-report.md")
CHECKPOINT_FILE = Path("results/robustness-checkpoint-v2.json")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MAX_BLOCKS_PER_RUN = 20
N_PERMUTATIONS = 10000
CHECKPOINT_INTERVAL = 500  # Save more frequently
N_JOBS = -1  # Use all available cores

VOICES = ["MORMON", "NEPHI", "MORONI", "JACOB"]

# Function words list (same as original)
FUNCTION_WORDS = [
    "a", "an", "the",
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "ye", "thee", "thou", "thy", "thine", "thyself",
    "who", "whom", "whose", "which", "what", "that",
    "this", "these", "that", "those",
    "all", "any", "both", "each", "every", "few", "many", "more", "most",
    "much", "neither", "none", "one", "other", "several", "some", "such",
    "and", "but", "or", "nor", "for", "yet", "so",
    "if", "then", "because", "although", "though", "while", "unless",
    "when", "where", "before", "after", "since", "until", "as",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "above", "below", "to", "from", "up", "down",
    "out", "off", "over", "under", "again", "further",
    "of", "upon", "unto", "forth", "among", "concerning", "according",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "not", "no", "never", "neither", "nothing", "nobody", "nowhere",
    "very", "too", "also", "only", "just", "even", "still", "already",
    "now", "here", "there", "therefore", "thus", "hence", "wherefore",
    "how", "why", "when", "where", "whence", "whither",
    "hath", "doth", "art", "wast", "wert", "shalt", "wilt",
    "behold", "yea", "nay", "verily", "amen",
    "insomuch", "inasmuch", "notwithstanding", "nevertheless", "moreover",
    "exceedingly", "wherefore", "thereof", "wherein", "whereby", "whatsoever",
    "whosoever", "whomsoever", "wheresoever", "whithersoever", "howsoever"
]

WORD_TO_IDX = {w: i for i, w in enumerate(FUNCTION_WORDS)}


def save_checkpoint(checkpoint_data: dict):
    """Save checkpoint to disk."""
    checkpoint_data["checkpoint_time"] = datetime.now(timezone.utc).isoformat()
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"  [Checkpoint saved: {checkpoint_data.get('stage', 'unknown')}]")


def load_checkpoint() -> dict | None:
    """Load checkpoint if exists."""
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        print(f"  [Checkpoint found: {checkpoint.get('stage', 'unknown')}]")
        return checkpoint
    except (json.JSONDecodeError, IOError) as e:
        print(f"  [Warning: Could not load checkpoint: {e}]")
        return None


def extract_function_word_features(text: str) -> np.ndarray:
    """Extract function word frequency features."""
    words = text.lower().split()
    total = len(words)
    if total == 0:
        return np.zeros(len(FUNCTION_WORDS))

    counts = np.zeros(len(FUNCTION_WORDS))
    for word in words:
        word_clean = word.strip('.,;:!?"\'-')
        if word_clean in WORD_TO_IDX:
            counts[WORD_TO_IDX[word_clean]] += 1

    return (counts / total) * 1000  # Per 1000 words


def normalize_text_for_ngrams(text: str) -> str:
    """Normalize text for character n-gram extraction."""
    import re
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def precompute_block_indices(blocks: list, target_size: int, include_quotes: bool,
                              max_blocks_per_run: int, seed: int) -> dict:
    """Pre-compute which blocks to use per run (deterministic capping)."""
    if include_quotes:
        filtered = [
            (i, b) for i, b in enumerate(blocks)
            if b["target_size"] == target_size and b["voice"] in VOICES
        ]
    else:
        filtered = [
            (i, b) for i, b in enumerate(blocks)
            if b["target_size"] == target_size
            and b["quote_status"] == "original"
            and b["voice"] in VOICES
        ]

    runs_dict = defaultdict(list)
    for orig_idx, block in filtered:
        runs_dict[block["run_id"]].append(orig_idx)

    rng = np.random.RandomState(seed)
    selected_indices = {}

    for run_id, indices in sorted(runs_dict.items()):
        if max_blocks_per_run and len(indices) > max_blocks_per_run:
            selected = rng.choice(indices, max_blocks_per_run, replace=False)
            selected_indices[run_id] = sorted(selected.tolist())
        else:
            selected_indices[run_id] = sorted(indices)

    return selected_indices


class PrecomputedVariantData:
    """
    Pre-computed feature matrices for a single variant.

    This is the KEY OPTIMIZATION: compute features ONCE, then only permute labels.
    """
    def __init__(self, blocks: list, precomputed_indices: dict, feature_type: str):
        self.feature_type = feature_type
        self.run_ids = []
        self.run_voices = {}  # Original (true) labels
        self.voice_runs = defaultdict(list)

        # Build block data
        all_fw_features = []
        all_texts = []
        self.groups = []  # run_id for each block
        self.y_true = []  # true voice label for each block

        for run_id, indices in sorted(precomputed_indices.items()):
            if not indices:
                continue

            voice = blocks[indices[0]]["voice"]
            self.run_ids.append(run_id)
            self.run_voices[run_id] = voice
            self.voice_runs[voice].append(run_id)

            for idx in indices:
                block = blocks[idx]
                self.groups.append(run_id)
                self.y_true.append(voice)

                if feature_type in ["fw", "combined"]:
                    all_fw_features.append(extract_function_word_features(block["text"]))

                if feature_type in ["ng", "combined"]:
                    all_texts.append(normalize_text_for_ngrams(block["text"]))

        self.groups = np.array(self.groups)
        self.y_true = np.array(self.y_true)
        self.n_runs = len(self.run_ids)
        self.n_blocks = len(self.groups)

        # Build feature matrices (ONCE)
        self.X_fw = None
        self.X_ng = None  # Keep sparse!

        if feature_type in ["fw", "combined"]:
            self.X_fw = np.array(all_fw_features)

        if feature_type in ["ng", "combined"]:
            vectorizer = HashingVectorizer(
                analyzer='char',
                ngram_range=(3, 3),
                n_features=2**14,
                norm='l1',
                alternate_sign=False
            )
            self.X_ng = vectorizer.transform(all_texts)  # SPARSE! No .toarray()

        # Determine feature count
        if feature_type == "fw":
            self.n_features = self.X_fw.shape[1]
        elif feature_type == "ng":
            self.n_features = self.X_ng.shape[1]
        else:  # combined
            self.n_features = self.X_fw.shape[1] + self.X_ng.shape[1]

    def get_permuted_labels(self, perm_mapping: dict) -> np.ndarray:
        """Get label array with permuted run->voice mapping."""
        return np.array([perm_mapping[run_id] for run_id in self.groups])


def run_cv_with_labels(variant_data: PrecomputedVariantData, y: np.ndarray,
                       classifier_type: str) -> float:
    """
    Run leave-one-run-out CV with given labels.

    Uses pre-computed features - only labels change between permutations.
    """
    unique_runs = variant_data.run_ids

    y_true_by_run = defaultdict(list)
    y_pred_by_run = defaultdict(list)

    # Build classifier
    if classifier_type == "logreg":
        clf = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver='saga'  # Works well with sparse
        )
    else:  # svm
        clf = LinearSVC(
            class_weight='balanced',
            C=1.0,
            max_iter=2000,
            random_state=RANDOM_SEED,
            dual='auto'
        )

    feature_type = variant_data.feature_type

    for held_out_run in unique_runs:
        train_mask = variant_data.groups != held_out_run
        test_mask = variant_data.groups == held_out_run

        y_train, y_test = y[train_mask], y[test_mask]

        if feature_type == "fw":
            # Dense FW features - standard scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(variant_data.X_fw[train_mask])
            X_test = scaler.transform(variant_data.X_fw[test_mask])

        elif feature_type == "ng":
            # Sparse n-gram features - MaxAbs scaling (keeps sparse)
            scaler = MaxAbsScaler()
            X_train = scaler.fit_transform(variant_data.X_ng[train_mask])
            X_test = scaler.transform(variant_data.X_ng[test_mask])

        else:  # combined
            # Scale separately then concatenate
            fw_scaler = StandardScaler()
            fw_train = fw_scaler.fit_transform(variant_data.X_fw[train_mask])
            fw_test = fw_scaler.transform(variant_data.X_fw[test_mask])

            ng_scaler = MaxAbsScaler()
            ng_train = ng_scaler.fit_transform(variant_data.X_ng[train_mask])
            ng_test = ng_scaler.transform(variant_data.X_ng[test_mask])

            # Convert sparse to dense for concatenation (unavoidable for combined)
            X_train = np.hstack([fw_train, ng_train.toarray()])
            X_test = np.hstack([fw_test, ng_test.toarray()])

        # Train and predict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        # Get the true labels for this run (from permuted y, not original)
        y_true_by_run[held_out_run] = y_test.tolist()
        y_pred_by_run[held_out_run] = y_pred.tolist()

    # Compute run-weighted balanced accuracy
    voice_run_accuracies = defaultdict(list)

    for run_id in unique_runs:
        true_labels = np.array(y_true_by_run[run_id])
        pred_labels = np.array(y_pred_by_run[run_id])
        run_acc = np.mean(true_labels == pred_labels)
        # Use the permuted voice for this run
        voice = y[variant_data.groups == run_id][0]
        voice_run_accuracies[voice].append(run_acc)

    class_means = []
    for voice in sorted(voice_run_accuracies.keys()):
        if voice_run_accuracies[voice]:
            class_means.append(np.mean(voice_run_accuracies[voice]))

    return np.mean(class_means) if class_means else 0.0


def sample_restricted_permutations(voice_runs: dict, n_samples: int, seed: int) -> list:
    """Sample restricted permutations preserving class run-counts."""
    run_ids = []
    voices_list = []
    for voice, runs in sorted(voice_runs.items()):
        for run_id in runs:
            run_ids.append(run_id)
            voices_list.append(voice)

    permutations_list = []
    seen = set()
    rng = np.random.RandomState(seed)

    attempts = 0
    max_attempts = n_samples * 20

    while len(permutations_list) < n_samples and attempts < max_attempts:
        shuffled = voices_list.copy()
        rng.shuffle(shuffled)

        perm_tuple = tuple(shuffled)
        if perm_tuple not in seen:
            seen.add(perm_tuple)
            permutations_list.append(dict(zip(run_ids, shuffled)))
        attempts += 1

    return permutations_list


def run_single_permutation(perm_idx: int, perm_mapping: dict,
                           variant_data_dict: dict, maxt_variants: set) -> dict:
    """
    Run a single permutation across all maxT variants.

    This function is called in parallel.
    Returns dict with scores or None if failed.
    """
    variant_scores = {}

    for vid, vdata in variant_data_dict.items():
        if vid not in maxt_variants:
            continue

        # Check if permutation covers all runs in this variant
        if not all(run_id in perm_mapping for run_id in vdata.run_ids):
            return None  # Permutation doesn't apply

        try:
            y_permuted = vdata.get_permuted_labels(perm_mapping)
            score = run_cv_with_labels(vdata, y_permuted,
                                       "svm" if vid == "A6" else "logreg")
            variant_scores[vid] = score
        except Exception as e:
            return None  # Mark permutation as failed

    if not variant_scores:
        return None

    return variant_scores


def main():
    parser = argparse.ArgumentParser(description="Run robustness testing (optimized)")
    parser.add_argument("--permutations", "-p", type=int, default=N_PERMUTATIONS,
                        help=f"Number of permutations (default: {N_PERMUTATIONS})")
    parser.add_argument("--jobs", "-j", type=int, default=N_JOBS,
                        help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Start fresh, ignore existing checkpoint")
    args = parser.parse_args()

    n_perms = args.permutations
    n_jobs = args.jobs

    print("=" * 70)
    print("PHASE 2.A: ROBUSTNESS TESTING (OPTIMIZED v2.0)")
    print("=" * 70)
    print(f"Configuration: {n_perms} permutations, {n_jobs} parallel jobs")

    # Load data
    print(f"\nLoading blocks from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    blocks = data["blocks"]
    print(f"  Loaded {len(blocks)} total blocks")

    # Check for checkpoint
    checkpoint = None if args.no_checkpoint else load_checkpoint()

    # Define variants
    variants = {
        "A1": {"description": "Block size 500", "target_size": 500,
               "include_quotes": False, "feature_type": "fw", "classifier": "logreg"},
        "A2": {"description": "Block size 2000", "target_size": 2000,
               "include_quotes": False, "feature_type": "fw", "classifier": "logreg"},
        "A3": {"description": "Include quotations", "target_size": 1000,
               "include_quotes": True, "feature_type": "fw", "classifier": "logreg"},
        "A4": {"description": "Character 3-grams", "target_size": 1000,
               "include_quotes": False, "feature_type": "ng", "classifier": "logreg"},
        "A5": {"description": "FW + char 3-grams", "target_size": 1000,
               "include_quotes": False, "feature_type": "combined", "classifier": "logreg"},
        "A6": {"description": "SVM classifier", "target_size": 1000,
               "include_quotes": False, "feature_type": "fw", "classifier": "svm"},
    }

    # Pre-compute indices
    print("\nPre-computing block indices...")
    variant_indices = {}
    for vid, vconfig in variants.items():
        indices = precompute_block_indices(
            blocks, vconfig["target_size"], vconfig["include_quotes"],
            MAX_BLOCKS_PER_RUN,
            RANDOM_SEED + int(hashlib.sha256(vid.encode()).hexdigest()[:8], 16) % 1000
        )
        variant_indices[vid] = indices
        print(f"  {vid}: {len(indices)} runs, {sum(len(v) for v in indices.values())} blocks")

    # Pre-compute feature matrices (KEY OPTIMIZATION)
    print("\nPre-computing feature matrices (this is the key optimization)...")
    variant_data = {}
    observed_scores = {}
    master_voice_runs = None

    for vid, vconfig in variants.items():
        print(f"  {vid}: {vconfig['description']}...", end=" ", flush=True)

        vdata = PrecomputedVariantData(blocks, variant_indices[vid], vconfig["feature_type"])
        variant_data[vid] = vdata

        # Compute observed (true) accuracy
        score = run_cv_with_labels(vdata, vdata.y_true, vconfig["classifier"])
        observed_scores[vid] = score

        print(f"accuracy={score:.1%}, runs={vdata.n_runs}, features={vdata.n_features}")

        if master_voice_runs is None and vdata.n_runs == 14:
            master_voice_runs = dict(vdata.voice_runs)

    # Define maxT family (exclude A3 - different run count)
    maxt_variants = {"A1", "A2", "A4", "A5", "A6"}
    excluded_variants = {"A3"}

    print(f"\nMaxT family: {sorted(maxt_variants)}")
    print(f"Excluded (different run count): {sorted(excluded_variants)}")

    # Generate permutations
    print(f"\nGenerating {n_perms} permutations...")
    perms = sample_restricted_permutations(master_voice_runs, n_perms, RANDOM_SEED)
    print(f"  Generated {len(perms)} unique permutations")

    # Run permutation test IN PARALLEL
    print(f"\nRunning permutation test with {n_jobs} parallel jobs...")
    print("  (Progress will be shown every 500 permutations)")

    # Process in batches for checkpointing
    batch_size = CHECKPOINT_INTERVAL
    all_results = []
    n_failed = 0

    start_idx = 0
    if checkpoint and checkpoint.get("stage") == "permutations_v2":
        start_idx = checkpoint.get("completed_perms", 0)
        all_results = checkpoint.get("results", [])
        n_failed = checkpoint.get("n_failed", 0)
        print(f"  Resuming from permutation {start_idx}...")

    for batch_start in range(start_idx, len(perms), batch_size):
        batch_end = min(batch_start + batch_size, len(perms))
        batch_perms = perms[batch_start:batch_end]

        # Run batch in parallel
        batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(run_single_permutation)(i, perm, variant_data, maxt_variants)
            for i, perm in enumerate(batch_perms, start=batch_start)
        )

        # Process results
        for result in batch_results:
            if result is None:
                n_failed += 1
            else:
                all_results.append(result)

        print(f"    Completed {batch_end}/{len(perms)} "
              f"({len(all_results)} successful, {n_failed} failed)")

        # Checkpoint
        save_checkpoint({
            "stage": "permutations_v2",
            "completed_perms": batch_end,
            "results": all_results,
            "n_failed": n_failed,
            "observed_scores": observed_scores,
        })

    print(f"\n  Completed: {len(all_results)} successful, {n_failed} failed permutations")

    # Compute p-values
    observed_max = max(observed_scores[vid] for vid in maxt_variants)
    perm_max_scores = np.array([max(r[vid] for vid in maxt_variants if vid in r)
                                 for r in all_results])

    n_exceed = np.sum(perm_max_scores >= observed_max)
    corrected_p = (1 + n_exceed) / (1 + len(perm_max_scores))

    # Per-variant p-values
    uncorrected_p = {}
    adjusted_p = {}
    for vid in maxt_variants:
        vid_scores = np.array([r[vid] for r in all_results if vid in r])
        if len(vid_scores) > 0:
            n_exceed_v = np.sum(vid_scores >= observed_scores[vid])
            uncorrected_p[vid] = (1 + n_exceed_v) / (1 + len(vid_scores))

            n_exceed_adj = np.sum(perm_max_scores >= observed_scores[vid])
            adjusted_p[vid] = (1 + n_exceed_adj) / (1 + len(perm_max_scores))

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nCorrected p-value (maxT): {corrected_p:.4f}")
    print(f"Observed max accuracy: {observed_max:.1%}")
    if len(perm_max_scores) > 0:
        print(f"Null distribution: mean={np.mean(perm_max_scores):.1%}, "
              f"95th percentile={np.percentile(perm_max_scores, 95):.1%}")

    print("\nPer-variant results:")
    for vid in sorted(variants.keys()):
        acc = observed_scores[vid]
        if vid in maxt_variants:
            up = uncorrected_p.get(vid, None)
            ap = adjusted_p.get(vid, None)
            up_str = f"{up:.4f}" if up else "N/A"
            ap_str = f"{ap:.4f}" if ap else "N/A"
            print(f"  {vid}: {acc:.1%} (uncorr p={up_str}, adj p={ap_str})")
        else:
            print(f"  {vid}: {acc:.1%} [excluded from maxT]")

    # Save results
    results = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script_version": "2.0.0-optimized",
            "random_seed": RANDOM_SEED,
            "n_permutations": n_perms,
            "n_successful": len(all_results),
            "n_failed": n_failed,
        },
        "observed_scores": observed_scores,
        "permutation": {
            "corrected_p_value": corrected_p,
            "uncorrected_p_values": uncorrected_p,
            "adjusted_p_values": adjusted_p,
            "observed_max": observed_max,
            "null_max_mean": float(np.mean(perm_max_scores)) if len(perm_max_scores) > 0 else None,
            "null_max_std": float(np.std(perm_max_scores)) if len(perm_max_scores) > 0 else None,
            "maxt_variants": sorted(maxt_variants),
            "excluded_variants": sorted(excluded_variants),
        }
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Robustness conclusion
    is_robust = corrected_p >= 0.05
    print(f"\n{'='*70}")
    print(f"CONCLUSION: Null result is {'ROBUST' if is_robust else 'NOT ROBUST'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
