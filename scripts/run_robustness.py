#!/usr/bin/env python3
"""
Phase 2.A: Robustness Testing for Stylometric Classification

Implements 6 pre-registered variants to test stability of Phase 2.0 null result.
Uses max-statistic permutation correction for multiple comparisons.

Pre-registered variants (OSF DOI: 10.17605/OSF.IO/4W3KH):
  A1: Block size 500 words (more samples, noisier)
  A2: Block size 2000 words (cleaner signal, fewer samples)
  A3: Include quotations (test exclusion policy)
  A4: Character 3-grams (alternative feature family)
  A5: Combined FW + char 3-grams (maximum features)
  A6: SVM classifier (alternative algorithm)

Methodology corrections per GPT-5.2 Pro review:
  - HashingVectorizer for char 3-grams (no vocabulary leakage)
  - Pre-computed deterministic block capping (reproducible across permutations)
  - Separate standardization for each feature family in A5
  - Corrected p-value formula: (1 + sum) / (1 + B)
  - LinearSVC with fixed C for A6 (no tuning with small N)
  - Simplified robustness criterion: corrected p >= 0.05

See: docs/decisions/phase2-execution-plan.md

Version: 1.1.0
Date: 2026-02-04
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict
import re

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer

# Configuration
INPUT_FILE = Path("data/text/processed/bom-voice-blocks.json")
OUTPUT_FILE = Path("results/robustness-results.json")
REPORT_FILE = Path("results/robustness-report.md")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MAX_BLOCKS_PER_RUN = 20
N_PERMUTATIONS = 10000  # Sufficient for maxT; will increase if p near threshold
VOICES = ["MORMON", "NEPHI", "MORONI", "JACOB"]

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


def normalize_text_for_ngrams(text: str) -> str:
    """
    Normalize text for character n-gram extraction.

    Locked preprocessing choices per GPT recommendation:
    - Lowercase
    - Normalize whitespace to single spaces
    - Keep spaces (important for stylometry)
    - Remove punctuation except apostrophes
    - Unicode NFKC normalization
    """
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    # Keep letters, spaces, and apostrophes
    text = re.sub(r"[^a-z\s']", '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_function_word_features(text: str) -> np.ndarray:
    """Extract function word frequencies as numpy array."""
    tokens = tokenize(text)
    total_words = len(tokens)

    if total_words == 0:
        return np.zeros(len(FUNCTION_WORDS))

    word_counts = Counter(tokens)
    features = np.zeros(len(FUNCTION_WORDS))
    for i, word in enumerate(FUNCTION_WORDS):
        count = word_counts.get(word, 0)
        features[i] = (count / total_words) * 1000

    return features


def precompute_block_indices(blocks: list, target_size: int, include_quotes: bool,
                              max_blocks_per_run: int, seed: int) -> dict:
    """
    Pre-compute which blocks to use per run (deterministic capping).

    Returns dict mapping run_id -> list of block indices within that run.
    This ensures identical block selection across all permutations.
    """
    # Filter blocks
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

    # Group by run
    runs_dict = defaultdict(list)
    for orig_idx, block in filtered:
        runs_dict[block["run_id"]].append(orig_idx)

    # Deterministic capping
    rng = np.random.RandomState(seed)
    selected_indices = {}

    for run_id, indices in sorted(runs_dict.items()):
        if max_blocks_per_run and len(indices) > max_blocks_per_run:
            # Deterministic random selection
            selected = rng.choice(indices, max_blocks_per_run, replace=False)
            selected_indices[run_id] = sorted(selected.tolist())
        else:
            selected_indices[run_id] = sorted(indices)

    return selected_indices


def build_run_data_from_indices(blocks: list, selected_indices: dict,
                                 feature_type: str = "fw") -> dict:
    """
    Build run data using pre-computed block indices.

    Args:
        blocks: Full blocks list
        selected_indices: Dict mapping run_id -> list of block indices
        feature_type: "fw", "ng", or "combined"

    Returns dict with runs, run_voices, voice_runs, texts (for n-gram extraction)
    """
    runs = []
    run_voices = {}
    voice_runs = defaultdict(list)
    texts_by_run = {}  # For n-gram extraction

    for run_id, indices in sorted(selected_indices.items()):
        if not indices:
            continue

        voice = blocks[indices[0]]["voice"]

        block_data = []
        texts = []

        for idx in indices:
            block = blocks[idx]
            text = block["text"]
            texts.append(text)

            if feature_type in ["fw", "combined"]:
                fw_features = extract_function_word_features(text)
            else:
                fw_features = None

            block_data.append({
                "block_id": block["block_id"],
                "original_index": idx,
                "fw_features": fw_features,
                "text": text,
                "text_normalized": normalize_text_for_ngrams(text)
            })

        runs.append({
            "run_id": run_id,
            "voice": voice,
            "blocks": block_data
        })
        run_voices[run_id] = voice
        voice_runs[voice].append(run_id)
        texts_by_run[run_id] = texts

    return {
        "runs": runs,
        "run_voices": run_voices,
        "voice_runs": dict(voice_runs),
        "texts_by_run": texts_by_run
    }


def create_feature_matrices(run_data: dict, feature_type: str,
                            hashing_vectorizer: HashingVectorizer = None) -> tuple:
    """
    Create X, y, groups matrices from run data.

    For n-grams, uses HashingVectorizer (no leakage, deterministic).
    For combined, keeps FW and n-gram features separate for later scaling.

    Returns: (X, y, groups, feature_info)
    """
    runs = run_data["runs"]

    X_list = []
    y_list = []
    groups_list = []

    # Collect all texts for n-gram extraction
    all_texts = []
    text_indices = []  # Map back to (run_idx, block_idx)

    for run_idx, run in enumerate(runs):
        for block_idx, block in enumerate(run["blocks"]):
            text_indices.append((run_idx, block_idx))
            all_texts.append(block["text_normalized"])
            y_list.append(run["voice"])
            groups_list.append(run["run_id"])

    if feature_type == "fw":
        # Function words only
        for run_idx, block_idx in text_indices:
            X_list.append(runs[run_idx]["blocks"][block_idx]["fw_features"])
        X = np.array(X_list)
        feature_info = {"type": "fw", "n_features": len(FUNCTION_WORDS)}

    elif feature_type == "ng":
        # Character n-grams only (using hashing)
        if hashing_vectorizer is None:
            hashing_vectorizer = HashingVectorizer(
                analyzer='char',
                ngram_range=(3, 3),
                n_features=2**14,  # 16384 features
                norm='l1',
                alternate_sign=False
            )
        X = hashing_vectorizer.transform(all_texts).toarray()
        feature_info = {"type": "ng", "n_features": X.shape[1]}

    elif feature_type == "combined":
        # Both FW and n-grams, kept separate for scaling
        fw_matrix = np.array([
            runs[run_idx]["blocks"][block_idx]["fw_features"]
            for run_idx, block_idx in text_indices
        ])

        if hashing_vectorizer is None:
            hashing_vectorizer = HashingVectorizer(
                analyzer='char',
                ngram_range=(3, 3),
                n_features=2**14,
                norm='l1',
                alternate_sign=False
            )
        ng_matrix = hashing_vectorizer.transform(all_texts).toarray()

        # Return separately for proper scaling
        feature_info = {
            "type": "combined",
            "n_fw": fw_matrix.shape[1],
            "n_ng": ng_matrix.shape[1],
            "n_features": fw_matrix.shape[1] + ng_matrix.shape[1]
        }
        return (fw_matrix, ng_matrix), np.array(y_list), np.array(groups_list), feature_info

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return X, np.array(y_list), np.array(groups_list), feature_info


def compute_run_weighted_balanced_accuracy(y_true_by_run: dict, y_pred_by_run: dict,
                                            run_voices: dict) -> float:
    """
    Compute run-weighted balanced accuracy.

    1. Per run: proportion of blocks correctly classified
    2. Per voice: average of run-level accuracies for runs of that voice
    3. Overall: macro-average across voices (not dominated by voices with more runs)
    """
    voice_run_accuracies = defaultdict(list)

    for run_id in y_true_by_run:
        true_labels = np.array(y_true_by_run[run_id])
        pred_labels = np.array(y_pred_by_run[run_id])
        run_acc = np.mean(true_labels == pred_labels)
        voice = run_voices[run_id]
        voice_run_accuracies[voice].append(run_acc)

    # Macro-average across voices
    class_means = []
    for voice in sorted(voice_run_accuracies.keys()):
        if voice_run_accuracies[voice]:
            class_means.append(np.mean(voice_run_accuracies[voice]))

    return np.mean(class_means) if class_means else 0.0


def leave_one_run_out_cv(run_data: dict, feature_type: str,
                          classifier: str = "logreg",
                          hashing_vectorizer: HashingVectorizer = None) -> dict:
    """
    Leave-one-run-out cross-validation with proper feature handling.

    For combined features (A5): scales FW and n-grams separately per fold.
    """
    runs = run_data["runs"]
    run_voices = run_data["run_voices"]

    if len(runs) < 4:
        return {
            "run_weighted_balanced_accuracy": None,
            "error": "Insufficient runs"
        }

    # Get feature matrices
    features, y, groups, feature_info = create_feature_matrices(
        run_data, feature_type, hashing_vectorizer
    )

    y_true_by_run = defaultdict(list)
    y_pred_by_run = defaultdict(list)

    # Build classifier
    if classifier == "logreg":
        clf = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver='lbfgs'
        )
    elif classifier == "svm":
        # LinearSVC with fixed C per GPT recommendation (no tuning)
        clf = LinearSVC(
            class_weight='balanced',
            C=1.0,  # Fixed, no tuning
            max_iter=2000,
            random_state=RANDOM_SEED
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    unique_runs = list(run_voices.keys())

    for held_out_run in unique_runs:
        train_mask = groups != held_out_run
        test_mask = groups == held_out_run

        y_train, y_test = y[train_mask], y[test_mask]

        if feature_type == "combined":
            # Scale FW and n-grams separately per GPT recommendation
            fw_matrix, ng_matrix = features

            # Scale FW
            fw_scaler = StandardScaler()
            fw_train = fw_scaler.fit_transform(fw_matrix[train_mask])
            fw_test = fw_scaler.transform(fw_matrix[test_mask])

            # Scale n-grams (with_mean=False for sparse-safe, though we densified)
            ng_scaler = StandardScaler(with_mean=False)
            ng_train = ng_scaler.fit_transform(ng_matrix[train_mask])
            ng_test = ng_scaler.transform(ng_matrix[test_mask])

            # Concatenate
            X_train = np.hstack([fw_train, ng_train])
            X_test = np.hstack([fw_test, ng_test])
        else:
            # Standard scaling for single feature type
            scaler = StandardScaler()
            X_train = scaler.fit_transform(features[train_mask])
            X_test = scaler.transform(features[test_mask])

        # Train and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_true_by_run[held_out_run] = y_test.tolist()
        y_pred_by_run[held_out_run] = y_pred.tolist()

    rwba = compute_run_weighted_balanced_accuracy(
        dict(y_true_by_run), dict(y_pred_by_run), run_voices
    )

    return {
        "run_weighted_balanced_accuracy": rwba,
        "y_true_by_run": dict(y_true_by_run),
        "y_pred_by_run": dict(y_pred_by_run),
        "run_voices": run_voices,
        "n_features": feature_info["n_features"]
    }


def sample_restricted_permutations(voice_runs: dict, n_samples: int,
                                    seed: int) -> list:
    """
    Sample restricted permutations preserving class run-counts.

    Returns list of dicts mapping run_id -> permuted_voice.
    """
    # Build base assignment
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


def run_single_variant(blocks: list, variant_config: dict,
                        precomputed_indices: dict = None) -> dict:
    """
    Run a single variant analysis.

    Uses pre-computed block indices for deterministic results.
    """
    # Pre-compute indices if not provided
    if precomputed_indices is None:
        precomputed_indices = precompute_block_indices(
            blocks,
            target_size=variant_config["target_size"],
            include_quotes=variant_config["include_quotes"],
            max_blocks_per_run=MAX_BLOCKS_PER_RUN,
            seed=RANDOM_SEED
        )

    # Build run data
    run_data = build_run_data_from_indices(
        blocks, precomputed_indices, variant_config["feature_type"]
    )

    n_runs = len(run_data["runs"])
    n_blocks = sum(len(r["blocks"]) for r in run_data["runs"])

    if n_runs < 4:
        return {
            "accuracy": None,
            "n_runs": n_runs,
            "n_blocks": n_blocks,
            "n_features": 0,
            "error": "Insufficient runs for analysis",
            "precomputed_indices": precomputed_indices,
            "voice_runs": run_data["voice_runs"]
        }

    # Create hashing vectorizer (shared across folds for consistency)
    hashing_vectorizer = HashingVectorizer(
        analyzer='char',
        ngram_range=(3, 3),
        n_features=2**14,
        norm='l1',
        alternate_sign=False
    )

    # Run CV
    result = leave_one_run_out_cv(
        run_data,
        feature_type=variant_config["feature_type"],
        classifier=variant_config["classifier"],
        hashing_vectorizer=hashing_vectorizer
    )

    return {
        "accuracy": result["run_weighted_balanced_accuracy"],
        "n_runs": n_runs,
        "n_blocks": n_blocks,
        "n_features": result.get("n_features", 0),
        "precomputed_indices": precomputed_indices,
        "voice_runs": run_data["voice_runs"],
        "run_voices": run_data["run_voices"]
    }


def run_permutation_for_variant(blocks: list, variant_config: dict,
                                 perm_mapping: dict,
                                 precomputed_indices: dict) -> float:
    """
    Run CV with permuted labels for a single variant.

    Uses same pre-computed indices for reproducibility.
    """
    # Build run data
    run_data = build_run_data_from_indices(
        blocks, precomputed_indices, variant_config["feature_type"]
    )

    if len(run_data["runs"]) < 4:
        return 0.0

    # Apply permutation to run voices
    permuted_run_voices = {
        run_id: perm_mapping.get(run_id, voice)
        for run_id, voice in run_data["run_voices"].items()
    }
    run_data["run_voices"] = permuted_run_voices

    # Update voice_runs accordingly
    permuted_voice_runs = defaultdict(list)
    for run_id, voice in permuted_run_voices.items():
        permuted_voice_runs[voice].append(run_id)
    run_data["voice_runs"] = dict(permuted_voice_runs)

    # Also update voice labels in runs
    for run in run_data["runs"]:
        run["voice"] = permuted_run_voices[run["run_id"]]

    hashing_vectorizer = HashingVectorizer(
        analyzer='char',
        ngram_range=(3, 3),
        n_features=2**14,
        norm='l1',
        alternate_sign=False
    )

    try:
        result = leave_one_run_out_cv(
            run_data,
            feature_type=variant_config["feature_type"],
            classifier=variant_config["classifier"],
            hashing_vectorizer=hashing_vectorizer
        )
        return result["run_weighted_balanced_accuracy"] or 0.0
    except Exception:
        return 0.0


def run_max_statistic_permutation_test(blocks: list, variants: dict,
                                        observed_scores: dict,
                                        variant_indices: dict,
                                        master_voice_runs: dict,
                                        n_permutations: int) -> dict:
    """
    Run max-statistic permutation test across all variants.

    Same permutation applied to all variants for valid maxT correction.
    Uses corrected p-value formula: (1 + sum) / (1 + B)
    """
    print(f"  Running max-statistic permutation test ({n_permutations} permutations)...")

    # Generate permutations using master voice_runs (same for all variants)
    perms = sample_restricted_permutations(master_voice_runs, n_permutations, RANDOM_SEED)
    print(f"  Generated {len(perms)} unique permutations")

    # Observed max
    observed_max = max(observed_scores.values())

    # Track permutation results
    perm_max_scores = []
    perm_variant_scores = {vid: [] for vid in variants.keys()}

    for i, perm_mapping in enumerate(perms):
        if (i + 1) % 1000 == 0:
            print(f"    Permutation {i+1}/{len(perms)}...")

        # Compute score for each variant under this permutation
        variant_scores = {}
        for vid, vconfig in variants.items():
            score = run_permutation_for_variant(
                blocks, vconfig, perm_mapping, variant_indices[vid]
            )
            variant_scores[vid] = score
            perm_variant_scores[vid].append(score)

        # Record max across variants
        perm_max_scores.append(max(variant_scores.values()))

    perm_max_scores = np.array(perm_max_scores)

    # Corrected p-value formula per GPT: (1 + sum(perm >= obs)) / (1 + B)
    n_exceed = np.sum(perm_max_scores >= observed_max)
    corrected_p = (1 + n_exceed) / (1 + len(perms))

    # Per-variant uncorrected p-values (also corrected formula)
    uncorrected_p = {}
    for vid in variants.keys():
        scores = np.array(perm_variant_scores[vid])
        n_exceed_v = np.sum(scores >= observed_scores[vid])
        uncorrected_p[vid] = (1 + n_exceed_v) / (1 + len(scores))

    # Per-variant adjusted p-values (maxT-adjusted)
    # p_adj_i = Pr(max_j T_perm_j >= T_obs_i)
    adjusted_p = {}
    for vid in variants.keys():
        obs_score = observed_scores[vid]
        n_exceed_adj = np.sum(perm_max_scores >= obs_score)
        adjusted_p[vid] = (1 + n_exceed_adj) / (1 + len(perm_max_scores))

    return {
        "corrected_p_value": corrected_p,
        "uncorrected_p_values": uncorrected_p,
        "adjusted_p_values": adjusted_p,
        "observed_max": observed_max,
        "null_max_mean": float(np.mean(perm_max_scores)),
        "null_max_std": float(np.std(perm_max_scores)),
        "null_max_95_percentile": float(np.percentile(perm_max_scores, 95)),
        "n_permutations": len(perms)
    }


def generate_report(results: dict, output_path: Path):
    """Generate robustness testing report."""
    lines = [
        "# Phase 2.A: Robustness Testing Results",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Pre-registration:** DOI 10.17605/OSF.IO/4W3KH",
        "",
        "---",
        "",
        "## Summary",
        "",
        "**Primary result (Phase 2.0):** p = 0.177, accuracy = 24.2%",
        "",
        "**Robustness criterion (per GPT review):** corrected p >= 0.05",
        "",
        f"**Corrected p-value (max-statistic):** {results['permutation']['corrected_p_value']:.4f}",
        "",
    ]

    # Determine robustness (simplified per GPT)
    is_robust = results['permutation']['corrected_p_value'] >= 0.05

    if is_robust:
        lines.append("**CONCLUSION: Null result is ROBUST**")
        lines.append("")
        lines.append("No variant shows significant signal after multiple comparisons correction.")
    else:
        lines.append("**CONCLUSION: Null result is NOT robust**")
        lines.append("")
        lines.append("At least one variant shows significant signal. Investigate further.")

    lines.extend([
        "",
        "---",
        "",
        "## Variant Results",
        "",
        "| ID | Variant | Accuracy | Uncorrected p | Adjusted p | Runs | Blocks |",
        "|----|---------|----------|---------------|------------|------|--------|",
    ])

    for vid in sorted(results['variants'].keys()):
        vresult = results['variants'][vid]
        vconfig = results['variant_configs'][vid]
        uncorr_p = results['permutation']['uncorrected_p_values'].get(vid, None)
        adj_p = results['permutation']['adjusted_p_values'].get(vid, None)

        if vresult.get('error'):
            lines.append(f"| {vid} | {vconfig['description']} | ERROR | - | - | {vresult['n_runs']} | {vresult['n_blocks']} |")
        else:
            uncorr_str = f"{uncorr_p:.4f}" if uncorr_p else "-"
            adj_str = f"{adj_p:.4f}" if adj_p else "-"
            lines.append(f"| {vid} | {vconfig['description']} | {vresult['accuracy']:.1%} | {uncorr_str} | {adj_str} | {vresult['n_runs']} | {vresult['n_blocks']} |")

    lines.extend([
        "",
        "**Note:** Adjusted p-values use maxT correction (accounts for testing 6 variants).",
        "",
        "---",
        "",
        "## Variant Specifications",
        "",
        "| ID | Block Size | Quotes | Features | Classifier |",
        "|----|------------|--------|----------|------------|",
    ])

    for vid in sorted(results['variant_configs'].keys()):
        vconfig = results['variant_configs'][vid]
        feat_desc = {
            "fw": "Function words (169)",
            "ng": "Char 3-grams (hashed)",
            "combined": "FW + 3-grams"
        }.get(vconfig['feature_type'], vconfig['feature_type'])
        clf_desc = {
            "logreg": "Logistic Regression",
            "svm": "Linear SVM (C=1.0)"
        }.get(vconfig['classifier'], vconfig['classifier'])
        lines.append(f"| {vid} | {vconfig['target_size']} | {'Yes' if vconfig['include_quotes'] else 'No'} | {feat_desc} | {clf_desc} |")

    lines.extend([
        "",
        "---",
        "",
        "## Max-Statistic Permutation Test",
        "",
        f"- **Observed max accuracy:** {results['permutation']['observed_max']:.1%}",
        f"- **Null max distribution:** {results['permutation']['null_max_mean']:.1%} ± {results['permutation']['null_max_std']:.1%}",
        f"- **Null max 95th percentile:** {results['permutation']['null_max_95_percentile']:.1%}",
        f"- **Corrected p-value:** {results['permutation']['corrected_p_value']:.4f}",
        f"- **Permutations:** {results['permutation']['n_permutations']:,}",
        "",
        "The max-statistic method controls familywise error rate by comparing the",
        "maximum observed accuracy against the distribution of maximum accuracies",
        "under the null hypothesis (same permutation applied to all variants).",
        "",
        "P-value formula: (1 + sum(perm >= obs)) / (1 + B) per GPT recommendation.",
        "",
        "---",
        "",
        "## Methodology Notes (GPT-5.2 Pro Review)",
        "",
        "This analysis incorporates corrections from GPT-5.2 Pro review:",
        "",
        "1. **HashingVectorizer for char 3-grams** — No vocabulary leakage",
        "2. **Deterministic block capping** — Pre-computed indices reused across permutations",
        "3. **Separate scaling for combined features** — FW and n-grams scaled independently",
        "4. **LinearSVC with fixed C=1.0** — No tuning (insufficient N for nested CV)",
        "5. **Corrected p-value formula** — (1+sum)/(1+B) avoids zero p-values",
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ])

    if is_robust:
        lines.extend([
            "The null result from Phase 2.0 is **robust to analytic choices**:",
            "",
            "- Different block sizes (500, 1000, 2000 words) → no signal",
            "- Including quotations → no signal",
            "- Alternative features (character n-grams) → no signal",
            "- Combined features (FW + n-grams) → no signal",
            "- Alternative classifier (SVM) → no signal",
            "",
            "This strengthens confidence that the lack of stylistic differentiation",
            "is a genuine finding, not an artifact of specific analytic decisions.",
            "",
            "**Next steps:**",
            "- Phase 2.D: Garnett calibration (translation-layer context)",
            "- Phase 2.E: Write-up",
        ])
    else:
        lines.extend([
            "The null result is **NOT robust**. At least one variant shows signal.",
            "",
            "**Next steps:**",
            "- Identify which variant(s) show signal",
            "- Determine if signal is narrator, genre, or artifact",
            "- Run Phase 2.B (genre-controlled analysis) for diagnostics",
        ])

    lines.extend([
        "",
        "---",
        "",
        "*Analysis per pre-registered protocol with GPT-5.2 Pro methodology review.*",
        "*See docs/decisions/phase2-execution-plan.md*",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    print("=" * 70)
    print("PHASE 2.A: ROBUSTNESS TESTING")
    print("=" * 70)
    print()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    print(f"Loading blocks from {INPUT_FILE}...")
    data = load_blocks(INPUT_FILE)
    blocks = data["blocks"]
    print(f"  Loaded {len(blocks)} total blocks")

    # Define variants (pre-registered)
    variants = {
        "A1": {
            "description": "Block size 500",
            "target_size": 500,
            "include_quotes": False,
            "feature_type": "fw",
            "classifier": "logreg"
        },
        "A2": {
            "description": "Block size 2000",
            "target_size": 2000,
            "include_quotes": False,
            "feature_type": "fw",
            "classifier": "logreg"
        },
        "A3": {
            "description": "Include quotations",
            "target_size": 1000,
            "include_quotes": True,
            "feature_type": "fw",
            "classifier": "logreg"
        },
        "A4": {
            "description": "Character 3-grams",
            "target_size": 1000,
            "include_quotes": False,
            "feature_type": "ng",
            "classifier": "logreg"
        },
        "A5": {
            "description": "FW + char 3-grams",
            "target_size": 1000,
            "include_quotes": False,
            "feature_type": "combined",
            "classifier": "logreg"
        },
        "A6": {
            "description": "SVM classifier",
            "target_size": 1000,
            "include_quotes": False,
            "feature_type": "fw",
            "classifier": "svm"
        },
    }

    # Pre-compute block indices for each variant (deterministic)
    print("\nPre-computing block indices for deterministic capping...")
    variant_indices = {}
    for vid, vconfig in variants.items():
        indices = precompute_block_indices(
            blocks,
            target_size=vconfig["target_size"],
            include_quotes=vconfig["include_quotes"],
            max_blocks_per_run=MAX_BLOCKS_PER_RUN,
            seed=RANDOM_SEED + hash(vid) % 1000  # Variant-specific but deterministic
        )
        variant_indices[vid] = indices
        print(f"  {vid}: {len(indices)} runs, {sum(len(v) for v in indices.values())} blocks")

    # Run each variant
    print("\nRunning variants...")
    variant_results = {}
    observed_scores = {}
    master_voice_runs = None  # Will use first valid variant's voice_runs

    for vid, vconfig in variants.items():
        print(f"\n  {vid}: {vconfig['description']}...")
        result = run_single_variant(blocks, vconfig, variant_indices[vid])
        variant_results[vid] = result

        if result["accuracy"] is not None:
            observed_scores[vid] = result["accuracy"]
            print(f"    Accuracy: {result['accuracy']:.1%}")
            print(f"    Runs: {result['n_runs']}, Blocks: {result['n_blocks']}, Features: {result['n_features']}")

            # Use first valid variant's voice_runs as master
            if master_voice_runs is None:
                master_voice_runs = result.get("voice_runs", {})
        else:
            print(f"    ERROR: {result.get('error', 'Unknown')}")

    if not master_voice_runs:
        print("ERROR: No valid variants to run permutation test")
        return

    # Run max-statistic permutation test
    print("\n" + "=" * 70)
    perm_results = run_max_statistic_permutation_test(
        blocks, variants, observed_scores, variant_indices,
        master_voice_runs, N_PERMUTATIONS
    )

    print(f"\n  Corrected p-value: {perm_results['corrected_p_value']:.4f}")
    print(f"  Observed max: {perm_results['observed_max']:.1%}")
    print(f"  Null max: {perm_results['null_max_mean']:.1%} ± {perm_results['null_max_std']:.1%}")

    # Compile results
    results = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script_version": "1.1.0",
            "random_seed": RANDOM_SEED,
            "max_blocks_per_run": MAX_BLOCKS_PER_RUN,
            "n_permutations": N_PERMUTATIONS,
            "preregistration_doi": "10.17605/OSF.IO/4W3KH",
            "gpt_review": "GPT-5.2 Pro methodology review incorporated"
        },
        "variant_configs": variants,
        "variants": {
            vid: {k: v for k, v in vres.items()
                  if k not in ["precomputed_indices", "voice_runs", "run_voices"]}
            for vid, vres in variant_results.items()
        },
        "permutation": perm_results
    }

    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Generating report at {REPORT_FILE}...")
    generate_report(results, REPORT_FILE)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    is_robust = perm_results['corrected_p_value'] >= 0.05

    print(f"\nCorrected p-value (max-statistic): {perm_results['corrected_p_value']:.4f}")
    print(f"\nPer-variant results:")
    for vid in sorted(variants.keys()):
        acc = observed_scores.get(vid, None)
        uncorr_p = perm_results['uncorrected_p_values'].get(vid, None)
        adj_p = perm_results['adjusted_p_values'].get(vid, None)
        if acc is not None:
            print(f"  {vid}: {acc:.1%} (uncorr p={uncorr_p:.4f}, adj p={adj_p:.4f})")
        else:
            print(f"  {vid}: ERROR")

    if is_robust:
        print("\nCONCLUSION: Null result is ROBUST")
        print("  → Proceed to Phase 2.D (Garnett) and Phase 2.E (Write-up)")
    else:
        print("\nCONCLUSION: Null result is NOT robust")
        print("  → Investigate signal source (Phase 2.B)")

    print(f"\nFull report: {REPORT_FILE}")

    return results


if __name__ == "__main__":
    main()
