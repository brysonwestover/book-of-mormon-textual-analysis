#!/usr/bin/env python3
"""
Extract stylometric features from voice blocks for authorship attribution.

Features implemented:
1. Most-frequent function words (MFW) - Burrows Delta style
2. Character n-grams (2-4 grams)
3. Word length distribution
4. Sentence length statistics

See: docs/decisions/block-derivation-strategy.md

Version: 1.0.0
Date: 2026-02-01
"""

import json
import re
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Configuration
INPUT_FILE = Path("data/text/processed/bom-voice-blocks.json")
OUTPUT_FILE = Path("data/text/processed/bom-stylometric-features.json")

# Standard function word list (Burrows-style)
# These are style-heavy, content-light features
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
    # Prepositions
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "off", "over", "under",
    "of", "unto", "upon",
    # Conjunctions
    "and", "but", "or", "nor", "for", "yet", "so",
    "if", "then", "because", "although", "while", "whereas",
    # Auxiliary verbs
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    # Negation
    "not", "no", "never", "neither",
    # Adverbs (common)
    "now", "then", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "any", "such", "only", "also", "very", "just", "even",
    # Book of Mormon specific archaisms
    "ye", "yea", "nay", "thus", "therefore", "wherefore", "behold",
    "hath", "doth", "didst", "hast", "art", "wilt", "shalt",
    "thee", "thou", "thy", "thine", "thyself",
    "verily", "except", "lest", "whoso", "whosoever",
    "forth", "hence", "hither", "thither", "whither",
]


def load_blocks(filepath: Path) -> dict:
    """Load the voice blocks JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def tokenize(text: str) -> list:
    """Simple word tokenization."""
    # Lowercase and split on non-word characters
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    return tokens


def get_sentences(text: str) -> list:
    """Split text into sentences."""
    # Simple sentence splitting on . ! ?
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_mfw_features(text: str, word_list: list = FUNCTION_WORDS) -> dict:
    """
    Extract most-frequent function word features.

    Returns relative frequencies (per 1000 words) for each function word.
    """
    tokens = tokenize(text)
    total_words = len(tokens)

    if total_words == 0:
        return {word: 0.0 for word in word_list}

    word_counts = Counter(tokens)

    # Calculate relative frequency per 1000 words
    features = {}
    for word in word_list:
        count = word_counts.get(word, 0)
        features[f"mfw_{word}"] = (count / total_words) * 1000

    return features


def extract_char_ngrams(text: str, n_range: tuple = (2, 4), top_k: int = 100) -> dict:
    """
    Extract character n-gram frequencies.

    Args:
        text: Input text
        n_range: Range of n-gram sizes (inclusive)
        top_k: Number of top n-grams to include per size

    Returns:
        Dictionary of n-gram relative frequencies
    """
    text = text.lower()
    # Keep only letters and spaces
    text = re.sub(r'[^a-z ]', '', text)

    features = {}

    for n in range(n_range[0], n_range[1] + 1):
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        total = len(ngrams)

        if total == 0:
            continue

        counts = Counter(ngrams)

        # Get relative frequencies for top k
        for ngram, count in counts.most_common(top_k):
            features[f"char{n}_{ngram}"] = (count / total) * 1000

    return features


def extract_word_length_features(text: str) -> dict:
    """Extract word length distribution features."""
    tokens = tokenize(text)

    if not tokens:
        return {
            "wl_mean": 0.0,
            "wl_std": 0.0,
            "wl_1": 0.0, "wl_2": 0.0, "wl_3": 0.0, "wl_4": 0.0,
            "wl_5": 0.0, "wl_6": 0.0, "wl_7": 0.0, "wl_8plus": 0.0
        }

    lengths = [len(t) for t in tokens]
    total = len(lengths)

    # Mean and standard deviation
    mean_len = sum(lengths) / total
    variance = sum((l - mean_len) ** 2 for l in lengths) / total
    std_len = math.sqrt(variance)

    # Distribution by length
    length_counts = Counter(lengths)

    features = {
        "wl_mean": mean_len,
        "wl_std": std_len,
    }

    # Proportion of words of each length (1-7, 8+)
    for i in range(1, 8):
        features[f"wl_{i}"] = (length_counts.get(i, 0) / total) * 100

    features["wl_8plus"] = (sum(count for length, count in length_counts.items()
                                if length >= 8) / total) * 100

    return features


def extract_sentence_features(text: str) -> dict:
    """Extract sentence-level features."""
    sentences = get_sentences(text)

    if not sentences:
        return {
            "sent_mean_words": 0.0,
            "sent_mean_chars": 0.0,
            "sent_count": 0
        }

    # Words per sentence
    words_per_sent = [len(tokenize(s)) for s in sentences]
    chars_per_sent = [len(s) for s in sentences]

    features = {
        "sent_mean_words": sum(words_per_sent) / len(words_per_sent),
        "sent_mean_chars": sum(chars_per_sent) / len(chars_per_sent),
        "sent_count": len(sentences)
    }

    return features


def extract_vocabulary_features(text: str) -> dict:
    """Extract vocabulary richness features."""
    tokens = tokenize(text)
    total = len(tokens)

    if total == 0:
        return {
            "vocab_size": 0,
            "vocab_ttr": 0.0,  # Type-token ratio
            "vocab_hapax": 0.0  # Proportion of words appearing once
        }

    types = set(tokens)
    type_count = len(types)
    word_counts = Counter(tokens)
    hapax = sum(1 for word, count in word_counts.items() if count == 1)

    features = {
        "vocab_size": type_count,
        "vocab_ttr": type_count / total,  # Type-token ratio
        "vocab_hapax": hapax / type_count if type_count > 0 else 0  # Hapax ratio
    }

    return features


def extract_all_features(text: str) -> dict:
    """Extract all stylometric features from a text block."""
    features = {}

    # Function words (most important for authorship)
    features.update(extract_mfw_features(text))

    # Character n-grams
    features.update(extract_char_ngrams(text))

    # Word length distribution
    features.update(extract_word_length_features(text))

    # Sentence features
    features.update(extract_sentence_features(text))

    # Vocabulary richness
    features.update(extract_vocabulary_features(text))

    return features


def main():
    print("=" * 70)
    print("STYLOMETRIC FEATURE EXTRACTION")
    print("=" * 70)
    print()

    # Load blocks
    print(f"Loading blocks from {INPUT_FILE}...")
    data = load_blocks(INPUT_FILE)
    blocks = data["blocks"]
    print(f"  Loaded {len(blocks)} blocks")
    print()

    # Filter for primary analysis (1000-word, original voice, major voices)
    target_size = 1000
    major_voices = ["MORMON", "NEPHI", "MORONI", "JACOB"]

    primary_blocks = [
        b for b in blocks
        if b["target_size"] == target_size
        and b["quote_status"] == "original"
        and b["voice"] in major_voices
    ]
    print(f"Primary analysis blocks (1000w, original, 4 voices): {len(primary_blocks)}")

    # Extract features
    print()
    print("Extracting features...")

    feature_data = []
    all_feature_names = set()

    for i, block in enumerate(primary_blocks):
        if (i + 1) % 50 == 0:
            print(f"  Processing block {i + 1}/{len(primary_blocks)}...")

        text = block["text"]
        features = extract_all_features(text)

        # Add metadata
        record = {
            "block_id": block["block_id"],
            "run_id": block["run_id"],
            "voice": block["voice"],
            "frame_narrator": block["frame_narrator"],
            "word_count": block["word_count"],
            "book": block["book"],
            "start_ref": block["start_ref"],
            "end_ref": block["end_ref"],
            "features": features
        }

        feature_data.append(record)
        all_feature_names.update(features.keys())

    print(f"  Extracted {len(all_feature_names)} features per block")
    print()

    # Compute feature statistics
    print("Computing feature statistics...")

    # Group by voice
    by_voice = {}
    for record in feature_data:
        voice = record["voice"]
        if voice not in by_voice:
            by_voice[voice] = []
        by_voice[voice].append(record)

    voice_stats = {}
    for voice, records in by_voice.items():
        voice_stats[voice] = {
            "block_count": len(records),
            "total_words": sum(r["word_count"] for r in records)
        }

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 70)
    print()
    print(f"Total features extracted: {len(all_feature_names)}")
    print(f"  - Function words (MFW): {len([f for f in all_feature_names if f.startswith('mfw_')])}")
    print(f"  - Character n-grams: {len([f for f in all_feature_names if f.startswith('char')])}")
    print(f"  - Word length: {len([f for f in all_feature_names if f.startswith('wl_')])}")
    print(f"  - Sentence: {len([f for f in all_feature_names if f.startswith('sent_')])}")
    print(f"  - Vocabulary: {len([f for f in all_feature_names if f.startswith('vocab_')])}")
    print()
    print("Blocks by voice:")
    for voice, stats in sorted(voice_stats.items(), key=lambda x: -x[1]["block_count"]):
        print(f"  {voice}: {stats['block_count']} blocks, {stats['total_words']:,} words")
    print()

    # Prepare output
    output = {
        "metadata": {
            "source_file": str(INPUT_FILE),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script_version": "1.0.0",
            "target_block_size": target_size,
            "voices_included": major_voices,
            "filter": "quote_status == 'original'",
            "feature_count": len(all_feature_names),
            "block_count": len(feature_data)
        },
        "feature_names": sorted(all_feature_names),
        "voice_statistics": voice_stats,
        "function_words_used": FUNCTION_WORDS,
        "blocks": feature_data
    }

    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Blocks: {len(feature_data)}")
    print(f"Features: {len(all_feature_names)}")
    print()
    print("Ready for classification (Step 3)")

    return output


if __name__ == "__main__":
    main()
