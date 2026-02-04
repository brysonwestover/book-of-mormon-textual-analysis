#!/usr/bin/env python3
"""
Segment-level comparison: match quoted speech segments between sources.
"""

import re
from xml.etree import ElementTree as ET
from pathlib import Path
from difflib import SequenceMatcher

HILLMAN_PATH = Path("data/reference/copyrighted/The Book of Mormon With Voices Identified.txt")
BCGMAXWELL_DOCX_XML = Path("data/reference/external/bcgmaxwell-extracted/word/document.xml")


def parse_hillman_segments(filepath):
    """Parse Hillman into individual speaker segments with text."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    segments = []
    speaker_pattern = re.compile(r'^([A-Z][A-Z₀₁₂₃₄₅₆₇₈₉\s,]+(?:\([^)]+\))?)\s*$')
    chapter_pattern = re.compile(r'^Chapter\s+(\d+)\s*$')

    current_speaker = None
    current_text = []
    current_book = None
    current_chapter = None

    book_patterns = [
        (re.compile(r'THE FIRST BOOK OF NEPHI'), '1 Nephi'),
        (re.compile(r'THE SECOND BOOK OF NEPHI'), '2 Nephi'),
        (re.compile(r'THE BOOK OF JACOB'), 'Jacob'),
        (re.compile(r'THE BOOK OF ENOS'), 'Enos'),
        (re.compile(r'THE BOOK OF JAROM'), 'Jarom'),
        (re.compile(r'THE BOOK OF OMNI'), 'Omni'),
        (re.compile(r'THE WORDS OF MORMON'), 'Words of Mormon'),
        (re.compile(r'THE BOOK OF MOSIAH'), 'Mosiah'),
        (re.compile(r'THE BOOK OF ALMA'), 'Alma'),
        (re.compile(r'THE BOOK OF HELAMAN'), 'Helaman'),
        (re.compile(r'THIRD NEPHI'), '3 Nephi'),
        (re.compile(r'FOURTH NEPHI'), '4 Nephi'),
        (re.compile(r'THE BOOK OF MORMON'), 'Mormon'),
        (re.compile(r'THE BOOK OF ETHER'), 'Ether'),
        (re.compile(r'THE BOOK OF MORONI'), 'Moroni'),
    ]

    lines = content.split('\n')
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Book detection
        for pattern, book_name in book_patterns:
            if pattern.search(line_stripped):
                current_book = book_name
                break

        # Chapter detection
        chapter_match = chapter_pattern.match(line_stripped)
        if chapter_match:
            if current_speaker and current_text:
                segments.append({
                    'speaker': current_speaker,
                    'text': ' '.join(current_text),
                    'book': current_book,
                    'chapter': current_chapter
                })
                current_text = []
            current_chapter = int(chapter_match.group(1))
            continue

        # Speaker detection
        speaker_match = speaker_pattern.match(line_stripped)
        if speaker_match and len(line_stripped) < 80:
            if current_speaker and current_text:
                segments.append({
                    'speaker': current_speaker,
                    'text': ' '.join(current_text),
                    'book': current_book,
                    'chapter': current_chapter
                })
                current_text = []
            current_speaker = speaker_match.group(1).strip()
            continue

        # Text content (strip verse numbers)
        text = re.sub(r'^\d+\s+', '', line_stripped)
        if text and current_speaker:
            current_text.append(text)

    # Final segment
    if current_speaker and current_text:
        segments.append({
            'speaker': current_speaker,
            'text': ' '.join(current_text),
            'book': current_book,
            'chapter': current_chapter
        })

    return segments


def parse_bcgmaxwell_segments(filepath):
    """Parse bcgmaxwell into individual speaker segments."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    texts = []
    for t in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
        if t.text:
            texts.append(t.text)

    full_text = ''.join(texts)

    segments = []
    speaker_pattern = re.compile(r'\[([^\[]+)\[([^\]]*)\]\]')
    verse_pattern = re.compile(r'<(\d+),([^,]+),(\d+),(\d+)>')

    current_book = None
    current_chapter = None

    parts = re.split(r'(<\d+,[^>]+>)', full_text)
    for part in parts:
        verse_match = verse_pattern.match(part)
        if verse_match:
            _, book, chapter, verse = verse_match.groups()
            current_book = book.strip()
            current_chapter = int(chapter)
        else:
            for speaker_match in speaker_pattern.finditer(part):
                speaker = speaker_match.group(1)
                text = speaker_match.group(2)
                if text.strip():
                    segments.append({
                        'speaker': speaker,
                        'text': text.strip(),
                        'book': current_book,
                        'chapter': current_chapter
                    })

    return segments


def normalize_text(text):
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_speaker(speaker):
    """Normalize speaker name to category."""
    s = speaker.lower()

    # Divine
    if any(x in s for x in ['lord', 'god', 'christ', 'spirit', 'godhead', 'revelatory']):
        return 'DIVINE'
    if any(x in s for x in ['angel', 'celestial']):
        return 'ANGEL'

    # Main characters
    if 'nephi' in s:
        return 'NEPHI'
    if 'lehi' in s and 'nephi' not in s:
        return 'LEHI'
    if 'alma' in s:
        return 'ALMA'
    if 'moroni' in s:
        return 'MORONI'
    if 'mormon' in s:
        return 'MORMON'
    if 'benjamin' in s:
        return 'BENJAMIN'
    if 'mosiah' in s:
        return 'MOSIAH'
    if 'jacob' in s:
        return 'JACOB'
    if 'sariah' in s:
        return 'SARIAH'
    if 'laman' in s and 'lemuel' in s:
        return 'LAMAN_LEMUEL'
    if 'laman' in s:
        return 'LAMAN'
    if 'laban' in s:
        return 'LABAN'
    if 'abinadi' in s:
        return 'ABINADI'
    if 'ammon' in s:
        return 'AMMON'
    if 'isaiah' in s:
        return 'ISAIAH'

    return re.sub(r'[₀₁₂₃₄₅₆₇₈₉\s]+', '_', speaker.upper()).strip('_')


def find_matching_segment(text, segments, min_ratio=0.7):
    """Find best matching segment by text similarity."""
    text_norm = normalize_text(text)
    best_match = None
    best_ratio = 0

    for seg in segments:
        seg_text_norm = normalize_text(seg['text'])
        # Quick length filter
        if abs(len(text_norm) - len(seg_text_norm)) > max(len(text_norm), len(seg_text_norm)) * 0.5:
            continue

        ratio = SequenceMatcher(None, text_norm[:200], seg_text_norm[:200]).ratio()
        if ratio > best_ratio and ratio >= min_ratio:
            best_ratio = ratio
            best_match = seg

    return best_match, best_ratio


def main():
    print("=" * 70)
    print("SEGMENT-LEVEL VOICE COMPARISON")
    print("=" * 70)
    print()

    hillman = parse_hillman_segments(HILLMAN_PATH)
    bcgmaxwell = parse_bcgmaxwell_segments(BCGMAXWELL_DOCX_XML)

    print(f"Hillman segments: {len(hillman)}")
    print(f"bcgmaxwell segments: {len(bcgmaxwell)}")
    print()

    # Sample comparison on first 100 bcgmaxwell segments
    matches = 0
    mismatches = 0
    no_match = 0
    examples = []

    print("Comparing bcgmaxwell segments against Hillman...")
    print()

    for i, b_seg in enumerate(bcgmaxwell[:200]):
        h_seg, ratio = find_matching_segment(b_seg['text'], hillman)

        if h_seg:
            b_cat = normalize_speaker(b_seg['speaker'])
            h_cat = normalize_speaker(h_seg['speaker'])

            if b_cat == h_cat:
                matches += 1
            else:
                mismatches += 1
                if len(examples) < 15:
                    examples.append({
                        'bcgmaxwell': b_seg,
                        'hillman': h_seg,
                        'b_cat': b_cat,
                        'h_cat': h_cat,
                        'ratio': ratio
                    })
        else:
            no_match += 1

    total_compared = matches + mismatches
    print("=" * 70)
    print(f"RESULTS (first 200 bcgmaxwell segments)")
    print("=" * 70)
    print(f"Matched text, same speaker category: {matches}")
    print(f"Matched text, different speaker: {mismatches}")
    print(f"No text match found: {no_match}")
    if total_compared > 0:
        print(f"Agreement rate: {matches/total_compared*100:.1f}%")
    print()

    print("=" * 70)
    print("SAMPLE DISAGREEMENTS")
    print("=" * 70)
    for ex in examples:
        print()
        print(f"Text: {ex['bcgmaxwell']['text'][:70]}...")
        print(f"  bcgmaxwell: {ex['bcgmaxwell']['speaker']} -> [{ex['b_cat']}]")
        print(f"  Hillman:    {ex['hillman']['speaker']} -> [{ex['h_cat']}]")


if __name__ == "__main__":
    main()
