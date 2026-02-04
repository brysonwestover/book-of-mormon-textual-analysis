#!/usr/bin/env python3
"""
Detailed verse-level comparison of voice attributions between Hillman/Hopkin and bcgmaxwell.
"""

import re
from xml.etree import ElementTree as ET
from collections import defaultdict
from pathlib import Path

HILLMAN_PATH = Path("data/reference/copyrighted/The Book of Mormon With Voices Identified.txt")
BCGMAXWELL_DOCX_XML = Path("data/reference/external/bcgmaxwell-extracted/word/document.xml")


def parse_bcgmaxwell(filepath):
    """Parse bcgmaxwell into verse-level attributions."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    texts = []
    for t in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
        if t.text:
            texts.append(t.text)

    full_text = ''.join(texts)

    verse_pattern = re.compile(r'<(\d+),([^,]+),(\d+),(\d+)>')
    speaker_pattern = re.compile(r'\[([^\[]+)\[([^\]]*)\]\]')

    # Build verse -> speaker mapping
    verse_speakers = defaultdict(list)
    parts = re.split(r'(<\d+,[^>]+>)', full_text)

    current_ref = None
    for part in parts:
        verse_match = verse_pattern.match(part)
        if verse_match:
            block_num, book, chapter, verse = verse_match.groups()
            # Normalize book name
            book = book.strip()
            current_ref = (book, int(chapter), int(verse))
        else:
            for speaker_match in speaker_pattern.finditer(part):
                speaker = speaker_match.group(1)
                text = speaker_match.group(2)
                if text.strip() and current_ref:
                    verse_speakers[current_ref].append({
                        'speaker': speaker,
                        'text': text.strip()[:100]
                    })

    return verse_speakers


def parse_hillman(filepath):
    """Parse Hillman into verse-level attributions."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    verse_speakers = defaultdict(list)

    # State tracking
    current_book = None
    current_chapter = None
    current_speaker = None

    # Patterns
    speaker_pattern = re.compile(r'^([A-Z][A-Z₀₁₂₃₄₅₆₇₈₉\s,]+(?:\([^)]+\))?)\s*$')
    chapter_pattern = re.compile(r'^Chapter\s+(\d+)\s*$')
    verse_pattern = re.compile(r'^(\d+)\s+(.+)$')
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

        # Check for book markers
        for pattern, book_name in book_patterns:
            if pattern.search(line_stripped):
                current_book = book_name
                break

        # Check for chapter markers
        chapter_match = chapter_pattern.match(line_stripped)
        if chapter_match:
            current_chapter = int(chapter_match.group(1))
            continue

        # Check for speaker markers
        speaker_match = speaker_pattern.match(line_stripped)
        if speaker_match and len(line_stripped) < 80:
            current_speaker = speaker_match.group(1).strip()
            continue

        # Check for verse content
        verse_match = verse_pattern.match(line_stripped)
        if verse_match and current_speaker and current_book and current_chapter:
            verse_num = int(verse_match.group(1))
            text = verse_match.group(2)
            ref = (current_book, current_chapter, verse_num)
            verse_speakers[ref].append({
                'speaker': current_speaker,
                'text': text[:100]
            })

    return verse_speakers


def normalize_speaker_category(speaker):
    """Map speaker names to broad categories for comparison."""
    speaker_lower = speaker.lower()

    # Divine/revelatory category
    if any(x in speaker_lower for x in ['lord', 'god', 'christ', 'spirit', 'godhead', 'revelatory']):
        return 'DIVINE'

    # Celestial beings
    if any(x in speaker_lower for x in ['angel', 'celestial']):
        return 'ANGEL'

    # Nephi variations
    if 'nephi' in speaker_lower:
        if '1' in speaker or 'son of lehi' in speaker_lower:
            return 'NEPHI1'
        elif '2' in speaker or 'son of helaman' in speaker_lower:
            return 'NEPHI2'
        elif '3' in speaker:
            return 'NEPHI3'
        elif '4' in speaker:
            return 'NEPHI4'
        return 'NEPHI'

    # Lehi
    if 'lehi' in speaker_lower and 'nephi' not in speaker_lower:
        return 'LEHI'

    # Alma variations
    if 'alma' in speaker_lower:
        if '1' in speaker or 'elder' in speaker_lower:
            return 'ALMA1'
        elif '2' in speaker or 'younger' in speaker_lower:
            return 'ALMA2'
        return 'ALMA'

    # Moroni variations
    if 'moroni' in speaker_lower:
        if '1' in speaker or 'captain' in speaker_lower:
            return 'MORONI1'
        elif '2' in speaker or 'son of mormon' in speaker_lower:
            return 'MORONI2'
        return 'MORONI'

    # Mormon
    if 'mormon' in speaker_lower and 'moroni' not in speaker_lower:
        return 'MORMON'

    # Common speakers
    if 'benjamin' in speaker_lower:
        return 'BENJAMIN'
    if 'mosiah' in speaker_lower:
        return 'MOSIAH'
    if 'jacob' in speaker_lower:
        return 'JACOB'
    if 'enos' in speaker_lower:
        return 'ENOS'
    if 'sariah' in speaker_lower:
        return 'SARIAH'
    if 'laman' in speaker_lower and 'lemuel' in speaker_lower:
        return 'LAMAN_LEMUEL'
    if 'laman' in speaker_lower:
        return 'LAMAN'
    if 'lemuel' in speaker_lower:
        return 'LEMUEL'
    if 'laban' in speaker_lower:
        return 'LABAN'
    if 'abinadi' in speaker_lower:
        return 'ABINADI'
    if 'ammon' in speaker_lower:
        return 'AMMON'
    if 'isaiah' in speaker_lower:
        return 'ISAIAH'
    if 'malachi' in speaker_lower:
        return 'MALACHI'

    # Return cleaned version for others
    return re.sub(r'[₀₁₂₃₄₅₆₇₈₉\s]+', '_', speaker.upper()).strip('_')


def compare_verses():
    """Compare verse-level attributions."""
    print("=" * 70)
    print("DETAILED VERSE-LEVEL VOICE COMPARISON")
    print("=" * 70)
    print()

    hillman = parse_hillman(HILLMAN_PATH)
    bcgmaxwell = parse_bcgmaxwell(BCGMAXWELL_DOCX_XML)

    print(f"Hillman verses with speakers: {len(hillman)}")
    print(f"bcgmaxwell verses with speakers: {len(bcgmaxwell)}")

    # Find common verses
    common_verses = set(hillman.keys()) & set(bcgmaxwell.keys())
    print(f"Common verses: {len(common_verses)}")
    print()

    # Compare attributions
    matches = 0
    mismatches = 0
    mismatch_examples = []

    for ref in sorted(common_verses):
        h_speakers = [normalize_speaker_category(s['speaker']) for s in hillman[ref]]
        b_speakers = [normalize_speaker_category(s['speaker']) for s in bcgmaxwell[ref]]

        # Check if primary speaker matches
        if h_speakers and b_speakers:
            if h_speakers[0] == b_speakers[0]:
                matches += 1
            else:
                mismatches += 1
                if len(mismatch_examples) < 30:
                    mismatch_examples.append({
                        'ref': ref,
                        'hillman': hillman[ref][0],
                        'hillman_cat': h_speakers[0],
                        'bcgmaxwell': bcgmaxwell[ref][0],
                        'bcgmaxwell_cat': b_speakers[0]
                    })

    total = matches + mismatches
    print("=" * 70)
    print("PRIMARY SPEAKER AGREEMENT")
    print("=" * 70)
    print(f"Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    print(f"Agreement rate: {matches/total*100:.1f}%")
    print()

    print("=" * 70)
    print("SAMPLE MISMATCHES (first 30)")
    print("=" * 70)
    print()

    for ex in mismatch_examples:
        book, ch, v = ex['ref']
        print(f"{book} {ch}:{v}")
        print(f"  Hillman:    {ex['hillman']['speaker']} -> [{ex['hillman_cat']}]")
        print(f"  bcgmaxwell: {ex['bcgmaxwell']['speaker']} -> [{ex['bcgmaxwell_cat']}]")
        print(f"  Text: {ex['hillman']['text'][:60]}...")
        print()

    return matches, mismatches


if __name__ == "__main__":
    compare_verses()
