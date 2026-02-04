#!/usr/bin/env python3
"""
Compare voice/speaker attributions between Hillman/Hopkin and bcgmaxwell sources.

This script analyzes the level of agreement between the two sources to determine
if the open-source bcgmaxwell version can serve as a suitable alternative to
the copyrighted Hillman/Hopkin work.
"""

import re
from xml.etree import ElementTree as ET
from collections import defaultdict
from pathlib import Path

# Paths
HILLMAN_PATH = Path("data/reference/copyrighted/The Book of Mormon With Voices Identified.txt")
BCGMAXWELL_DOCX_XML = Path("data/reference/external/bcgmaxwell-extracted/word/document.xml")


def parse_hillman(filepath):
    """Parse the Hillman/Hopkin text file format."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find chapter markers and speaker attributions
    # Format: Speaker names in ALL CAPS (possibly with subscripts like ₁)
    # followed by text until the next speaker or chapter marker

    segments = []
    current_book = None
    current_chapter = None

    lines = content.split('\n')
    current_speaker = None
    current_text = []

    # Pattern for speaker names (ALL CAPS, possibly with subscript numbers and parentheticals)
    speaker_pattern = re.compile(r'^([A-Z][A-Z₀₁₂₃₄₅₆₇₈₉\s]+(?:\([^)]+\))?)\s*$')
    chapter_pattern = re.compile(r'^Chapter\s+(\d+)\s*$')
    book_pattern = re.compile(r'^(THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH)?\s*BOOK OF\s+(\w+)|^(\w+)\s*$')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for chapter markers
        chapter_match = chapter_pattern.match(line)
        if chapter_match:
            if current_speaker and current_text:
                segments.append({
                    'speaker': current_speaker,
                    'book': current_book,
                    'chapter': current_chapter,
                    'text': ' '.join(current_text)
                })
                current_text = []
            current_chapter = int(chapter_match.group(1))
            continue

        # Check for speaker markers
        speaker_match = speaker_pattern.match(line)
        if speaker_match and len(line) < 100:  # Speakers are short labels
            if current_speaker and current_text:
                segments.append({
                    'speaker': current_speaker,
                    'book': current_book,
                    'chapter': current_chapter,
                    'text': ' '.join(current_text)
                })
                current_text = []
            current_speaker = speaker_match.group(1).strip()
            continue

        # Otherwise it's text content
        if current_speaker:
            # Extract verse numbers
            verse_match = re.match(r'^(\d+)\s+(.+)$', line)
            if verse_match:
                current_text.append(verse_match.group(2))
            else:
                current_text.append(line)

    # Don't forget last segment
    if current_speaker and current_text:
        segments.append({
            'speaker': current_speaker,
            'book': current_book,
            'chapter': current_chapter,
            'text': ' '.join(current_text)
        })

    return segments


def parse_bcgmaxwell(filepath):
    """Parse the bcgmaxwell docx XML format."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Extract all text content
    texts = []
    for t in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
        if t.text:
            texts.append(t.text)

    full_text = ''.join(texts)

    # Parse the markup format:
    # <block_num,Book,Chapter,Verse>
    # {Writer{...}}
    # [Speaker[...]]

    segments = []

    # Pattern to find verse markers and speaker blocks
    verse_pattern = re.compile(r'<(\d+),([^,]+),(\d+),(\d+)>')
    speaker_pattern = re.compile(r'\[([^\[]+)\[([^\]]*)\]\]')

    # Split by verse markers
    parts = re.split(r'(<\d+,[^>]+>)', full_text)

    current_book = None
    current_chapter = None
    current_verse = None

    for part in parts:
        verse_match = verse_pattern.match(part)
        if verse_match:
            block_num, book, chapter, verse = verse_match.groups()
            current_book = book
            current_chapter = int(chapter)
            current_verse = int(verse)
        else:
            # Find all speaker blocks in this part
            for speaker_match in speaker_pattern.finditer(part):
                speaker = speaker_match.group(1)
                text = speaker_match.group(2)
                if text.strip():
                    segments.append({
                        'speaker': speaker,
                        'book': current_book,
                        'chapter': current_chapter,
                        'verse': current_verse,
                        'text': text.strip()
                    })

    return segments


def normalize_speaker(speaker):
    """Normalize speaker names for comparison."""
    # Remove subscript numbers and convert to lowercase
    speaker = speaker.lower()
    speaker = re.sub(r'[₀₁₂₃₄₅₆₇₈₉]', '', speaker)
    speaker = re.sub(r'\s+', ' ', speaker)
    speaker = speaker.strip()
    return speaker


def extract_unique_speakers(segments):
    """Extract unique speaker names from segments."""
    speakers = set()
    for seg in segments:
        speakers.add(seg['speaker'])
    return sorted(speakers)


def compare_sources():
    """Main comparison function."""
    print("=" * 70)
    print("VOICE SOURCE COMPARISON: Hillman/Hopkin vs bcgmaxwell")
    print("=" * 70)
    print()

    # Parse both sources
    print("Parsing Hillman/Hopkin...")
    hillman_segments = parse_hillman(HILLMAN_PATH)
    print(f"  Found {len(hillman_segments)} segments")

    print("Parsing bcgmaxwell...")
    bcgmaxwell_segments = parse_bcgmaxwell(BCGMAXWELL_DOCX_XML)
    print(f"  Found {len(bcgmaxwell_segments)} segments")
    print()

    # Extract unique speakers
    hillman_speakers = extract_unique_speakers(hillman_segments)
    bcgmaxwell_speakers = extract_unique_speakers(bcgmaxwell_segments)

    print("=" * 70)
    print("SPEAKER INVENTORY COMPARISON")
    print("=" * 70)
    print(f"\nHillman/Hopkin: {len(hillman_speakers)} unique speakers")
    print(f"bcgmaxwell: {len(bcgmaxwell_speakers)} unique speakers")
    print()

    # Normalize and compare speaker sets
    hillman_normalized = {normalize_speaker(s): s for s in hillman_speakers}
    bcgmaxwell_normalized = {normalize_speaker(s): s for s in bcgmaxwell_speakers}

    hillman_set = set(hillman_normalized.keys())
    bcgmaxwell_set = set(bcgmaxwell_normalized.keys())

    common = hillman_set & bcgmaxwell_set
    only_hillman = hillman_set - bcgmaxwell_set
    only_bcgmaxwell = bcgmaxwell_set - hillman_set

    print(f"Common speakers (normalized): {len(common)}")
    print(f"Only in Hillman: {len(only_hillman)}")
    print(f"Only in bcgmaxwell: {len(only_bcgmaxwell)}")
    print()

    if only_hillman:
        print("Speakers only in Hillman/Hopkin:")
        for s in sorted(only_hillman)[:20]:
            print(f"  - {hillman_normalized[s]}")
        if len(only_hillman) > 20:
            print(f"  ... and {len(only_hillman) - 20} more")
        print()

    if only_bcgmaxwell:
        print("Speakers only in bcgmaxwell:")
        for s in sorted(only_bcgmaxwell)[:20]:
            print(f"  - {bcgmaxwell_normalized[s]}")
        if len(only_bcgmaxwell) > 20:
            print(f"  ... and {len(only_bcgmaxwell) - 20} more")
        print()

    # Sample comparison of actual attributions
    print("=" * 70)
    print("SAMPLE ATTRIBUTION COMPARISON (First 20 bcgmaxwell segments)")
    print("=" * 70)
    print()

    for seg in bcgmaxwell_segments[:20]:
        text_preview = seg['text'][:60] + "..." if len(seg['text']) > 60 else seg['text']
        print(f"bcgmaxwell: [{seg['speaker']}]")
        print(f"  Book: {seg['book']}, Ch: {seg['chapter']}, V: {seg.get('verse', '?')}")
        print(f"  Text: {text_preview}")
        print()

    # Return summary stats
    return {
        'hillman_segments': len(hillman_segments),
        'bcgmaxwell_segments': len(bcgmaxwell_segments),
        'hillman_speakers': len(hillman_speakers),
        'bcgmaxwell_speakers': len(bcgmaxwell_speakers),
        'common_speakers': len(common),
        'only_hillman': len(only_hillman),
        'only_bcgmaxwell': len(only_bcgmaxwell)
    }


if __name__ == "__main__":
    stats = compare_sources()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Agreement potential: {stats['common_speakers']} / {max(stats['hillman_speakers'], stats['bcgmaxwell_speakers'])} speakers")
    overlap_pct = stats['common_speakers'] / max(stats['hillman_speakers'], stats['bcgmaxwell_speakers']) * 100
    print(f"Speaker overlap: {overlap_pct:.1f}%")
