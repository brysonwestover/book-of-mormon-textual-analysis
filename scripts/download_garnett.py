#!/usr/bin/env python3
"""
Download Constance Garnett translations from Project Gutenberg.

This script downloads Russian literature translated by Constance Garnett
for use in the translation-layer calibration study.
"""

import json
import os
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "reference" / "garnett" / "raw"
USER_AGENT = "BookOfMormonTextualAnalysis/1.0 (Academic Research; Contact: research@example.com)"
DELAY_BETWEEN_REQUESTS = 2  # seconds

# Works to download - all translated by Constance Garnett
WORKS = [
    # Dostoevsky (6 works)
    {
        "author": "Dostoevsky",
        "title": "Crime and Punishment",
        "gutenberg_id": 2554,
        "publication_year": 1914,
        "genre": "novel",
    },
    {
        "author": "Dostoevsky",
        "title": "The Brothers Karamazov",
        "gutenberg_id": 28054,
        "publication_year": 1912,
        "genre": "novel",
    },
    {
        "author": "Dostoevsky",
        "title": "The Idiot",
        "gutenberg_id": 2638,
        "publication_year": 1913,
        "genre": "novel",
    },
    {
        "author": "Dostoevsky",
        "title": "Notes from Underground",
        "gutenberg_id": 600,
        "publication_year": 1918,
        "genre": "novella",
    },
    {
        "author": "Dostoevsky",
        "title": "The Possessed",
        "gutenberg_id": 8117,
        "publication_year": 1916,
        "genre": "novel",
    },
    {
        "author": "Dostoevsky",
        "title": "The Gambler",
        "gutenberg_id": 2197,
        "publication_year": 1917,
        "genre": "novella",
    },
    # Tolstoy (4 works)
    {
        "author": "Tolstoy",
        "title": "Anna Karenina",
        "gutenberg_id": 1399,
        "publication_year": 1901,
        "genre": "novel",
    },
    {
        "author": "Tolstoy",
        "title": "War and Peace",
        "gutenberg_id": 2600,
        "publication_year": 1904,
        "genre": "novel",
    },
    {
        "author": "Tolstoy",
        "title": "The Death of Ivan Ilych",
        "gutenberg_id": 43424,
        "publication_year": 1915,
        "genre": "novella",
    },
    {
        "author": "Tolstoy",
        "title": "The Kreutzer Sonata",
        "gutenberg_id": 689,
        "publication_year": 1915,
        "genre": "novella",
    },
    # Chekhov (4 works - short story collections)
    {
        "author": "Chekhov",
        "title": "The Lady with the Dog and Other Stories",
        "gutenberg_id": 13415,
        "publication_year": 1917,
        "genre": "stories",
    },
    {
        "author": "Chekhov",
        "title": "The Darling and Other Stories",
        "gutenberg_id": 13416,
        "publication_year": 1916,
        "genre": "stories",
    },
    {
        "author": "Chekhov",
        "title": "The Party and Other Stories",
        "gutenberg_id": 1945,
        "publication_year": 1917,
        "genre": "stories",
    },
    {
        "author": "Chekhov",
        "title": "The Wife and Other Stories",
        "gutenberg_id": 1732,
        "publication_year": 1918,
        "genre": "stories",
    },
    # Turgenev (5 works)
    {
        "author": "Turgenev",
        "title": "Fathers and Sons",
        "gutenberg_id": 30723,
        "publication_year": 1895,
        "genre": "novel",
    },
    {
        "author": "Turgenev",
        "title": "A House of Gentlefolk",
        "gutenberg_id": 6588,
        "publication_year": 1894,
        "genre": "novel",
    },
    {
        "author": "Turgenev",
        "title": "On the Eve",
        "gutenberg_id": 6319,
        "publication_year": 1895,
        "genre": "novel",
    },
    {
        "author": "Turgenev",
        "title": "Smoke",
        "gutenberg_id": 5765,
        "publication_year": 1896,
        "genre": "novel",
    },
    {
        "author": "Turgenev",
        "title": "Virgin Soil",
        "gutenberg_id": 6323,
        "publication_year": 1896,
        "genre": "novel",
    },
]


def slugify(title: str) -> str:
    """Convert title to a filename-safe slug."""
    # Convert to lowercase
    slug = title.lower()
    # Replace spaces and special chars with underscores
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    # Remove leading/trailing underscores
    slug = slug.strip("_")
    return slug


def get_gutenberg_url(gutenberg_id: int) -> str:
    """Get the plain text URL for a Project Gutenberg work."""
    # Project Gutenberg plain text URL format
    return f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"


def download_text(url: str) -> str:
    """Download text from URL with proper headers."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response:
            # Try UTF-8 first, fall back to Latin-1
            content = response.read()
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return content.decode("latin-1")
    except HTTPError as e:
        print(f"  HTTP Error {e.code}: {e.reason}")
        raise
    except URLError as e:
        print(f"  URL Error: {e.reason}")
        raise


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def main():
    """Download all Garnett translations and create manifest."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []

    print(f"Downloading {len(WORKS)} works translated by Constance Garnett")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)

    for i, work in enumerate(WORKS):
        author = work["author"]
        title = work["title"]
        gutenberg_id = work["gutenberg_id"]
        publication_year = work.get("publication_year")

        # Create filename
        title_slug = slugify(title)
        filename = f"{author.lower()}_{title_slug}.txt"
        filepath = OUTPUT_DIR / filename

        print(f"\n[{i+1}/{len(WORKS)}] {author} - {title}")
        print(f"  Gutenberg ID: {gutenberg_id}")
        print(f"  Filename: {filename}")

        # Download
        url = get_gutenberg_url(gutenberg_id)
        print(f"  Downloading from: {url}")

        try:
            text = download_text(url)

            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

            # Count words
            word_count = count_words(text)
            print(f"  Downloaded: {word_count:,} words")

            # Add to manifest
            manifest.append({
                "author": author,
                "title": title,
                "translator": "Constance Garnett",
                "gutenberg_id": gutenberg_id,
                "publication_year": publication_year,
                "genre": work.get("genre", "unknown"),
                "filename": filename,
                "word_count": word_count,
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        # Delay between requests (except for last one)
        if i < len(WORKS) - 1:
            print(f"  Waiting {DELAY_BETWEEN_REQUESTS}s before next request...")
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Write manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Successfully downloaded: {len(manifest)}/{len(WORKS)} works")
    print(f"Total words: {sum(m['word_count'] for m in manifest):,}")
    print(f"Manifest saved to: {manifest_path}")

    # Summary by author
    print("\nBy author:")
    authors = {}
    for m in manifest:
        author = m["author"]
        if author not in authors:
            authors[author] = {"count": 0, "words": 0}
        authors[author]["count"] += 1
        authors[author]["words"] += m["word_count"]

    for author, stats in sorted(authors.items()):
        print(f"  {author}: {stats['count']} works, {stats['words']:,} words")


if __name__ == "__main__":
    main()
