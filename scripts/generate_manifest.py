#!/usr/bin/env python3
"""
Generate artifact manifest for the preprocessing pipeline.

Creates a machine-checkable manifest linking all artifacts with their
hashes, parent relationships, and generation metadata.
"""

import hashlib
import json
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

SCRIPT_VERSION = "1.0.0"

DATA_DIR = Path(__file__).parent.parent / "data/text"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_FILE = PROCESSED_DIR / "manifest.json"


def compute_hash(filepath: Path) -> Optional[str]:
    """Compute SHA-256 hash of file."""
    if not filepath.exists():
        return None
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit_hash() -> Optional[str]:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_file_info(filepath: Path, parent: Optional[str] = None,
                  generator: Optional[str] = None) -> dict:
    """Get file metadata."""
    if not filepath.exists():
        return {"exists": False, "path": str(filepath)}

    stat = filepath.stat()
    return {
        "path": str(filepath),
        "exists": True,
        "hash_sha256": compute_hash(filepath),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "parent": parent,
        "generator": generator,
    }


def main():
    """Generate the artifact manifest."""
    print("=" * 60)
    print("Generating Artifact Manifest")
    print("=" * 60)

    manifest = {
        "manifest_version": "1.0.0",
        "generated": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/generate_manifest.py",
        "git_commit": get_git_commit_hash(),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "dependencies": {
            "note": "All scripts use Python standard library only",
            "python_minimum": "3.10",
            "external_packages": [],
        },
        "artifacts": {
            "inputs": {},
            "processed": {},
            "logs": {},
            "scripts": {},
        }
    }

    # Input files
    print("\nScanning input files...")
    inputs = [
        ("book-of-mormon-1830-replica.txt", None, "Archive.org download"),
        ("book-of-mormon-1830-gutenberg.txt", None, "Project Gutenberg download"),
        ("book-of-mormon-modern-lds.txt", None, "scriptures.nephi.org extraction"),
    ]
    for filename, parent, source in inputs:
        filepath = DATA_DIR / filename
        info = get_file_info(filepath, parent, source)
        manifest["artifacts"]["inputs"][filename] = info
        status = "✓" if info.get("exists") else "✗"
        print(f"  {status} {filename}")

    # Processed files
    print("\nScanning processed files...")
    processed = [
        ("bom-1830-clean.txt", "book-of-mormon-1830-replica.txt", "preprocess_1830.py"),
        ("bom-1830-clean-headers-marked.txt", "book-of-mormon-1830-replica.txt", "preprocess_1830.py"),
        ("bom-1830-clean-no-corrections.txt", "book-of-mormon-1830-replica.txt", "preprocess_1830.py"),
        ("bom-1830-dehyphenated.txt", "bom-1830-clean.txt", "dehyphenate.py"),
        ("bom-1830-normalized.txt", "bom-1830-dehyphenated.txt", "normalize_unicode.py"),
        ("removed-modern-preface.txt", "book-of-mormon-1830-replica.txt", "preprocess_1830.py"),
        ("removed-garbled-toc.txt", "book-of-mormon-1830-replica.txt", "preprocess_1830.py"),
    ]
    for filename, parent, generator in processed:
        filepath = PROCESSED_DIR / filename
        info = get_file_info(filepath, parent, generator)
        manifest["artifacts"]["processed"][filename] = info
        status = "✓" if info.get("exists") else "✗"
        print(f"  {status} {filename}")

    # Log files
    print("\nScanning log files...")
    logs = [
        "preprocessing_log.json",
        "dehyphenation_log.json",
        "normalization_log.json",
    ]
    for filename in logs:
        filepath = PROCESSED_DIR / filename
        info = get_file_info(filepath)
        manifest["artifacts"]["logs"][filename] = info
        status = "✓" if info.get("exists") else "✗"
        print(f"  {status} {filename}")

    # Scripts
    print("\nScanning scripts...")
    scripts_dir = Path(__file__).parent
    scripts = [
        "preprocess_1830.py",
        "dehyphenate.py",
        "normalize_unicode.py",
        "generate_manifest.py",
    ]
    for filename in scripts:
        filepath = scripts_dir / filename
        info = get_file_info(filepath)
        manifest["artifacts"]["scripts"][filename] = info
        status = "✓" if info.get("exists") else "✗"
        print(f"  {status} {filename}")

    # Provenance chain summary
    manifest["provenance_chain"] = [
        {
            "step": 1,
            "input": "book-of-mormon-1830-replica.txt",
            "output": ["bom-1830-clean.txt", "bom-1830-clean-headers-marked.txt",
                      "bom-1830-clean-no-corrections.txt", "removed-modern-preface.txt",
                      "removed-garbled-toc.txt"],
            "script": "preprocess_1830.py",
            "log": "preprocessing_log.json",
        },
        {
            "step": 2,
            "input": "bom-1830-clean.txt",
            "output": ["bom-1830-dehyphenated.txt"],
            "script": "dehyphenate.py",
            "log": "dehyphenation_log.json",
        },
        {
            "step": 3,
            "input": "bom-1830-dehyphenated.txt",
            "output": ["bom-1830-normalized.txt"],
            "script": "normalize_unicode.py",
            "log": "normalization_log.json",
        },
    ]

    # Write manifest
    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    manifest_hash = compute_hash(MANIFEST_FILE)
    print(f"\nManifest saved: {MANIFEST_FILE}")
    print(f"Manifest hash: {manifest_hash}")

    # Summary
    total = sum(
        1 for category in manifest["artifacts"].values()
        for item in category.values()
        if item.get("exists")
    )
    missing = sum(
        1 for category in manifest["artifacts"].values()
        for item in category.values()
        if not item.get("exists")
    )

    print("\n" + "=" * 60)
    print(f"SUMMARY: {total} artifacts found, {missing} missing")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
