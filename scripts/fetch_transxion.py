#!/usr/bin/env python
"""Fetch the TransXion raw data, verify it against its pinned digest, and record
a provenance manifest.

TransXion (Chen 2026) is released as a git repository whose ``data/tx.csv``
holds the payments, stored in Git LFS. The repository is cloned at a pinned
commit, the LFS object is pulled, and ``tx.csv`` is copied to
``data/raw/transxion``. The file is checked against the sha256 and byte size
pinned in ``EXPECTED``, those of the copy the study read; the manifest this
script writes records the source, the pinned revision and the file's sha256
and byte size.

Re-running is idempotent: when the file is present and matches its pin,
nothing is fetched.

Usage
-----
    uv run python scripts/fetch_transxion.py
    uv run python scripts/fetch_transxion.py --verify-only

Requires ``git`` and ``git-lfs``. Total download is ~250 MB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

SOURCE = "https://github.com/chaos-max/TransXion"
REVISION = "53932595c37c23b9f55ea5ddf5984e4d57b88369"
EXPECTED: dict[str, tuple[str, int]] = {
    "tx.csv": ("d6c345f07a8d8e26123dba5fe4f6572ef198e04fb94f9b53facb3fef6d197a35",
               245_159_741),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "data" / "raw" / "transxion"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_manifest() -> None:
    man = {"source": SOURCE, "revision": REVISION, "fetch_date": str(date.today()),
           "files": {f: {"sha256": d, "bytes": b} for f, (d, b) in EXPECTED.items()}}
    (TARGET / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")


def clone_into(dst: Path) -> None:
    """Clone at the pinned commit and pull its LFS objects into ``dst/data``."""
    env = os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(["git", "clone", "--quiet", SOURCE, str(dst)], check=True, env=env)
    subprocess.run(["git", "-C", str(dst), "checkout", "--quiet", REVISION], check=True, env=env)
    subprocess.run(["git", "-C", str(dst), "lfs", "pull"], check=True)


def status(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest, size = EXPECTED[path.name]
    if path.stat().st_size != size or sha256(path) != digest:
        return "HASH MISMATCH"
    return "ok"


def fetch(verify_only: bool) -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    present = {f: status(TARGET / f) for f in EXPECTED}

    if not verify_only and any(s != "ok" for s in present.values()):
        print(f"[   fetching ] {SOURCE} @ {REVISION[:12]} ...")
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "TransXion"
            clone_into(clone)
            for fname, s in present.items():
                if s != "ok":
                    shutil.copyfile(clone / "data" / fname, TARGET / fname)
        present = {f: status(TARGET / f) for f in EXPECTED}

    for fname, s in present.items():
        print(f"[{s:>13}] {fname}  ({EXPECTED[fname][1]:,} B)")
    bad = sum(s != "ok" for s in present.values())
    if not bad:
        save_manifest()
        print(f"\nmanifest: {TARGET / 'manifest.json'}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the TransXion raw data.")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the present file against the pinned sha256 and byte "
                         "size; download nothing")
    return fetch(ap.parse_args().verify_only)


if __name__ == "__main__":
    raise SystemExit(main())
