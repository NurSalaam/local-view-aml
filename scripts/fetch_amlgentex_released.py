#!/usr/bin/env python
"""Fetch the AMLgentex dataset, verify it against its pinned digest, and record
a provenance manifest.

The AMLgentex dataset is ``sweden_100K_easy``, one of the datasets Östman et
al. 2025 release as Hugging Face repositories under
https://huggingface.co/AMLGentex. Its repository is pinned here to an exact
git revision, and its ``tx_log.csv`` to the sha256 and byte size of the copy
the study read. The file is the raw generator log, including the rows against
the source and sink accounts that the loader drops. It lands under
``data/raw/amlgentex/sweden_100K_easy/``; the manifest this script writes
beside it records the repo id, the pinned revision and the file's sha256 and
byte size.

Re-running is idempotent: when the file is present and matches its pin,
nothing is fetched.

Usage
-----
    uv run python scripts/fetch_amlgentex_released.py
    uv run python scripts/fetch_amlgentex_released.py --verify-only

Total download is ~634 MB. No authentication needed: the repo is public (MIT
licence), so the download sends no Hugging Face token, even one saved on the
machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "AMLGentex/sweden_100K_easy"
REVISION = "9164f174032ba1f93361f07613c6c4df03634089"
EXPECTED: dict[str, tuple[str, int]] = {
    "tx_log.csv": ("baade848ac187f2518a61b6cf02c1afa0a805e4940ada7f50a20803ee6dc8273",
                   633_864_438),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "raw" / "amlgentex" / REPO_ID.split("/")[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_manifest() -> None:
    man = {"source": f"https://huggingface.co/datasets/{REPO_ID}",
           "reference": "Ostman et al. (2025), AMLgentex",
           "repo_id": REPO_ID, "revision": REVISION, "fetch_date": str(date.today()),
           "files": {f: {"sha256": d, "bytes": b} for f, (d, b) in EXPECTED.items()}}
    (DEST / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")


def status(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest, size = EXPECTED[path.name]
    if path.stat().st_size != size or sha256(path) != digest:
        return "HASH MISMATCH"
    return "ok"


def fetch(verify_only: bool) -> int:
    present = {f: status(DEST / f) for f in EXPECTED}

    missing = [f for f, s in present.items() if s != "ok"]
    if not verify_only and missing:
        for fname in missing:
            print(f"[   fetching ] {REPO_ID}/{fname} @ {REVISION[:12]} ...")
            hf_hub_download(REPO_ID, fname, repo_type="dataset", revision=REVISION,
                            local_dir=DEST, token=False)
        present = {f: status(DEST / f) for f in EXPECTED}

    for fname, s in present.items():
        print(f"[{s:>13}] {fname}  ({EXPECTED[fname][1]:,} B)")
    bad = sum(s != "ok" for s in present.values())
    if not bad:
        save_manifest()
        print(f"\nmanifest: {DEST / 'manifest.json'}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the AMLgentex dataset.")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the present file against the pinned sha256 and byte "
                         "size; download nothing")
    return fetch(ap.parse_args().verify_only)


if __name__ == "__main__":
    raise SystemExit(main())
