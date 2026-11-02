#!/usr/bin/env python
"""Fetch IBM AMLworld raw data, verify it against pinned digests, and record a
provenance manifest.

The dataset bytes stay outside git under ``data/raw/ibm-aml``. Every file is
checked against the sha256 and byte size pinned in ``EXPECTED``, which are the
digests of the files the study read. Kaggle serves its current version of the
dataset and the CLI takes no version, so the pinned digests are what detect an
upstream change. The manifest written beside the files records the dataset
slug, the upstream source date and each verified file's sha256 and byte size.
Re-running is idempotent: a file already present that matches its pinned
digest is left untouched; anything missing or mismatched is (re)fetched and
verified.

Each named set (e.g. ``HI-Small``) is its payments file, ``*_Trans.csv``.

Usage
-----
    uv run python scripts/fetch_ibm_aml.py HI-Small LI-Small HI-Medium LI-Medium
    uv run python scripts/fetch_ibm_aml.py --verify-only        # check what is present

Requires Kaggle auth (``~/.kaggle/access_token`` or ``KAGGLE_USERNAME``/``KAGGLE_KEY``).
The four sets total ~7.1 GB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
SOURCE_DATE = "2025-07-08"  # upstream file creation date reported by the Kaggle API
SETS = ["HI-Small", "LI-Small", "HI-Medium", "LI-Medium"]
EXPECTED: dict[str, tuple[str, int]] = {
    "HI-Small_Trans.csv": ("b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040",
                           475_664_283),
    "LI-Small_Trans.csv": ("f7a9940339c78b5d1476071505b5867c0f5319640b9c4279d13f531bf965319d",
                           650_422_357),
    "HI-Medium_Trans.csv": ("3126afb8155e7c8815d62bc5370549a7b5ae6bf7dfe872b3cd7813e66a3d7ff5",
                            3_031_783_420),
    "LI-Medium_Trans.csv": ("0fc89584453c97472b1bfde7ca746e21d7167140c22a6c4875272b197a4a3910",
                            2_976_436_978),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "data" / "raw" / "ibm-aml"


def kaggle_exe() -> str:
    """The kaggle console script that ships in the same venv as this interpreter."""
    candidate = Path(sys.executable).with_name("kaggle")
    return str(candidate) if candidate.exists() else "kaggle"


def files_for(sets: list[str]) -> list[str]:
    return [f for f in EXPECTED if f.split("_", 1)[0] in sets]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest, size = EXPECTED[path.name]
    if path.stat().st_size != size or sha256(path) != digest:
        return "HASH MISMATCH"
    return "ok"


def recorded_files() -> dict:
    mp = TARGET / "manifest.json"
    return json.loads(mp.read_text())["files"] if mp.exists() else {}


def save_manifest(files: dict) -> None:
    man = {"dataset": DATASET, "source_date": SOURCE_DATE,
           "files": {f: files[f] for f in EXPECTED if f in files}}
    (TARGET / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")


def download_file(fname: str, dst: Path) -> None:
    """Fetch one file. Kaggle zips large files; small ones arrive as-is."""
    subprocess.run([kaggle_exe(), "datasets", "download", DATASET, "-f", fname, "-p", str(dst), "-o"],
                   check=True)
    zp = dst / f"{fname}.zip"
    if zp.exists():
        with zipfile.ZipFile(zp) as z:
            z.extractall(dst)
        zp.unlink()


def fetch(sets: list[str], verify_only: bool) -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    files = recorded_files()
    bad = 0

    for fname in files_for(sets):
        path = TARGET / fname
        state = status(path)
        if state != "ok" and not verify_only:
            print(f"[   fetching ] {fname} ...")
            download_file(fname, TARGET)
            state = status(path)
        print(f"[{state:>13}] {fname}  ({EXPECTED[fname][1]:,} B)")
        if state == "ok":
            files[fname] = {"sha256": EXPECTED[fname][0], "bytes": EXPECTED[fname][1]}
            save_manifest(files)  # persist incrementally so a long run is resumable
        else:
            files.pop(fname, None)
            bad += 1

    save_manifest(files)
    print(f"\nmanifest: {TARGET / 'manifest.json'}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch IBM AMLworld raw data.")
    ap.add_argument("sets", nargs="*", metavar="SET",
                    help=f"sets to fetch (default: all; any of: {', '.join(SETS)})")
    ap.add_argument("--verify-only", action="store_true",
                    help="check present files against the pinned sha256 and byte size; "
                         "download nothing")
    args = ap.parse_args()
    unknown = [s for s in args.sets if s not in SETS]
    if unknown:
        ap.error(f"unknown set(s) {unknown}; choose from {SETS}")
    return fetch(args.sets or SETS, args.verify_only)


if __name__ == "__main__":
    raise SystemExit(main())
