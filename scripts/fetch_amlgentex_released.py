#!/usr/bin/env python
"""Fetch the twelve released AMLgentex datasets, verify each against its
pinned digest, and record a provenance manifest per dataset.

Östman et al. 2025 release twelve datasets as Hugging Face repositories under
https://huggingface.co/AMLGentex. The AMLgentex dataset of the study is
``sweden_100K_easy``; the other eleven are measured by Experiment 1 alone
(Appendix C.9). Each repository is pinned here to an exact git revision, and
its ``tx_log.csv`` to the sha256 and byte size of the copy the study read.
The file is the raw generator log, including the rows against the source and
sink accounts that the loader drops. Each lands under
``data/raw/amlgentex/<name>/``; the manifest written beside it records the
repo id, the pinned revision and the file's sha256 and byte size.

Re-running is idempotent: a file present and matching its pin is not fetched.

Usage
-----
    uv run python scripts/fetch_amlgentex_released.py                    # all twelve
    uv run python scripts/fetch_amlgentex_released.py sweden_100K_easy   # named ones
    uv run python scripts/fetch_amlgentex_released.py --verify-only

Total download is ~20.5 GB, of which ``sweden_100K_easy`` is ~634 MB and the
two ``sweden_1M`` datasets ~6.4 GB each. No authentication needed: the repos
are public (MIT licence), so the download sends no Hugging Face token, even
one saved on the machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from huggingface_hub import hf_hub_download

NAMESPACE = "AMLGentex"
FILE = "tx_log.csv"
PINS: dict[str, tuple[str, str, int]] = {  # name: (revision, sha256, bytes)
    "spain_100K_difficult": (
        "044100187d660add3cf0b4a7685ba99e7970058d",
        "e5028d4150add8fbee8c5f31e378b4c7da452df66de4a088f599295799e5551c", 649_613_870),
    "spain_100K_easy": (
        "544ad1e80fd2cd7554340b930b846a2fd18a3e1e",
        "b852a4274e102835a9eb587079c7beb3e5018a9c43962bbeafc8523c5d2e22fa", 653_718_618),
    "sweden_100K_difficult": (
        "f09ec96982c58759b0475531e263dd065cbdd1f3",
        "d87e9742d019242ab1075076219c4c536cc8d082ca844590d7359f14d58666cb", 628_949_539),
    "sweden_100K_dist_change_difficult": (
        "4b236f8c692efeaf8e0378223d9a9f81f6a31935",
        "0d40d615f8e5323e407657e7f9725a96f01649094cb33b1ebdd9bce39b67eaae", 628_572_577),
    "sweden_100K_dist_change_easy": (
        "f35f093330c376816d1fafec8d29a6e220323e35",
        "caf97afcfdc68cb7cff21b1634e9b0c8a47bcd10ed0e8f370b3f56d05113a193", 633_199_615),
    "sweden_100K_easy": (
        "9164f174032ba1f93361f07613c6c4df03634089",
        "baade848ac187f2518a61b6cf02c1afa0a805e4940ada7f50a20803ee6dc8273", 633_864_438),
    "sweden_100K_no_dist_change_difficult": (
        "02477bf74d6fb6504af6cf63acc8cc1609d06746",
        "1103915d09cf474724da8168be1e8c32abcbe6afaf9ef4b950d42f8b49f6deb0", 628_887_087),
    "sweden_100K_no_dist_change_easy": (
        "84a9ccdabeae6ae52aee645eb0799952dd781bb5",
        "a3ab257c29cf497baf14184757f13124fc08122de54a9f9c48df9eb2075c5f8a", 631_752_969),
    "sweden_1M_difficult": (
        "412ebf0a6abebbb0b221379af7d0cd3318f6fc2e",
        "b47148eb03ebdcccc64699bb7378dccf97c48c26acf332e30664f8c94d3fe47b", 6_371_133_654),
    "sweden_1M_easy": (
        "59c3261501f5be2ea5421c7044544b956c89149f",
        "cac08c3abb7f8787481dc4e9794f66f3bc77499cbb79c6b5dab036d9b3a36d4b", 6_378_091_481),
    "us_100K_difficult": (
        "4fe3c022c06e480f14922d95305570b6c3f66513",
        "51fbdc09809762c10c030cc1b011518457f83cdd6cb2ae36ea625b96b9e2595e", 699_090_117),
    "us_100K_easy": (
        "20628a0563ad7b23f37d3f17dc1ec50def898761",
        "445c63a19c9d5333c75a36d26264597f0c0b8bde5f321110be191a687f392768", 703_304_179),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "data" / "raw" / "amlgentex"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_manifest(name: str) -> None:
    revision, digest, size = PINS[name]
    repo_id = f"{NAMESPACE}/{name}"
    man = {"source": f"https://huggingface.co/datasets/{repo_id}",
           "reference": "Ostman et al. (2025), AMLgentex",
           "repo_id": repo_id, "revision": revision, "fetch_date": str(date.today()),
           "files": {FILE: {"sha256": digest, "bytes": size}}}
    (ROOT / name / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")


def status(name: str) -> str:
    path = ROOT / name / FILE
    if not path.exists():
        return "missing"
    _, digest, size = PINS[name]
    if path.stat().st_size != size or sha256(path) != digest:
        return "HASH MISMATCH"
    return "ok"


def fetch(names: list[str], verify_only: bool) -> int:
    bad = 0
    for name in names:
        state = status(name)
        if not verify_only and state != "ok":
            revision = PINS[name][0]
            print(f"[   fetching ] {NAMESPACE}/{name}/{FILE} @ {revision[:12]} ...")
            hf_hub_download(f"{NAMESPACE}/{name}", FILE, repo_type="dataset",
                            revision=revision, local_dir=ROOT / name, token=False)
            state = status(name)
        print(f"[{state:>13}] {name}/{FILE}  ({PINS[name][2]:,} B)")
        if state == "ok":
            save_manifest(name)
        bad += state != "ok"
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the released AMLgentex datasets.")
    ap.add_argument("names", nargs="*", metavar="name",
                    help="datasets to fetch (default: all twelve)")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the present files against the pinned sha256 and "
                         "byte size; download nothing")
    args = ap.parse_args()
    unknown = sorted(set(args.names) - set(PINS))
    if unknown:
        ap.error(f"unknown datasets {unknown}; known: {sorted(PINS)}")
    return fetch(args.names or list(PINS), args.verify_only)


if __name__ == "__main__":
    raise SystemExit(main())
