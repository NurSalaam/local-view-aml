#!/usr/bin/env python
"""Fetch the released AMLgentex datasets, verify them against pinned digests,
and record a provenance manifest.

The twelve dataset repositories under https://huggingface.co/AMLGentex are the
released datasets of Östman et al. 2025. Each is pinned here to an exact git
revision, and every file to the sha256 and byte size of the copy the study
read. The bytes stay outside git under ``data/raw/amlgentex/<name>/``; the
manifest written beside them records each repo id, its pinned revision and each
verified file's sha256 and byte size.

Re-running is idempotent: a file already present that matches its pinned
digest is left untouched; anything missing or mismatched is (re)fetched at the
pinned revision and verified. From each repo the study reads ``tx_log.csv``,
the raw generator log, including the rows against the source and sink
accounts that the loader drops.

Usage
-----
    uv run python scripts/fetch_amlgentex_released.py                    # all twelve
    uv run python scripts/fetch_amlgentex_released.py sweden_100K_easy   # one repo
    uv run python scripts/fetch_amlgentex_released.py --verify-only

Total download is ~19.2 GB (the two ``sweden_1M`` repos are ~6.4 GB each).
No authentication needed: the repos are public (MIT licence), so the download
sends no Hugging Face token, even one saved on the machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from huggingface_hub import hf_hub_download

NAMESPACE = "AMLGentex"

# name -> (pinned revision, {file: (sha256, bytes)})
EXPECTED: dict[str, tuple[str, dict[str, tuple[str, int]]]] = {
    "sweden_100K_difficult": ("f09ec96982c58759b0475531e263dd065cbdd1f3", {
        "tx_log.csv": ("d87e9742d019242ab1075076219c4c536cc8d082ca844590d7359f14d58666cb", 628_949_539),
    }),
    "sweden_100K_easy": ("9164f174032ba1f93361f07613c6c4df03634089", {
        "tx_log.csv": ("baade848ac187f2518a61b6cf02c1afa0a805e4940ada7f50a20803ee6dc8273", 633_864_438),
    }),
    "us_100K_difficult": ("4fe3c022c06e480f14922d95305570b6c3f66513", {
        "tx_log.csv": ("51fbdc09809762c10c030cc1b011518457f83cdd6cb2ae36ea625b96b9e2595e", 699_090_117),
    }),
    "spain_100K_easy": ("544ad1e80fd2cd7554340b930b846a2fd18a3e1e", {
        "tx_log.csv": ("b852a4274e102835a9eb587079c7beb3e5018a9c43962bbeafc8523c5d2e22fa", 653_718_618),
    }),
    "us_100K_easy": ("20628a0563ad7b23f37d3f17dc1ec50def898761", {
        "tx_log.csv": ("445c63a19c9d5333c75a36d26264597f0c0b8bde5f321110be191a687f392768", 703_304_179),
    }),
    "spain_100K_difficult": ("044100187d660add3cf0b4a7685ba99e7970058d", {
        "tx_log.csv": ("e5028d4150add8fbee8c5f31e378b4c7da452df66de4a088f599295799e5551c", 649_613_870),
    }),
    "sweden_1M_difficult": ("412ebf0a6abebbb0b221379af7d0cd3318f6fc2e", {
        "tx_log.csv": ("b47148eb03ebdcccc64699bb7378dccf97c48c26acf332e30664f8c94d3fe47b", 6_371_133_654),
    }),
    "sweden_1M_easy": ("59c3261501f5be2ea5421c7044544b956c89149f", {
        "tx_log.csv": ("cac08c3abb7f8787481dc4e9794f66f3bc77499cbb79c6b5dab036d9b3a36d4b", 6_378_091_481),
    }),
    "sweden_100K_no_dist_change_easy": ("84a9ccdabeae6ae52aee645eb0799952dd781bb5", {
        "tx_log.csv": ("a3ab257c29cf497baf14184757f13124fc08122de54a9f9c48df9eb2075c5f8a", 631_752_969),
    }),
    "sweden_100K_dist_change_easy": ("f35f093330c376816d1fafec8d29a6e220323e35", {
        "tx_log.csv": ("caf97afcfdc68cb7cff21b1634e9b0c8a47bcd10ed0e8f370b3f56d05113a193", 633_199_615),
    }),
    "sweden_100K_no_dist_change_difficult": ("02477bf74d6fb6504af6cf63acc8cc1609d06746", {
        "tx_log.csv": ("1103915d09cf474724da8168be1e8c32abcbe6afaf9ef4b950d42f8b49f6deb0", 628_887_087),
    }),
    "sweden_100K_dist_change_difficult": ("4b236f8c692efeaf8e0378223d9a9f81f6a31935", {
        "tx_log.csv": ("0d40d615f8e5323e407657e7f9725a96f01649094cb33b1ebdd9bce39b67eaae", 628_572_577),
    }),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "data" / "raw" / "amlgentex"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(name: str, fname: str) -> str:
    path = TARGET / name / fname
    if not path.exists():
        return "missing"
    digest, size = EXPECTED[name][1][fname]
    if path.stat().st_size != size or sha256(path) != digest:
        return "HASH MISMATCH"
    return "ok"


def recorded_repos() -> dict:
    mp = TARGET / "manifest.json"
    return json.loads(mp.read_text())["repos"] if mp.exists() else {}


def save_manifest(verified: dict[str, set[str]]) -> None:
    repos = {name: {"repo_id": f"{NAMESPACE}/{name}", "revision": revision,
                    "files": {f: {"sha256": d, "bytes": b}
                              for f, (d, b) in files.items() if f in verified.get(name, ())}}
             for name, (revision, files) in EXPECTED.items() if verified.get(name)}
    man = {"source": f"https://huggingface.co/{NAMESPACE}",
           "reference": "Ostman et al. (2025), AMLgentex",
           "fetch_date": str(date.today()), "repos": repos}
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")


def fetch(names: list[str], verify_only: bool) -> int:
    verified = {name: {f for f, spec in repo["files"].items()
                       if name in EXPECTED and EXPECTED[name][1].get(f) == (spec["sha256"], spec["bytes"])}
                for name, repo in recorded_repos().items()}
    bad = 0

    for name in names:
        revision, files = EXPECTED[name]
        done = verified.setdefault(name, set())
        for fname, (_, size) in files.items():
            state = status(name, fname)
            if state != "ok" and not verify_only:
                print(f"[   fetching ] {name}/{fname} ...")
                hf_hub_download(f"{NAMESPACE}/{name}", fname, repo_type="dataset",
                                revision=revision, local_dir=TARGET / name, token=False)
                state = status(name, fname)
            print(f"[{state:>13}] {name}/{fname}  ({size:,} B)")
            if state == "ok":
                done.add(fname)
                save_manifest(verified)  # persist incrementally so a long run is resumable
            else:
                done.discard(fname)
                bad += 1

    save_manifest(verified)
    print(f"\nmanifest: {TARGET / 'manifest.json'}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch the released AMLgentex datasets.")
    ap.add_argument("repos", nargs="*", metavar="REPO",
                    help=f"repos to fetch (default: all twelve; any of: {', '.join(EXPECTED)})")
    ap.add_argument("--verify-only", action="store_true",
                    help="check present files against the pinned sha256 and byte size; "
                         "download nothing")
    args = ap.parse_args()
    unknown = [r for r in args.repos if r not in EXPECTED]
    if unknown:
        ap.error(f"unknown repo(s) {unknown}; choose from {list(EXPECTED)}")
    return fetch(args.repos or list(EXPECTED), args.verify_only)


if __name__ == "__main__":
    raise SystemExit(main())
