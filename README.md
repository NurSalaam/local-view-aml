# Measuring and Remedying the Cost of the Local View in Money-Laundering Detection

This repository holds the code behind the Master's thesis *Measuring and
Remedying the Cost of the Local View in Money-Laundering Detection*
(Industrial Engineering, Stellenbosch University). A bank sees only the
payments that touch its own accounts; the code measures how much detection
performance that local view costs each bank on three public synthetic
datasets, splits the cost into its components, tests whether a bank can rank
itself by the cost from its own ledger, and tests whether borrowing another
bank's detector recovers it. The repository ships code only: the raw data are
fetched by the scripts below, and every Run, cache and figure is rebuilt from
them.

## Requirements

- macOS on Apple silicon (arm64). The lock file `uv.lock` resolves for
  `darwin`/`arm64` only.
- Python 3.11 and [uv](https://docs.astral.sh/uv/).
- `git` and `git-lfs`, for the TransXion fetch.
- Kaggle API credentials (`~/.kaggle/access_token`, or `KAGGLE_USERNAME` and
  `KAGGLE_KEY`), for the AMLworld fetch.
- About 27 GB of disk for the raw data, and roughly as much again for the
  derived caches under `data/interim/` and the Run log under `experiments/`.

The study ran on an Apple M4 Pro with 24 GB of memory. The launchers' default
budgets (`--budget 14 --machine-gb 24`) match that machine.

## Set-up

```sh
uv sync
```

`uv sync` installs the package `m` (from `src/m`) with its development group,
which supplies the Kaggle CLI and pytest.

Every Run records the commit it ran from in its `fingerprint.json`, and the
scripts of Experiments 1, 2 and 4 and `facts.py` refuse to start from a working
tree with uncommitted changes. Run the pipeline from a clean clone. The
execution settings (`M_TORCH_THREADS=4`, `M_LGB_THREADS_GLOBAL=10`,
`M_LGB_THREADS_SOLO=2`, `M_GNN_WORKERS=0`, `M_DEVICE=cpu`) are the defaults
and are recorded in the Run configs; the same scripts and the GNN
selection refuse other values. The anchor check (`checks_anchor_gnn.py`) is
the exception: it trains with 2 loader workers (`M_GNN_WORKERS=2`) and 10
torch threads, the settings of the study's anchor Runs, and picks the fastest
available device unless `M_DEVICE=cpu` is set, as it was for those Runs.

## Data

Each fetcher downloads into a fixed directory under `data/raw/`, checks every
file against the sha256 and byte size pinned in the script (the digests of the
files the study read), and writes a `manifest.json` beside the files recording
the source and each verified file's sha256 and byte size. AMLgentex and
TransXion are fetched at pinned revisions; Kaggle serves only its current
version of AMLworld, so the pinned digests are what detect an upstream change.
A re-run fetches only what is missing or does not match; `--verify-only`
checks without downloading.

```sh
uv run python scripts/fetch_ibm_aml.py HI-Small LI-Small HI-Medium LI-Medium
uv run python scripts/fetch_amlgentex_released.py
uv run python scripts/fetch_transxion.py
```

| Dataset | Source | Size | Destination |
|---|---|---|---|
| AMLworld (Altman et al., 2024) | Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` | ~7.5 GB for the four sets | `data/raw/ibm-aml/` |
| AMLgentex (Östman et al., 2025) | Twelve Hugging Face repositories under `huggingface.co/AMLGentex`, each pinned to a revision in the script | ~20.5 GB | `data/raw/amlgentex/<name>/` |
| TransXion (Chen et al., 2026) | `github.com/chaos-max/TransXion` at commit `53932595c37c23b9f55ea5ddf5984e4d57b88369` (`tx.csv` in Git LFS) | ~250 MB | `data/raw/transxion/` |

## Pipeline

Each stage writes Runs under `experiments/<directory>/runs/<run_id>/`
(`config.json`, `fingerprint.json`, `metrics.json`, `status.json`,
`artifacts/`). A unit is done when a complete Run with its config hash exists,
so every command below is safe to re-run: it skips finished work and resumes
after a crash. Later stages find earlier Runs by config and check name, never
by Run ID, so the pipeline runs from an empty `experiments/`. Most scripts
take `status`, which reports what is finished; the launchers also take `plan`.
The launchers write their worker logs under `logs/<experiment>/`.

The stages run in this order; each reads the Runs of the stages above it.

1. **Global-sight feature tables.** Optional: later stages build any missing
   table on first use, under `data/interim/features/`.

   ```sh
   uv run python scripts/extract.py HI-Small LI-Small HI-Medium LI-Medium transxion sweden_100K_easy
   ```

2. **Data checks** (`experiments/checks`). The censuses, the frame probe, the
   placebo, and the GNN anchor on HI-Small that the GNN selection's gate reads.

   ```sh
   uv run python scripts/checks_census_amlworld.py
   uv run python scripts/checks_census_transxion.py
   uv run python scripts/checks_census_amlgentex.py
   uv run python scripts/checks_frame_probe.py
   uv run python scripts/checks_placebo.py
   uv run python scripts/checks_anchor_gnn.py HI-Small
   ```

3. **GNN selection** (`experiments/select_gnn`, about 10 h). The gate,
   the screen of candidates 2 to 6, the commit of the two finalists at two
   further seeds, and the verdict. Experiment 1 adopts the winner's Runs as
   HI-Small's GNN global view, and stops if the winner's configuration
   differs from its own GNN.

   ```sh
   uv run python scripts/select_gnn.py launch gate
   uv run python scripts/select_gnn.py gate
   uv run python scripts/select_gnn.py launch screen
   uv run python scripts/select_gnn.py launch commit
   uv run python scripts/select_gnn.py verdict
   ```

4. **Experiment 1: the shortfall at each bank** (`experiments/exp1`, about
   44 h). The per-bank ledgers, the two gates, and every solo and global Run
   of the three families on all seventeen datasets; then the shortfall Runs
   and the verdict; then the shuffled-bank-label control and the further
   readings.

   ```sh
   uv run python scripts/exp1.py launch
   uv run python scripts/exp1_analysis.py all
   uv run python scripts/exp1_analysis.py verdict
   uv run python scripts/exp1_shuffle.py launch
   uv run python scripts/exp1_readings.py all
   ```

5. **Experiment 2: training examples or sight** (`experiments/exp2`, about
   11 h). The consortium cells, the examples dial and the own-pool
   network-sight cell; the copy-weighted C-global cell, which the launcher
   does not plan, is run by name; then the analyses and the four examples
   readings.

   ```sh
   uv run python scripts/exp2.py launch
   uv run python scripts/exp2.py run gnn_cglobaldup:HI-Medium:0 gnn_cglobaldup:HI-Medium:1 gnn_cglobaldup:HI-Medium:2
   uv run python scripts/exp2_analysis.py all
   uv run python scripts/exp2_analysis.py sightcheck
   uv run python scripts/exp2_analysis.py secondpath tree
   uv run python scripts/exp2_analysis.py secondpath gnn
   uv run python scripts/exp2_analysis.py secondpath blind
   uv run python scripts/exp2_examples.py all
   ```

6. **Experiment 3: rules on the ledger** (`experiments/exp3`). The descriptor
   gate, the descriptors and the selection, then every stage of the
   experiment.

   ```sh
   uv run python scripts/exp3_descriptors.py
   uv run python scripts/exp3.py
   ```

7. **Experiment 4: borrowing another bank's detector** (`experiments/exp4`).
   The lenders, the GNN identity gate and the transfer cells, then the policy
   Runs and the verdict. The analysis reads Experiment 2's components and
   examples Runs.

   ```sh
   uv run python scripts/exp4.py launch
   uv run python scripts/exp4_analysis.py all
   ```

8. **Facts** (`experiments/facts`). See [Facts](#facts).

   ```sh
   uv run python scripts/facts.py all
   ```

9. **Figures.** See [Figures](#figures).

   ```sh
   for f in figures/*.py; do uv run python "$f"; done
   ```

Stages not timed above each took under an hour.

## Where each part of the thesis comes from

| Thesis | Scripts | Run directory |
|---|---|---|
| Ch. 3, Data (censuses, never-seen-counterparty cue) | `checks_census_amlworld.py`, `checks_census_transxion.py`, `checks_census_amlgentex.py`, `checks_frame_probe.py` | `checks` |
| Ch. 3, The measurement (Figure of the four scenarios) | `figures/meth_scenarios.py` (reads Experiment 1's HI-Small ledgers) | none |
| Ch. 3, The detectors (measured scores beside the published ones) | `exp1.py`, `exp1_analysis.py` | `exp1` |
| Ch. 3, Checks on the measurement (placebo) | `checks_placebo.py` | `checks` |
| Ch. 4, Experiment 1 | `exp1.py`, `exp1_analysis.py`, `exp1_shuffle.py`, `exp1_readings.py`, `facts.py exp1` | `exp1`, `facts` |
| Ch. 4, Experiment 2 | `exp2.py`, `exp2_analysis.py`, `exp2_examples.py`, `facts.py exp2`, `facts.py exp2_examples` | `exp2`, `facts` |
| Ch. 5, Experiment 3 | `exp3_descriptors.py`, `exp3.py` | `exp3` |
| Ch. 6, Experiment 4 | `exp4.py`, `exp4_analysis.py`, `facts.py exp4`, `facts.py thesis` | `exp4`, `facts` |
| Ch. 7 and 8, abstract | `facts.py` (all blocks) | `facts` |
| App. A, Data | `fetch_ibm_aml.py`, `fetch_amlgentex_released.py`, `fetch_transxion.py` | none (manifests under `data/raw/`) |
| App. B, Detectors (tree columns; GNN configuration search) | `src/m/features.py`, `src/m/grid.py`; `checks_anchor_gnn.py`, `select_gnn.py` | `checks`, `select_gnn` |
| App. C, Methods and checks for the shortfall | `exp1_analysis.py verdict`, `exp1_readings.py`, `exp2_analysis.py` (`sightcheck`, `secondpath`, `dial`), `exp2_examples.py` | `exp1`, `exp2` |
| App. D, Methods and checks for ranking the banks | `exp3_descriptors.py`, `exp3.py` | `exp3` |

## Facts

`scripts/facts.py` computes the derived numbers the thesis quotes from the
recorded Runs; nothing trains. Each block is one `facts` Run under
`experiments/facts`, keyed by short names grouped by the table, figure or
paragraph that quotes them: `exp1`, `exp2`, `exp2_examples`, `exp4` and
`thesis` (numbers other chapters quote from the cached data). The `thesis`
block reads the `exp2` block, which `all` runs first.

```sh
uv run python scripts/facts.py all
uv run python scripts/facts.py exp1
uv run python scripts/facts.py status
```

## Figures

Each script in `figures/` draws one data figure of the thesis and writes
`figures/<name>.pdf` beside itself. Where a script recomputes a number a Run
recorded, it asserts that the two agree.

| Script | Figure | Reads |
|---|---|---|
| `meth_scenarios.py` | Ch. 3: the four scenarios over one neighbourhood of HI-Small | Experiment 1 ledgers |
| `res_shortfalls.py` | Ch. 4: every HI-Medium bank's shortfall with its interval | `exp1` shortfall |
| `res_size.py` | Ch. 4: the shortfall against the bank's size | `exp1` shortfall, shortfall_shuffled |
| `res_components.py` | Ch. 4: Experiment 2's split of the shortfall | `exp2` components |
| `res_reduction.py` | Ch. 4: the reduction of confirmed cases at 20 banks | `exp2` dial |
| `res_rules.py` | Ch. 5: the out-of-fold rank correlation of every rule | `exp3` selection, ceiling |
| `res_borrowed.py` | Ch. 6: the gain from borrowing against the shortfall | `exp4` policy |
| `app_null.py` | App. C: the null check of Cochran's Q | `exp1` shortfall |
| `app_rho.py` | App. C: how alike the two families rank the banks | `exp1` shortfall, verdict |
| `app_level100.py` | App. C: the score with 100 confirmed cases | `exp2` dial |
| `app_partial.py` | App. C: a six-bank example of the partial rank correlation | constructed values |
| `app_oof.py` | App. D: a six-bank example of the out-of-fold prediction | constructed values |

## Tests

```sh
uv run pytest
```

The tests need no Runs. One test in `tests/test_exp1.py` imports
`scripts/exp1.py`, which reads the AMLgentex manifest on import, so it needs
`fetch_amlgentex_released.py` to have run; one test in `tests/test_gnn_v2.py`
is skipped until the HI-Small feature table exists.

## Repository layout

```
src/m/            the library: Run log, statistics, features, GNN, descriptors, rules
src/m/worlds/     raw-data loaders, one canonical payments table per dataset
scripts/          fetchers, checks, experiments, analyses and facts
figures/          figure scripts; the PDFs they write are not tracked
tests/            unit tests
data/             raw data and derived caches (created by the scripts, not tracked)
experiments/      the Run log (created by the scripts, not tracked)
logs/             launcher worker logs (not tracked)
```

## Licence

MIT. Copyright (c) 2026 Nur Salaam. See `LICENSE`.
