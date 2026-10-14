# Measuring and Remedying the Cost of the Local View in Money-Laundering Detection

This repository holds the code behind the Master's thesis *Measuring and
Remedying the Cost of the Local View in Money-Laundering Detection*
(Industrial Engineering, Stellenbosch University). A bank sees only its
ledger, the payments that touch its own accounts. The thesis measures each
bank's *shortfall*, the cost of that local view: the average precision on the
bank's test set of a detector with the global view, which sees every bank's
payments and learns from every bank's confirmed cases, less that of the bank's
own detector. The measurements use six synthetic datasets from three
generators (AMLworld, TransXion and AMLgentex). Four experiments answer the
thesis's three research questions:

- **RQ1.** How large is the shortfall at each bank, how does it vary across
  banks, and how is it divided between few confirmed cases and limited sight
  of the network? Experiment 1, *The shortfall at each bank*, and
  Experiment 2, *Training examples or sight* (Chapter 4).
- **RQ2.** Can label-free properties of a bank's own ledger rank the banks by
  their shortfall? Experiment 3, *Rules on the ledger* (Chapter 5).
- **RQ3.** Can a bank with few confirmed cases reduce its shortfall by
  borrowing another bank's fitted detector, with no payment record crossing
  between banks? Experiment 4, *Borrowing another bank's detector*
  (Chapter 6).

The repository ships code only. The scripts below download the raw data,
check every file against the digest of the copy the study read, and rebuild
every Run, cache and figure from it.

## Requirements

- macOS on Apple silicon (arm64), the platform the study ran on. The lock
  file `uv.lock` is required to install there. It also pins wheels for Linux
  and Windows on x86_64, where the code has not been run.
- Python 3.11 and [uv](https://docs.astral.sh/uv/).
- `git`, since every Run records the commit it ran from, and `git-lfs`, for
  the TransXion fetch.
- Kaggle API credentials (`~/.kaggle/access_token`, or `KAGGLE_USERNAME` and
  `KAGGLE_KEY`), for the AMLworld fetch.
- About 8 GB of disk for the raw data, about 13 GB for the derived caches
  under `data/interim/` and about 4.5 GB for the Run log under `experiments/`.

Every experiment ran on the CPU of one Apple M4 Pro computer with 24 GB of
memory. The launchers' default budgets (`--budget 14 --machine-gb 24`) match
that machine.

## Set-up

```sh
git clone https://github.com/NurSalaam/local-view-aml.git
cd local-view-aml
uv sync
```

`uv sync` installs the package `m` (from `src/m`) with its development group,
which supplies the Kaggle CLI, pytest and ruff.

Every Run records the commit it ran from in its `fingerprint.json`, so a
script refuses to record a Run outside a git clone; a ZIP download carries no
commit. The scripts of Experiments 1, 2 and 4 and `facts.py` also refuse a
working tree with uncommitted changes, so run the pipeline from a clean clone.

The execution settings (`M_TORCH_THREADS=4`, `M_LGB_THREADS_GLOBAL=10`,
`M_LGB_THREADS_SOLO=2`, `M_DEVICE=cpu`) are the defaults and are recorded in
the Run configs; `exp1.py`, `exp2.py`, `exp4.py`, `exp1_analysis.py`,
`exp1_readings.py` and `exp2_analysis.py` refuse other values. Experiments 1
and 4 fix the GNN's loader workers at 0, and the GNN selection refuses any
`M_GNN_WORKERS`, `M_TORCH_THREADS` or `M_DEVICE` other than its defaults (0, 4
and `cpu`).

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
| AMLworld (Altman et al., 2024) | Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`: the four payments files and `HI-Medium_Patterns.txt` | ~7.1 GB | `data/raw/ibm-aml/` |
| AMLgentex (Östman et al., 2025) | `tx_log.csv` of the Hugging Face repository `huggingface.co/datasets/AMLGentex/sweden_100K_easy`, pinned to a revision in the script | ~634 MB | `data/raw/amlgentex/sweden_100K_easy/` |
| TransXion (Chen et al., 2026) | `github.com/chaos-max/TransXion` at commit `53932595c37c23b9f55ea5ddf5984e4d57b88369` (`data/tx.csv`, in Git LFS) | ~250 MB | `data/raw/transxion/` |

## Pipeline

Each stage writes Runs under `experiments/<directory>/runs/<run_id>/`
(`config.json`, `fingerprint.json`, `metrics.json`, `status.json`,
`artifacts/`). A unit is done when a complete Run with its config hash exists,
so every command below is safe to re-run: it skips finished work and resumes
after a crash. Later stages find earlier Runs by config and check name, never
by Run ID, so the pipeline runs from an empty `experiments/`. Most scripts
take `status`, which reports what is finished or names the earlier stage that
has yet to run; the launchers also take `plan`. Every script in the stages
below prints its usage with `--help`. The launchers write their worker logs
under `logs/<experiment>/`.

The stages run in this order; each reads the Runs of the stages above it.
Times are elapsed hours on the study's machine, from the start and finish
times its Runs recorded; the stages without a time were not timed.

1. **Feature tables under network sight.** Optional: later stages build any
   missing table on first use, under `data/interim/features/`.

   ```sh
   uv run python scripts/extract.py
   ```

2. **Data checks** (`experiments/checks`). The census of each dataset, the
   never-seen-counterparty cue and the placebo.

   ```sh
   uv run python scripts/checks_census_amlworld.py
   uv run python scripts/checks_census_transxion.py
   uv run python scripts/checks_census_amlgentex.py
   uv run python scripts/checks_frame_probe.py
   uv run python scripts/checks_placebo.py
   ```

3. **GNN selection** (`experiments/select_gnn`, about 3 h). The search for
   the GNN family's configuration: the screen of the six candidates at one
   seed, the two finalists at two further seeds, and the verdict. Experiment
   1 adopts the winner's Runs as HI-Small's detector with the global view in
   the GNN family, and stops if the winner's configuration differs from its
   own.

   ```sh
   uv run python scripts/select_gnn.py launch screen
   uv run python scripts/select_gnn.py launch commit
   uv run python scripts/select_gnn.py verdict
   ```

4. **Experiment 1: The shortfall at each bank** (`experiments/exp1`, about
   17 h). The per-bank ledgers; two checks of the code paths (the streamed
   tree table must reproduce the in-memory one, and Experiment 1's GNN
   inputs must reproduce the GNN selection's scores on HI-Small); the bank's
   own detector and the detector with the global view, in the tree and GNN
   families, on HI-Medium, HI-Small and the two controls; the detector with
   the global view alone on LI-Small and LI-Medium; and the blind tree, as the
   bank's own detector at every qualifying bank of HI-Medium and with the
   global view on HI-Medium and the AMLgentex dataset. Then the shortfall
   Runs and the verdict on hypothesis 1, which together hold the robustness
   readings, and two further readings: τ about the line on size, and each
   scenario scored at its own best configuration. `scores` prints the measured
   scores of the table of measured and published scores; the published scores
   are quoted from Altman et al. (2024) and Chen et al. (2026).

   ```sh
   uv run python scripts/exp1.py launch
   uv run python scripts/exp1_analysis.py all
   uv run python scripts/exp1_analysis.py verdict
   uv run python scripts/exp1_readings.py all
   uv run python scripts/exp1_analysis.py scores
   ```

5. **Experiment 2: Training examples or sight** (`experiments/exp2`, about
   7 h). The scenarios (ledger, qualifying), (network, qualifying) and
   (network, own) on HI-Medium, (network, qualifying) in the GNN family with
   the count held equal, and the case reduction; then the decomposition of
   the shortfall along both paths, the value of sight with the count held
   equal, the case reduction's reading, and the reading of confirmed cases
   across banks.

   ```sh
   uv run python scripts/exp2.py launch
   uv run python scripts/exp2_analysis.py all
   uv run python scripts/exp2_examples.py all
   ```

6. **Experiment 3: Rules on the ledger** (`experiments/exp3`, under an hour).
   The check on the label-free properties, the properties of every qualifying
   bank of HI-Medium and HI-Small, and the out-of-fold scores of the sixteen
   rules on them, whose comparison with the scale rule gives the verdict on
   hypothesis 2 (`outperform_the_scale_rule` and `hypothesis_2` in each
   `selection` Run); then the check on the shortfall tables read, the
   partial rank correlations, the scale rule applied to HI-Small, the rules
   that use labels, and the rule on scale and confirmed cases applied to
   HI-Small.

   ```sh
   uv run python scripts/exp3_descriptors.py
   uv run python scripts/exp3.py
   ```

7. **Experiment 4: Borrowing another bank's detector** (`experiments/exp4`,
   under an hour). The lenders, the check that the transfer path reproduces
   each GNN lender's own scores, and the transfer cells on HI-Medium and
   HI-Small; then one policy Run per dataset and family (the choice, the
   comparison with the own detector and the gain at each bank) and the
   verdict on hypothesis 3.
   The analysis reads Experiment 2's `components` and `examples` Runs.

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

## Code identifiers and thesis terms

The code keeps short identifiers in check names, config keys and table
columns. Each stands for one thesis term.

| In the code | In the thesis |
|---|---|
| `solo` | (ledger, own): the bank's own detector |
| `global` | (network, all): the detector with the global view |
| `setup` | the scenario |
| `clocal`, `cglobal`, `nown` | (ledger, qualifying), (network, qualifying), (network, own) |
| `gnn_cglobaldup`, `sightcheck` | (network, qualifying) with the count held equal, and the value of sight read with it |
| `components`, `secondpath` | the decomposition of the shortfall along path 1 and along path 2 |
| `examples`, `sight`, `pool_extension`, `total` (components) | the value of training examples with sight held at the ledger, the value of sight with the qualifying banks' training examples in place, extending the pool to every bank, and the shortfall |
| `sight_own`, `examples_network`, `interaction` | the value of sight at the own training examples, the value of training examples at network sight, and the interaction |
| `dial` | the case reduction |
| `examples` (check) | confirmed cases across banks |
| `never_seen_split`: `never_seen`, `rest` | the two parts of each test set: the first part, whose counterparty appears nowhere in the bank's ledger before the test window, and the remainder |
| `quartile` | a quarter of the qualifying banks by their number of confirmed cases |
| `roster` | the qualifying banks |
| `train_pos`, `test_pos` | confirmed cases; laundering payments in the test set |
| `delta`, `se` | the shortfall and its error bar |
| `blind` | the blind tree |
| `own_best` | each scenario scored at its own best configuration |
| `residual_after_size` | τ about the line on size |
| `descriptors` | the label-free properties |
| `scale`, `baseline` | the scale rule |
| `selection` | the sixteen rules on label-free properties, scored out of fold |
| `ceiling` | the rules that use labels |
| `bank_facing` | the rule on scale and confirmed cases |
| `policy`, `transfer`, `lender` | the choice and the comparison with the own detector, a borrowed detector's scores, a lender |

## Where each part of the thesis comes from

Each row names the scripts that write the Runs a section reads, and the
`facts.py` blocks, with their keys, that compute the numbers it quotes.

| Thesis | Producers | Run directory |
|---|---|---|
| Ch. 3 §3.1, Data: the conventions, the census table, the shares of payments within one bank and the never-seen-counterparty cue | `checks_census_amlworld.py`, `checks_census_transxion.py`, `checks_census_amlgentex.py` (payments, bank identifiers, qualifying banks, laundering payments per bank in the test set, the AMLgentex rows dropped, the shares of payments within one bank); `checks_frame_probe.py` (the never-seen-counterparty cue); `facts.py exp1` (`roster`: the payments in a qualifying bank's ledger and its confirmed cases); `facts.py thesis` (`census`: the laundering rates; `over_chance_range`; `rail`: the payment rail) | `checks`, `exp1`, `facts` |
| Ch. 3 §3.2.1 to §3.2.3, The measurement: the shortfall, the four scenarios and the row rule | `figures/meth_scenarios.py` (the figure of the four scenarios and the counts its caption quotes, from Experiment 1's HI-Small ledgers); `facts.py exp1` (`roster`: the laundering payments in a test set) | `exp1`, `facts` |
| Ch. 3 §3.2.4, Pairing and error bars | `exp1_analysis.py shortfall`, with `src/m/stats.py` (each bank's shortfall, its error bar and the common laundering rate, as in the section's algorithm) | `exp1` |
| Ch. 3 §3.3, The detectors | `src/m/features.py` (the tree family's 101 columns); `src/m/gnn.py` (the GNN family); `src/m/grid.py` and `exp1.py` (`GNN`) (the table of hyperparameters); `facts.py thesis` (`always_zero`: the AMLgentex dataset's columns that are always zero); `select_gnn.py` (the choice of the GNN configuration) | `exp1`, `select_gnn`, `facts` |
| Ch. 3 §3.4, Checks on the measurement | `exp1_analysis.py scores` (the measured scores of the table of measured and published scores: `test_f1_x100` and `test_ap` of each detector with the global view, the tree family at its configuration of highest validation average precision, the GNN family as the mean over its three seeds, HI-Small's GNN Runs from the GNN selection, and the blind tree on the AMLgentex dataset; the published scores are quoted from Altman et al. and Chen et al.); `checks_placebo.py` (the placebo) | `exp1`, `select_gnn`, `checks` |
| Ch. 4 §4.1, Experiment 1 | `exp1.py`; `exp1_analysis.py` (`shortfall`, `verdict`); `exp1_readings.py` (`paired`: each scenario scored at its own best configuration, §4.1.2; `residual`: τ about the line on size, §4.1.3); `facts.py exp1` (`spread`, `per_seed`, `tau`, `readings`, `size`, `controls`, `roster`) | `exp1`, `facts` |
| Ch. 4 §4.2, Experiment 2 | `exp2.py`; `exp2_analysis.py` (`components`, `secondpath`, `sightcheck`, `dial`); `exp2_examples.py` (`examples`: the shortfall by confirmed cases and the partial rank correlation); `facts.py exp2` (`components` with its `quartiles`, `secondpath`, `sightcheck`, `qualifying`, `never_seen_split`, `pool`, `copies`, `banks`, `placebo`, `reduction`); `facts.py exp1` (`roster`: ρ between confirmed cases and size); `facts.py thesis` (`quarters`: the weighted means of the table of the shortfall by quarter; `largest`: the fewest confirmed cases among the thirty qualifying banks with the most payments, §4.2.3) | `exp1`, `exp2`, `facts` |
| Ch. 5 §5.1, Experiment 3 | `exp3_descriptors.py` (the check on the properties; `descriptors`; `selection`: the out-of-fold ρ of the sixteen rules, each one's difference from the scale rule, the rules that outperform it, and the verdict on hypothesis 2 as its `hypothesis_2`); `exp3.py` (`inputs`: the check on the shortfall tables read; `screen`: each property's rank correlation with the shortfall, raw and with scale held fixed; `ceiling`: the rule on scale and confirmed cases out of fold; `holdout` and `bank_facing`: the scale rule and the rule on scale and confirmed cases applied to HI-Small) | `exp3` |
| Ch. 6 §6.1, Experiment 4 | `exp4.py` (the lenders, the check that the transfer path reproduces each GNN lender's own scores, and the transfer cells); `exp4_analysis.py` (`policy`: the choice, the comparison with the own detector and the gain at each bank, the weighted means and their intervals, the medians by quarter and the validation counts; `verdict`); `facts.py exp4` (`recovery`: the gain as a share of the shortfall and of the value of training examples, §6.1.2); `facts.py thesis` (`lenders` and `line`, §6.1.1; `quarters`: the table of the gain by quarter, §6.1.3; `single_lender` and `every_bank_borrows`: the readings with a single lender and in which every bank borrows, §6.1.4) | `exp4`, `facts` |
| Ch. 7 and 8, and the abstract | the numbers of the rows above, as `facts.py exp1`, `exp2`, `exp4` and `thesis` record them; `facts.py thesis` (`largest`: the qualifying banks among the thirty with the most payments, by each measure of size, §7.3) | `facts` |
| App. A, Data | `fetch_ibm_aml.py`, `fetch_amlgentex_released.py`, `fetch_transxion.py` (the files read and their digests); `checks_census_amlgentex.py` (the AMLgentex dataset's twelve banks and 112 daily steps); `facts.py thesis` (`census`: the AMLgentex dataset's laundering rate) | none (manifests under `data/raw/`), `checks`, `facts` |
| App. B.1, The tree family's columns | `src/m/features.py` | none |
| App. B.2, The search for the GNN family's configuration | `select_gnn.py` (the six candidates, the two finalists at three seeds and the choice) | `select_gnn` |
| App. C.1, The three tests | `exp1_analysis.py` (`shortfall`: the null check, leave one out and the tests at each seed; `verdict`: the GNN family's null-check 95th percentile on HI-Medium and the AMLgentex dataset, redrawn 200 times); `facts.py exp1` (`spread`, `per_seed`, `controls`); `figures/app_null.py` | `exp1`, `facts` |
| App. C.2, The standard deviation of the true shortfalls | `exp1_readings.py residual` (τ about the line on size); `facts.py exp1` (`tau`; `spread`: the standard deviation of the shortfalls and the median error bar, and `tau_terms`: the two terms of τ and the standard deviation of the shortfalls weighted by w_B) | `exp1`, `facts` |
| App. C.3, The agreement between the two families | `exp1_analysis.py verdict` (ρ, its 5th percentile and its 95 % interval over 1,000 redraws of the banks); `figures/app_rho.py` | `exp1` |
| App. C.4, The robustness readings | `exp1_analysis.py` (`shortfall`: `common_rate`, and the shortfall on HI-Small; `verdict`: `robustness`, the factors on the error bars on HI-Medium and HI-Small); `facts.py exp1` (`tau`: `common_rate` and `HI-Small`; `roster`: the common laundering rate) | `exp1`, `facts` |
| App. C.5, Scoring at each scenario's own best configuration | `exp1_readings.py paired` (the tree family and the blind tree: the weighted means, the changes of sign and ρ); `facts.py exp1` (`readings`: `own_best`, the gain from choosing each scenario's configuration) | `exp1`, `facts` |
| App. C.6, The decomposition of the shortfall | `exp2_analysis.py` (`components`, `secondpath`, `sightcheck`); `facts.py exp2` (`components`, `banks`, `secondpath`, `sightcheck`, `copies`, `placebo`, `never_seen_split`, `qualifying`); `facts.py thesis` (`interaction_tree`: the interaction formed three ways) | `exp2`, `facts` |
| App. C.7, The case reduction | `exp2.py` (the reduced fits); `exp2_analysis.py dial`; `facts.py exp2` (`reduction`); `figures/app_level100.py` | `exp2`, `facts` |
| App. C.8, The partial rank correlation | `exp2_examples.py` (`examples`: the partial rank correlations and their shuffles); `facts.py exp1` (`roster`: ρ between training-window payments and size); `figures/app_partial.py` (constructed values) | `exp1`, `exp2`, `facts` |
| App. D.1, The properties | `src/m/descriptors.py`; `exp3_descriptors.py` (the check on the properties; `descriptors`: each property's range); `exp3.py screen` (each property's rank correlation with the shortfall, raw and with scale held fixed); `facts.py thesis` (`shared_accounts`: the account numbers that occur at two banks) | `exp3`, `facts` |
| App. D.2, The out-of-fold ρ | `exp3_descriptors.py` (`selection`: each rule's out-of-fold ρ, its ten values and the fit without weights); `figures/app_oof.py` (constructed values) | `exp3` |
| App. D.3, The comparison with the scale rule | `exp3_descriptors.py` (`selection`: each rule's difference from the scale rule, its 5th percentile and its standard deviation over 1,000 redraws of the banks) | `exp3` |
| App. D.4, The rules that use labels | `src/m/ceiling.py` (the six quantities computed from labels); `exp3.py` (`ceiling`: the nine rules, and the homophily of cases with the rail left in; `holdout` and `bank_facing`: the scale rule and the rule on scale and confirmed cases, fitted on the 107 banks, with their coefficients, and applied to HI-Small); `facts.py thesis` (`typology`: the share of confirmed cases that the record of injected typologies names; `rho_payments_active_accounts`) | `exp3`, `facts` |

A few quoted values are read directly from a named artefact rather than
computed by a facts block: the fit times on HI-Medium and its counts of
training-window payments and copies (the `seconds` and `rows` of Experiment
1's `gnn_global` and `tree_global` Runs on HI-Medium); banks 0272142 and
008904, and the shortfalls and error bars of banks 0143079 and 0293268
(Experiment 1's `shortfall` Run on HI-Medium in the tree family, its
`banks.csv` and `roster.csv`); bank 0143079's scores of 0.140 and 0.957
(Experiment 2's `components` Run in the tree family, `banks.csv`); bank 020's
scores at each level of the case reduction (Experiment 2's `dial` Runs, their
`banks` block); the boundary share of exactly 1 at 51 banks (the `descriptors`
Run of HI-Medium, `descriptors.csv`); the ten values of the scale rule's
out-of-fold ρ (the `selection` Runs, `candidates.csv`, `score_r0` to
`score_r9`); and the HI-Small lenders left out of the GNN family's choice (the
`excluded` block of Experiment 4's `transfer_gnn` Runs on HI-Small).

## Facts

`scripts/facts.py` computes the derived numbers the thesis quotes from the
recorded Runs; nothing trains. Each block is one `facts` Run under
`experiments/facts`, keyed by short names grouped by the table, figure or
paragraph that quotes them: `exp1`, `exp2`, `exp4` and `thesis` (numbers
other chapters quote from the censuses and the cached data). The `thesis`
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
| `meth_scenarios.py` | Ch. 3 §3.2.2: the four scenarios over one neighbourhood of HI-Small | Experiment 1 ledgers |
| `res_shortfalls.py` | Ch. 4 §4.1.2: the shortfall at each bank of HI-Medium | `exp1` shortfall |
| `res_size.py` | Ch. 4 §4.1.3: the shortfall and the bank's size | `exp1` shortfall |
| `res_components.py` | Ch. 4 §4.2.2: the value of training examples and of sight at each bank | `exp2` components |
| `res_reduction.py` | Ch. 4 §4.2.4: the score with fewer confirmed cases | `exp2` dial |
| `res_rules.py` | Ch. 5 §5.1.2: the out-of-fold ρ of each rule | `exp3` selection, ceiling |
| `res_borrowed.py` | Ch. 6 §6.1.2: the gain from borrowing and the shortfall at each bank | `exp4` policy |
| `app_null.py` | App. C.1: the null check on two combinations | `exp1` shortfall |
| `app_rho.py` | App. C.3: the two families' rankings of HI-Medium's banks | `exp1` shortfall, verdict |
| `app_level100.py` | App. C.7: the score with 100 confirmed cases | `exp2` dial |
| `app_partial.py` | App. C.8: a partial rank correlation on six banks | constructed values |
| `app_oof.py` | App. D.2: out-of-fold predictions on six banks | constructed values |

## Tests

```sh
uv run pytest
uv run ruff check .
```

The tests need no Runs and pass before any data are fetched; one test in
`tests/test_gnn.py` is skipped until the HI-Small feature table exists.

## Repository layout

```
src/m/            the library: Run log, statistics, features, GNN, label-free properties (descriptors.py), rules
src/m/worlds/     raw-data loaders, one canonical payments table per dataset
scripts/          fetchers, checks, experiments, analyses and facts
figures/          figure scripts; the PDFs they write are not tracked
tests/            unit tests
data/             raw data and derived caches (created by the scripts, not tracked)
experiments/      the Run log (created by the scripts, not tracked)
logs/             launcher worker logs (not tracked)
```

## Licence

MIT. Copyright (c) 2026 Nur Mogammad Salaam. See `LICENSE`.
