# Measuring and Remedying the Cost of the Local View in Money-Laundering Detection

This repository holds the code behind the Master's thesis *Measuring and
Remedying the Cost of the Local View in Money-Laundering Detection*
(Industrial Engineering, Stellenbosch University). A bank sees only its
ledger, the payments that touch its own accounts. The thesis measures the
*shortfall* of each bank, the cost of that local view: the average precision
on the test set of the bank of the detector with the global view, which sees
the payments of every bank and learns from the confirmed cases of every bank,
less that of the own detector of the bank. The measurements use six
synthetic datasets from three generators (AMLworld, TransXion and AMLgentex),
and the shortfall is also measured on the other eleven datasets that
AMLgentex released. Four experiments answer the three research questions of
the thesis:

- **RQ1.** How large is the shortfall at each individual bank, how does it
  vary across banks, and how is it divided between few confirmed cases and
  limited sight of the network? Experiment 1, *The shortfall at each bank*,
  and Experiment 2, *Training examples or sight* (Chapter 4).
- **RQ2.** Can properties of the ledger of a bank, computed without any of
  its labels, rank the banks by their shortfall? Experiment 3 (Chapter 5,
  *Ranking the banks by their shortfall*).
- **RQ3.** Can a bank with few confirmed cases reduce its shortfall by
  borrowing a detector fitted at another bank, with no payment record
  crossing between banks? Experiment 4 (Chapter 6, *The borrowed detector*).

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
- About 27 GB of disk for the raw data, of which the twelve AMLgentex
  datasets take 19.2 GB; about 20 GB for the derived caches under
  `data/interim/`; and about 6 GB for the Run log under `experiments/`.

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
| AMLworld (Altman et al., 2024) | Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`: the four payments files | ~7.1 GB | `data/raw/ibm-aml/` |
| AMLgentex (Östman et al., 2025) | `tx_log.csv` of each of the twelve Hugging Face repositories `huggingface.co/datasets/AMLGentex/<name>`, each pinned to a revision in the script; `sweden_100K_easy` (~634 MB) is the AMLgentex dataset of the thesis | ~19.2 GB | `data/raw/amlgentex/<name>/` |
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
   never-seen-counterparty cue and the placebo of the payment-only tree.

   ```sh
   uv run python scripts/checks_census_amlworld.py
   uv run python scripts/checks_census_transxion.py
   uv run python scripts/checks_census_amlgentex.py
   uv run python scripts/checks_frame_probe.py
   uv run python scripts/checks_placebo.py
   ```

3. **GNN selection** (`experiments/select_gnn`, about 3 h). The search for
   the configuration of the GNN family: the six candidates at one seed, the
   two that scored highest at two further seeds, and the choice. Experiment 1
   adopts the Runs of the chosen candidate as the detector with the global
   view of the GNN family on HI-Small, and stops if its configuration differs
   from its own.

   ```sh
   uv run python scripts/select_gnn.py launch screen
   uv run python scripts/select_gnn.py launch commit
   uv run python scripts/select_gnn.py verdict
   ```

4. **Experiment 1: The shortfall at each bank** (`experiments/exp1`, about
   17 h, and about 6 h more for the eleven other released AMLgentex
   datasets). The per-bank ledgers; two checks of the code paths (the
   streamed tree table must reproduce the in-memory one, and the GNN inputs
   of Experiment 1 must reproduce the scores of the GNN selection on
   HI-Small); the own detector and the detector with the global view, in the
   tree and GNN families, on HI-Medium, HI-Small, the two controls and the
   eleven other released AMLgentex datasets; the detector with the global
   view alone on LI-Small and LI-Medium; and the payment-only tree, as the
   own detector at every qualifying bank of HI-Medium and with the global
   view on each of the six datasets. Then the shortfall Runs; the verdict on
   hypothesis 1, judged in each family on the null check alone, with the
   readings beside it; and three further readings: τ measured from a
   straight line fitted to the shortfall on the logarithm of size, each
   scenario scored at its own best configuration, and the readings of the
   spread (each bank left out in turn, the null check of each robustness
   check and the largest factor on every error bar).

   ```sh
   uv run python scripts/exp1.py launch
   uv run python scripts/exp1_analysis.py all
   uv run python scripts/exp1_analysis.py verdict
   uv run python scripts/exp1_readings.py all
   ```

5. **Experiment 2: Training examples or sight** (`experiments/exp2`, about
   7 h). The scenarios (ledger, qualifying), (network, qualifying) and
   (network, own) on HI-Medium, (network, qualifying) in the GNN family with
   each payment fitted as often as under ledger sight, and the case
   reduction; then the decomposition of the shortfall in either order (pool
   enlarged first, sight widened first), the value of sight with that count
   held equal, the reading of the case reduction, and the reading of
   confirmed cases across banks.

   ```sh
   uv run python scripts/exp2.py launch
   uv run python scripts/exp2_analysis.py all
   uv run python scripts/exp2_examples.py all
   ```

6. **Experiment 3** (`experiments/exp3`, under an hour). The check on the
   label-free properties, the properties of every qualifying bank of
   HI-Medium and HI-Small, and the out-of-fold ρ of the sixteen regression
   models on them in the tree family, the GNN family and the payment-only
   tree, whose comparison with the scale model gives the verdict on
   hypothesis 2 (`outperform_the_scale_model` and `hypothesis_2` in each
   `selection` Run); then the check on the shortfall tables read, the
   partial rank correlations, the scale model applied to HI-Small, the two
   models that use labels, and the model on scale and confirmed cases
   applied to HI-Small; then the readings beside the selection: the models
   on one count of scale, the coefficients of the scale model, the share of
   recurring counterparties at other thresholds, the correlation matrix of
   the properties, and every model's difference from the scale model under
   four redraws.

   ```sh
   uv run python scripts/exp3_descriptors.py
   uv run python scripts/exp3.py
   uv run python scripts/exp3_readings.py all
   ```

7. **Experiment 4** (`experiments/exp4`, under an hour). The lenders, the
   check that the transfer path reproduces the scores of each GNN lender,
   and the transfer cells on HI-Medium and HI-Small; then one policy Run per
   dataset and family (the choice, the comparison with the own detector and
   the gain at each bank) and the verdict on hypothesis 3. The analysis reads
   the `components` and `examples` Runs of Experiment 2.

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
| `solo` | (ledger, own): the own detector |
| `global` | (network, all): the detector with the global view |
| `setup` | the scenario |
| `clocal`, `cglobal`, `nown` | (ledger, qualifying), (network, qualifying), (network, own) |
| `gnn_cglobaldup`, `sightcheck` | (network, qualifying) with each payment fitted as often as under ledger sight, and the value of sight read with that count held equal |
| `components`, `secondpath` | the decomposition of the shortfall with the pool enlarged first and with sight widened first |
| `examples`, `sight`, `pool_extension`, `total` (components) | the value of training examples with sight held at the ledger, the value of sight with the training examples of the qualifying banks in place, extending the pool to every bank, and the shortfall |
| `sight_own`, `examples_network`, `interaction` | the value of sight at the own training examples, the value of training examples at network sight, and the interaction |
| `qualifying` (`facts.py exp2`) | the qualifying-pool shortfall, M(network, qualifying) - M(ledger, own) |
| `dial` | the case reduction |
| `examples` (check) | confirmed cases across banks |
| `never_seen_split`: `never_seen`, `rest` | the two groups of each test set: the first group, whose counterparty appears nowhere in the ledger of the bank before the test window opens, and the second group |
| `quartile` | a quarter of the qualifying banks by their number of confirmed cases |
| `roster` | the qualifying banks |
| `train_pos`, `test_pos` | confirmed cases; laundering payments in the test set |
| `delta`, `se` | the shortfall and its error bar |
| `blind` | the payment-only tree |
| `own_best` | each scenario scored at its own best configuration |
| `residual_after_size` | τ measured from a straight line fitted to the shortfall on the logarithm of size |
| `spread_readings` | the readings of the spread: each bank left out in turn, the null check of each robustness check, the largest factor on every error bar, τ in the world with no real spread |
| `RELEASES` | the other eleven released AMLgentex datasets (Appendix C.9) |
| `descriptors` | the label-free properties |
| `scale`, `baseline` | the scale model (scale: the numbers of payments and of active accounts) |
| `selection` | the sixteen regression models on label-free properties, scored out of fold: the scale model, ten that add one property to scale, and five on one group without scale |
| `ceiling` | the models that use labels |
| `holdout`, `bank_facing` | the scale model and the model on scale and confirmed cases, fitted on HI-Medium and applied to HI-Small |
| `policy`, `transfer`, `lender` | the choice and the comparison with the own detector, the scores of a borrowed detector, a lender |

The module `src/m/rule.py` holds the regression models.

## Where each part of the thesis comes from

Each row names the scripts that write the Runs a section reads, and the
`facts.py` blocks, with their keys, that compute the numbers it quotes.
Chapters 1 and 2 report no number computed here, and Chapter 8 restates
numbers of the rows below.

| Thesis | Producers | Run directory |
|---|---|---|
| Ch. 3 §3.1, The experiments | none: Table 3.1 and the address of this repository | none |
| Ch. 3 §3.2.1, Selection and roles | `checks_census_amlgentex.py` (the laundering payments per bank in the test set of the AMLgentex dataset); `exp1_analysis.py shortfall` (`primary.null_check` of each released AMLgentex dataset: the two of the twelve whose spread passes the null check, as in App. C.9) | `checks`, `exp1` |
| Ch. 3 §3.2.2, Preparation of the data | `facts.py thesis` (`rail`: the fifteen accounts of the payment rail, its share of the payments and of the laundering payments, the payments between its accounts, the share it sends out and their formats, and its payments against those of the next largest bank identifier; `census`: the share of the laundering-labelled rows of the AMLgentex dataset that touch the source or sink account, and the laundering rate); `checks_census_amlgentex.py` (`raw_pseudo_row_share`: the rows dropped) | `checks`, `facts` |
| Ch. 3 §3.2.3, Training, validation and test sets | `facts.py thesis` (`qualifying`: the qualifying banks at thresholds of 10, 20, 30 and 50, the share of the payments and laundering payments they cover, the training-window over test-window laundering payments, and the banks with laundering in the test window and no confirmed case); `facts.py exp1` (`spread`: `se_median_in_band`, the median error bar at banks with 30 to 39 laundering payments in the test set) | `facts` |
| Ch. 3 §3.2.4, Census | `checks_census_amlworld.py`, `checks_census_transxion.py`, `checks_census_amlgentex.py` (payments, bank identifiers, qualifying banks, the shares of payments within one bank); `checks_frame_probe.py` (the never-seen-counterparty cue); `facts.py exp1` (`roster`: the payments in the ledger of a qualifying bank and its confirmed cases); `facts.py thesis` (`census`: the laundering rates; `over_chance_range`; `rail`: the rail measured as a bank, and its share of the payments and laundering payments in the test sets of the qualifying banks) | `checks`, `exp1`, `facts` |
| Ch. 3 §3.3.1 and §3.3.2, The shortfall and the four scenarios | `figures/meth_scenarios.py` (the figure of the four scenarios and the counts its caption quotes, from the HI-Small ledgers of Experiment 1); `facts.py exp1` (`roster`: the laundering payments in a test set) | `exp1`, `facts` |
| Ch. 3 §3.3.3, Error bars | `exp1_analysis.py shortfall`, with `src/m/stats.py` (the shortfall of each bank, its error bar and the common laundering rate, as in Algorithm 3.1) | `exp1` |
| Ch. 3 §3.4.1 to §3.4.3, The detectors and the search for the hyperparameters | `src/m/features.py` (the 101 columns of the tree family); `src/m/gnn.py` (the GNN family); `src/m/grid.py` and `exp1.py` (`GNN`) (Table 3.4); `select_gnn.py` (the six candidates); `facts.py thesis` (`always_zero`: the columns of the AMLgentex dataset that are always zero); `facts.py detectors` (`network_all.HI-Medium.tree_selected_f1_x100`: the F1 of the configuration validation selected on HI-Medium) | `exp1`, `select_gnn`, `facts` |
| Ch. 3 §3.4.4, Selected configurations and training | `select_gnn.py` (Table 3.5, GNN rows: `screen` and `commit` of the `verdict` Run); `facts.py detectors` (`network_all.<dataset>`: `tree_val_ap`, `tree_selected` and `tree_val_ap_spread` for Table 3.5, `tree_stopping` and `gnn_stopping` per dataset; `ledger_own_selection`: the configuration selected at each bank at (ledger, own); `stopping`: Table 3.6; `global_gnn_other_datasets`; `screen_epochs`: the epochs of the six candidates at their first seed; `median_bank`); `figures/meth_learning_curves.py` | `select_gnn`, `exp1`, `exp2`, `facts` |
| Ch. 3 §3.4.5, The payment-only tree | `exp1.py` (`blind_solo`, `blind_global`) | `exp1` |
| Ch. 3 §3.5, Checks on the measurement | `facts.py detectors` (`network_all.<dataset>.laundering_rate` and `internal`: Table 3.7, the payment-only tree and the tree family as median and range over the eight configurations, the GNN family as mean and range over its three seeds, and the ranges of the configurations that kept one tree and that reached the cap); `checks_placebo.py` (the placebo) | `exp1`, `select_gnn`, `checks`, `facts` |
| Ch. 4 §4.1, Experiment 1 | `exp1.py`; `exp1_analysis.py` (`shortfall`; `verdict`: the null check in each family, at each GNN seed and on the controls, the agreement between the families, the largest factor on every error bar and the null check with fresh draws); `exp1_readings.py` (`paired`: Table 4.1 and the changes of sign; `residual`: τ measured from the line on size; `spread`: the lowest I² with each bank left out, the null check at each removal, the robustness checks of Table 4.3, the two terms of τ and τ with no real spread, the banks with at least 100 confirmed cases, the partial correlations of the agreement, size in either window, the reading at configuration 7 and the ranges of the controls); `facts.py exp1` (`spread`, `per_seed`, `tau`, `readings`, `size`, `rates`, `controls`, `roster`); `figures/res_shortfalls.py`, `figures/res_size.py` | `exp1`, `facts` |
| Ch. 4 §4.2, Experiment 2 | `exp2.py`; `exp2_analysis.py` (`components`, `secondpath`, `sightcheck`, `dial`); `exp2_examples.py` (`examples`: the shortfall by confirmed cases and the partial rank correlation); `facts.py exp2` (`components` with its `quartiles`, `secondpath`, `sightcheck`, `qualifying`, `never_seen_split`, `pool`, `copies`, `banks`, `placebo`, `reduction` with `range_width` and `level5_below_range`); `facts.py exp1` (`roster`: ρ between confirmed cases and size); `facts.py thesis` (`quarters`: Table 4.6; `largest`: the fewest confirmed cases among the thirty qualifying banks with the most payments, §4.2.2); `figures/res_components.py`, `figures/res_reduction.py` | `exp1`, `exp2`, `facts` |
| Ch. 5 §5.1 and §5.2, Experimental design; the scale model and the properties | `exp3_descriptors.py` (the check on the properties; `descriptors`; `selection`: the out-of-fold ρ of the sixteen regression models, the difference of each from the scale model, its 5th and 95th percentiles, and the verdict on hypothesis 2); `exp3.py screen` (the rank correlation of each property with the shortfall, raw and with scale held fixed); `exp3_readings.py` (`properties`: the models on one count of scale, the recurring counterparties at five thresholds, the correlations between the properties and how far each varies; `uncertainty`: the redraws with every model fitted again); `facts.py exp1` (`roster`: ρ between the training-window and test-window payments); `facts.py thesis` (`rho_payments_active_accounts`); `figures/res_rules.py` | `exp1`, `exp3`, `facts` |
| Ch. 5 §5.3, The number of confirmed cases | `exp3.py` (`ceiling`: the two models that use labels out of fold; `holdout` and `bank_facing`: the scale model and the model on scale and confirmed cases applied to HI-Small, with their intervals); `exp3_readings.py uncertainty` (the 5th percentile with every model fitted again) | `exp3` |
| Ch. 5 §5.4, Answer to research question 2 | `exp3_readings.py uncertainty` (the comparison on the resampled shortfalls) | `exp3` |
| Ch. 6 §6.1, Experimental design | `exp4.py` (the lenders, the check that the transfer path reproduces the scores of each GNN lender, and the transfer cells, with the HI-Small lenders left out of the choice); `exp4_analysis.py policy` (`selection_signal`: the validation windows with at most ten laundering payments); `facts.py thesis` (`lenders`) | `exp4`, `facts` |
| Ch. 6 §6.2, The gain at the banks with the fewest confirmed cases | `exp4_analysis.py` (`policy`: the weighted mean gain, its interval and the median scores by quarter; `verdict`, with the repeat on HI-Small); `facts.py exp4` (`recovery`: the gain as a share of the shortfall and of the value of training examples); `facts.py thesis` (`every_bank_borrows.<family>.q1_intervals`); `figures/res_borrowed.py` | `exp4`, `facts` |
| Ch. 6 §6.3, The gain across banks | `facts.py thesis` (`quarters`: Table 6.1 with `gain_ci95`; `all_banks_ci95`; `rank_correlation_gain_train_pos`) | `exp4`, `facts` |
| Ch. 6 §6.4, The choice and the comparison with the own detector | `facts.py thesis` (`single_lender`: the reading with a single lender, the banks below zero and `worst_lender`; `every_bank_borrows`); `exp4_analysis.py policy` (`safety`: the banks that keep their own detector although the chosen detector would raise their score) | `exp4`, `facts` |
| Ch. 6 §6.5, Answer to research question 3 | `exp4_analysis.py verdict` | `exp4` |
| Ch. 7, Discussion, and the abstract | the numbers of the rows above; `facts.py thesis` (`arrangements`: the weighted mean score of each arrangement at the quarter with the fewest confirmed cases and over all banks; `single_lender`: `ap_with_comparison` and `lender_train_pos`; `qualifying`: the qualifying banks at a threshold of twenty; `largest`, §7.3); Table 7.2 from `exp4_analysis.py policy` (`all_banks`), `facts.py exp2` (`components`, `secondpath`, `qualifying`); `facts.py exp1` (`rates`: the laundering rate of the training window, §7.5) | `facts` |
| App. A, Data | `fetch_ibm_aml.py`, `fetch_amlgentex_released.py`, `fetch_transxion.py` (the files read and their digests); `checks_census_amlgentex.py` (the twelve banks and 112 daily steps of the AMLgentex dataset); `facts.py thesis` (`census`: the laundering rate of the AMLgentex dataset) | none (manifests under `data/raw/`), `checks`, `facts` |
| App. B.1, The columns of the tree family | `src/m/features.py` | none |
| App. B.2, The validation scores of the graph neural network candidates at each seed | `select_gnn.py` (`commit`: candidates 2 and 3 at three seeds, and the choice) | `select_gnn` |
| App. B.3, Software | `facts.py detectors` (`software`: the package versions the fit Runs recorded); `uv.lock` (pyg-lib) | `facts` |
| App. C.1, The null check and leaving each bank out | `exp1_analysis.py` (`shortfall`: the null check and leave one out at each seed; `verdict`: `null_stability`, the 95th percentile of the GNN family on HI-Medium, TransXion and the AMLgentex dataset with fresh draws, 200 times); `exp1_readings.py spread` (`leave_each_out`, its `leave_each_out.csv`, and `widening`); `facts.py exp1` (`spread`, `per_seed`, `controls`); `figures/app_null.py` | `exp1`, `facts` |
| App. C.2, The standard deviation of the true shortfalls | `exp1_readings.py` (`residual`; `spread`: `tau_terms` and `tau_null`); `facts.py exp1` (`tau`; `spread`: the standard deviation of the shortfalls and the median error bar, and `tau_terms`) | `exp1`, `facts` |
| App. C.3, The agreement between the two families | `exp1_analysis.py verdict` (`agreement`: ρ, its 5th percentile and its 95 % interval over 1,000 redraws of the banks); `exp1_readings.py spread` (`agreement`: the partial correlations and ρ over the centred replicates); `figures/app_rho.py` | `exp1` |
| App. C.4, The robustness checks | `exp1_analysis.py` (`shortfall`: `common_rate`, and the shortfall on HI-Small; `verdict`: `robustness`); `exp1_readings.py spread` (`robustness_null` and `widening`: Table C.1); `facts.py exp1` (`tau`: `common_rate` and `HI-Small`; `roster`) | `exp1`, `facts` |
| App. C.5, Scoring each scenario at its own best configuration | `exp1_readings.py` (`paired`: the tree family and the payment-only tree, the weighted means, the changes of sign and ρ; `spread`: `own_best`); `facts.py exp1` (`readings`: `own_best`, the gain from choosing the configuration of each scenario) | `exp1`, `facts` |
| App. C.6, The decomposition of the shortfall | `exp2_analysis.py` (`components`, `secondpath`, `sightcheck`); `facts.py exp2` (`components`, `banks`, `secondpath`, `sightcheck`, `copies`, `placebo`, `never_seen_split`, `qualifying`); `facts.py thesis` (`interaction_tree`: the interaction formed three ways) | `exp2`, `facts` |
| App. C.7, The partial rank correlation | `exp2_examples.py` (`examples`: the partial rank correlations of the tree family, the GNN family and the payment-only tree, and their permutations); `facts.py exp1` (`roster`: ρ between training-window payments and size); `figures/app_partial.py` (constructed values) | `exp1`, `exp2`, `facts` |
| App. C.8, The case reduction | `exp2.py` (the reduced fits); `exp2_analysis.py dial`; `facts.py exp2` (`reduction`); `figures/app_level100.py` | `exp2`, `facts` |
| App. C.9, The other released AMLgentex datasets | `fetch_amlgentex_released.py` (all twelve); `exp1.py` (`RELEASES`); `exp1_analysis.py shortfall` (`primary.null_check` of each dataset and family: the null check of Table C.4); `exp1_readings.py spread` (`leave_each_out.<dataset>/<family>/primary`: I² with every bank, the lowest I² with one bank left out, the bank that carries most of Q and the null check without that bank) | `exp1` |
| App. D.1, The properties | `src/m/descriptors.py`; `exp3_descriptors.py` (the check on the properties; `descriptors`: the range of each property); `exp3.py screen`; `exp3_readings.py properties` (Table D.2 and Table D.3); `facts.py thesis` (`shared_accounts`: the account numbers that occur at two banks) | `exp3`, `facts` |
| App. D.2, The out-of-fold ρ | `exp3_descriptors.py` (`selection`: the out-of-fold ρ of each model, its ten values and the fit without weights); `exp3_readings.py uncertainty` (`all_twelve` and `scale_squares_and_product`: Table D.4); `figures/app_oof.py` (constructed values) | `exp3` |
| App. D.3, The comparison with the scale model | `exp3_descriptors.py` (`selection`); `exp3_readings.py` (`uncertainty`: Table D.5, every model under the four redraws, the agreement between replicates and the overlap of the intervals; `properties`: `one_count`, Table D.6) | `exp3` |
| App. D.4, The models that use labels | `src/m/ceiling.py` (the two quantities computed from labels); `exp3.py` (`ceiling`: Table D.8; `bank_facing`: the coefficients of the model on scale and confirmed cases); `exp3_readings.py properties` (`scale_pair`: the coefficients of the scale model); `facts.py thesis` (`rho_payments_active_accounts`) | `exp3`, `facts` |

A few quoted values are read directly from a named artefact rather than
computed by a facts block: the fit times on HI-Medium and its counts of
training-window payments and copies (the `seconds` and `rows` of the
`gnn_global` and `tree_global` Runs of Experiment 1 on HI-Medium; the time
for one configuration of the tree family is the `seconds.total` of the
`tree_global` Run over its eight configurations); the own fields of the
AMLgentex dataset that vary (its feature table,
`data/interim/features/sweden_100K_easy_global.parquet`); banks 0272142 and
008904, and the shortfalls and error bars of banks 0143079 and 0293268 (the
`shortfall` Run of Experiment 1 on HI-Medium in the tree family, its
`banks.csv` and `roster.csv`); the scores of bank 0143079 of 0.140 and 0.957
(the `components` Run of Experiment 2 in the tree family, `banks.csv`); the
scores of bank 020 at each level of the case reduction (the `dial` Runs of
Experiment 2, their `banks` block); the boundary share of exactly 1 at 51
banks (the `descriptors` Run of HI-Medium, `descriptors.csv`); the ten values
of the out-of-fold ρ of the scale model (the `selection` Runs,
`candidates.csv`, `score_r0` to `score_r9`); the share of the weight and of Q
carried by a named bank (the `spread_readings` Run, `leave_each_out.csv`);
and the HI-Small lenders left out of the choice of the GNN family and the
currencies their ledgers lack (the `excluded` block of the `transfer_gnn`
Runs of Experiment 4 on HI-Small, whose `lender_n_codes` give how many codes,
in the alphabetical order of the dataset, the detector of each lender embeds
for the payment format and the two currencies). The values quoted from the
sources are not computed here: the sizes of the two Large datasets of
AMLworld and the ranges of F1 and of the hyperparameters that Altman et al.
(2024) report (§3.4.3).

## Facts

`scripts/facts.py` computes the derived numbers the thesis quotes from the
recorded Runs; nothing trains. Each block is one `facts` Run under
`experiments/facts`, keyed by short names grouped by the table, figure or
paragraph that quotes them: `exp1`, `exp2`, `exp4`, `detectors` (how the fits
stopped, the configuration validation selected and the scores at (network,
all)) and `thesis` (the numbers other chapters quote from the policy Runs, the
censuses and the cached data). The `thesis` block reads the `exp2` block,
which `all` runs first.

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
| `meth_scenarios.py` | Ch. 3 §3.3.2, Fig. 3.2: the four scenarios over one neighbourhood of HI-Small | ledgers of Experiment 1 |
| `meth_learning_curves.py` | Ch. 3 §3.4.4, Fig. 3.3: the learning curves of the GNN family on HI-Medium, the three seeds at (network, all) and bank 011054, the qualifying bank at the median number of training-window payments, at (ledger, own) and (network, own) | `exp1` gnn_global, gnn_solo, shortfall; `exp2` gnn_nown |
| `res_shortfalls.py` | Ch. 4 §4.1.2, Fig. 4.1: the shortfall at each bank of HI-Medium | `exp1` shortfall |
| `res_size.py` | Ch. 4 §4.1.3, Fig. 4.2: the shortfall and the size of each bank | `exp1` shortfall |
| `res_components.py` | Ch. 4 §4.2.1, Fig. 4.4: the value of training examples and of sight at each bank | `exp2` components |
| `res_reduction.py` | Ch. 4 §4.2.3, Fig. 4.5: the score with fewer confirmed cases | `exp2` dial |
| `res_rules.py` | Ch. 5 §5.2, Fig. 5.1: the difference between the out-of-fold ρ of each of the fifteen other models on label-free properties and that of the scale model, with 5th to 95th percentile bars | `exp3` uncertainty |
| `res_borrowed.py` | Ch. 6 §6.2, Fig. 6.1: the gain from borrowing and the shortfall at each bank | `exp4` policy |
| `app_null.py` | App. C.1, Fig. C.1: the null check on two combinations | `exp1` shortfall |
| `app_rho.py` | App. C.3, Fig. C.3: the rankings of the banks of HI-Medium by the two families | `exp1` shortfall, verdict |
| `app_partial.py` | App. C.7, Fig. C.4: a partial rank correlation on six banks | constructed values |
| `app_level100.py` | App. C.8, Fig. C.5: the score with 100 confirmed cases | `exp2` dial |
| `app_oof.py` | App. D.2, Fig. D.1: out-of-fold predictions on six banks | constructed values |

## Tests

```sh
uv run pytest
uv run ruff check .
```

The tests need no Runs and pass before any data are fetched; one test in
`tests/test_gnn.py` is skipped until the HI-Small feature table exists.

## Repository layout

```
src/m/            the library: Run log, statistics, features, GNN, label-free properties (descriptors.py), regression models (rule.py)
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
