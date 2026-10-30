# SemBench Cars data used by the real-text experiments (v7)

The experiments use the public **SemBench Cars** processed archive `full_data.zip` referenced by the SemBench repository's Cars preparation script.

Expected extracted files:

- `full_data/text_complaints_data_full.csv`
- `full_data/car_data_full.csv`

Public Google Drive file ID referenced by SemBench: `1mNJaYSv5W_5sWhrGCCfdo5ljmfMthifI`.

Archive observed on 2026-10-26:

- file name: `full_data.zip`
- size: 102,650,263 bytes
- SHA-256: `956fac7534d14149226599f4bd54882cfddc669149d8aa63f8fdedd493def8c6`

## Predicate screen and primary protocol

The primary split is deterministic and group-disjoint (65/35) by `(country, year, fuel_type)`. A 20k-feature TF-IDF unigram representation and logistic-loss SGD are fit on training text; each classification threshold is selected from training positives only.

`results/sembench_predicate_screen.csv` contains the **complete screen of the 15 most frequent training-set component classes**. The four primary workloads are the screen rows satisfying the descriptive comparability filter:

- at least 1,000 candidate groups;
- full candidate FDP > 0.10;
- a safe 80% member in the original 30/45/60/80/100% family.

Those rows are:

- `AIR BAGS`
- `ENGINE AND ENGINE COOLING`
- `VISIBILITY/WIPER`
- `EXTERIOR LIGHTING`

The paper also reports the remaining screen outcomes, including ENGINE, SERVICE BRAKES, and FUEL SYSTEM, so the primary subset is not presented as the only predicates tried.

## Split robustness

`experiments/run_sembench_multiseed.py` repeats the full AIR BAGS split, representation fit, classifier fit, threshold selection, candidate construction, and 2,000 audits for seeds:

`20261026, 20261027, 20261028, 20261029, 20261030`.

Derived results:

- `results/sembench_airbags_multiseed.csv`
- `results/sembench_airbags_multiseed_summary.json`

## Epsilon/delta sensitivity

`experiments/run_revision7_eps_delta.py` holds the primary AIR BAGS frozen group population, `n=383`, the original five-set family, and the same 2,000 SRSWOR samples fixed while sweeping:

- `epsilon ∈ {0.05, 0.10, 0.20}`
- `delta ∈ {0.01, 0.05, 0.10}`

Output: `results/epsilon_delta_sweep_airbags.csv`.

The canonical `(0.10,0.05)` cell reproduces the primary mean retention `0.6304666` because it reuses the same audit sample stream as the primary AIR BAGS experiment.

## Reproduce

```bash
python experiments/screen_sembench_predicates.py --data-dir /path/to/full_data
python experiments/cache_sembench_predicates_r6.py --data-dir /path/to/full_data
python experiments/run_revision6_sembench_audits.py
python experiments/run_sembench_multiseed.py
python experiments/recompute_curve_only.py
python experiments/run_revision7_eps_delta.py
python experiments/make_revision6_figures.py
```

Compact derived group arrays are stored in `results/sembench_*_groups.npz`, allowing statistical audit sweeps without retraining the text models. The 102 MB source archive is not redistributed in this artifact.

## v8 complete-screen audit

`experiments/screen_sembench_predicates_with_groups.py --data-dir /path/to/full_data` recreates the top-15 screen and persists each predicate's risk-ordered group-truth array as `results/sembench_screen_*_groups.npz`. `experiments/run_all_predicate_audits.py` then runs the same original 30/45/60/80/100% fixed-sequence family at approximately 35% group audit for all 15 predicates and writes `results/sembench_all_predicate_audits_r8.csv`. The four starred workloads reuse the primary-table audit seeds to keep their displayed retention values identical across tables.

`experiments/audit_planning.py` provides exact prospective fixed-set power calculations and writes `results/audit_planning_airbags.csv`.

## Completion protocol (v9)

`experiments/completion_data.py` defines a separate 65/35 split on the coarser `(country, year)` parent key, seed 20261026. The split keeps both fuel-type and transmission groups disjoint. V8 splits/results are retained unchanged and must not be conflated with this new split. A label-independent SHA-256 ordering selects 16 training parents for calibration and 64 held-out parents for LLM evaluation, preserving every row of selected parents. Exact membership is in `../data/results/completion/llm_membership.csv`.

The LLM consumes complaint text and the named predicate only; `component_class` is withheld from inference. It is used only for calibration on the training side and as the corpus-relative reference in evaluation. This is a benchmark-label experiment, not a human-oracle evaluation. The grouping extension is constructed by this project over SemBench fields; it is not an official SemBench query.
