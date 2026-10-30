# Real-workload status (v7)

## SemBench Cars — executed end to end

The v7 artifact executes held-out text workloads using the public SemBench Cars processed complaint and car metadata tables.

- Group key: `(country, year, fuel_type)`.
- Model family: frozen TF-IDF + logistic-loss SGD trained only on group-disjoint training data.
- Predicate screen: all 15 most frequent training-set `component_class` values.
- Primary comparable workloads: `AIR BAGS`, `ENGINE AND ENGINE COOLING`, `VISIBILITY/WIPER`, `EXTERIOR LIGHTING`.
- Primary audit policy: ordered fixed-sequence certification.
- Main target: `(epsilon, delta)=(0.10, 0.05)`.
- Primary audit fraction: approximately 35% of candidate groups, 2,000 repetitions per predicate.
- AIR BAGS robustness: five complete group-disjoint split/fit/evaluate runs, 10,000 audit randomizations total.
- AIR BAGS target sensitivity: 9 `(epsilon, delta)` cells × 2,000 audits = 18,000 randomizations.

See `DATA_SEMBENCH.md`, `results/sembench_predicate_screen.csv`, `results/sembench_primary_predicates_r6.csv`, and `results/epsilon_delta_sweep_airbags.csv`.

## Multimodal/LLM-VLM workload — future systems extension

A true LLM/VLM workload remains the largest systems-level gap. The paper does not label the TF-IDF/SGD experiments as LLM inference. A future extension should evaluate model/prompt variance, oracle-label acquisition latency/cost, and a second dataset or grouping scheme.

## v8

The main paper now reports audit outcomes for every predicate in the complete SemBench top-15 screen. This does not add a second benchmark or a true LLM/VLM executor; those remain extension gates.

## v9 completion

The additional parent-disjoint split evaluates fuel-type and transmission grouping. A fully executed Qwen3-0.6B text pilot produces no candidate negatives, so useful LLM certification remains unestablished. See COMPLETION_REPORT.md for results and limits.
