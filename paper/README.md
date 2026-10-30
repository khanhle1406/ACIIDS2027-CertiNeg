# Completion update (v9)

See [COMPLETION_REPORT.md](COMPLETION_REPORT.md) for exact new protocols, verification, cost accounting and remaining submission decisions. The current PDF is `FINAL.pdf`; `FINAL.txt` contains extracted text. Cached LLM scores and all new results are in `../data/results/completion/`; the LLM pilot has no candidate negative groups.

# CertiNeg — ACIIDS 2027 research artifact (review-revised v8)

This artifact supports **“Certified Absence: Design-Based Risk Control for Semantic NOT EXISTS Queries.”**

## What v8 adds

- Runs the same approximately 35% fixed-sequence audit on **all 15 SemBench predicates** in the disclosed frequency screen, not only the four primary workloads. The paper now reports nonempty-certificate rate, mean retention, and unsafe-return rate for every row.
- Keeps the four primary workload audit streams numerically identical to the primary table, eliminating seed-induced discrepancies between the screen and focused results.
- Adds **audit-size planning** based on exact hypergeometric certification power. An illustrative 874-group target at design FDP `q0=0.07`, `(epsilon,delta)=(0.10,0.05)`, and 80% desired power needs 334 direct target-group audits.
- Consolidates novelty caveats into Related Work/Discussion and rewrites Results to emphasize what the method establishes rather than repeatedly restating that individual statistical ingredients are standard.
- Standardizes `ENGINE AND ENGINE COOLING` as `ENGINE+COOLING` in prose (`ENG.+COOL.` in the dense screen table).
- Replaces the defensive K=5 sentence with the precise statement that the uniform-grid ablation is not directly comparable with the original family.
- Preserves 15 LNCS pages without modifying font or margins.

## Complete SemBench audit screen

See `../data/results/sembench_all_predicate_audits_r8.csv`. At the primary `(epsilon,delta)=(0.10,0.05)` target and approximately 35% group audit:

- predicates with no safe coarse prefix usually abstain; SERVICE BRAKES returns nonempty in 0.65% of audits and STRUCTURE in 0.10%;
- FORWARD COLLISION AVOIDANCE and SUSPENSION expose intermediate 30% safe-prefix regimes, with unsafe-return rate 2.15% in 2,000 runs;
- FUEL SYSTEM is a full-output-safe case but retains 0.835 on average at this budget, quantifying finite-audit uncertainty;
- the four primary workloads retain 0.549–0.789 on average with 0 unsafe returns in 2,000 audits each.

Empirical rates are diagnostics; the design-based guarantee remains the formal validity statement.

## Audit planning

`experiments/audit_planning.py` computes exact fixed-set certification power under a prospective design FDP. It writes `../data/results/audit_planning_airbags.csv`. A pilot used to choose a planning FDP must be separate from, or formally accounted for in, the confirmatory audit.

## Reproduce

```bash
python -m pytest -q
python experiments/run_experiments.py
python experiments/run_stronger_baselines.py
python experiments/screen_sembench_predicates_with_groups.py --data-dir ../data/raw/full_data
python experiments/run_all_predicate_audits.py
python experiments/audit_planning.py
python experiments/run_sembench_multiseed.py
python experiments/run_revision7_eps_delta.py
```

See `DATA_SEMBENCH.md` for the SemBench processed-data source.

## Build the paper

From `paper/`:

```bash
cd latex
tectonic main.tex
```

The v8 manuscript is **15 LNCS pages including references**.

## Public artifact URL before submission

This local research workflow does **not** publish the artifact to a public GitHub/Zenodo/Drive endpoint automatically, because doing so can reveal account identity and is an external publication action. Before submission, decide the venue/anonymity policy and either:

1. publish this ZIP to an anonymous/reviewer-safe archival location and add that persistent URL to the manuscript, or
2. if the venue does not permit external artifacts during review, keep the manuscript wording limited to the reproducibility package and provide the archive at camera-ready time.

Do not invent a URL.
