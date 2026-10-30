# CertiNeg completion — v9, 28 October 2026

Implemented the agreed focused extension. The manuscript remains 15 LNCS pages. The original v8 source and PDF remain in archive/history/v8; original result files remain in data/results.

## Evidence

- 5 tests pass: exhaustive fixed-set coverage, fixed-sequence selected-set coverage, simultaneous/weighted selected-set coverage, stop-at-first-failure, and actual complete/disjoint data groups.
- Replayed all 15 predicate audits, stronger baselines and boundary tightness against frozen caches: numerical agreement within 1e-12. Repeated the complete five-seed AIR BAGS split/fit/audit pipeline: maximum difference zero for every numeric field. Reports: data/results/completion/v8_reproduction.json and multiseed_reproduction.json.
- Added a fixed 80% output baseline using the same outer samples: mean retention .768 versus .792 fixed-sequence; identical mean label cost 30410.378 and zero unsafe returns in 2000 repetitions.
- The parent-disjoint grouping extension covers four predicates, two grouping keys and two budgets. Transmission grouping increases candidate FDP and reduces retention; all results, including unsafe returns and abstention, are retained in workloads.csv.
- Qwen3-0.6B at revision c1899de289a04d12100db370d81485cdf75e47ca scored all 9018 selected complaints for two predicates (18036 scores). 16 calibration parents and 64 test parents are disjoint; all groups retain every record. Thinking is disabled; normalized Yes/No scores are used with 512 input tokens (1.752% truncated). Thresholds use calibration positives only.
- The LLM pilot is degenerate: F1 .08268/.04763, zero candidate negative groups for both predicates and both grouping keys. Eight planned configurations are retained, but their empty outputs are **not evidence of useful LLM certification**. No test-label tuning or extra model run was performed to obtain a favorable outcome.
- Full machine checks: data/results/completion/verification.json. Each nonempty workload has 2000 randomized audits; empty-candidate configurations have no substantive sampling experiment.

## Cost and execution

The successful T4 run took 1021.64 seconds including model loading. All three Modal apps are stopped. Whole-app wall time (including queues/builds) priced conservatively at capped compute rates is about $0.31. Adding $0.25 reserve for ancillary overhead gives less than $0.57, below the $1 ceiling. This is an estimate, not invoice confirmation. Startup failure and the cancelled L4 queue are included in cost_ledger.json; no further paid inference was launched.

## Reproduce from workspace root

```
python3 -m pytest -q
python3 paper/experiments/completion_baseline.py
python3 paper/experiments/completion_evaluate.py
python3 paper/experiments/completion_verify.py
```

The LLM cache is included. Re-running `modal run paper/experiments/completion_llm.py` incurs new compute usage and refuses to overwrite the existing cache; move/preserve it before intentionally rerunning. The raw archive must be extracted into data/raw/full_data as described in DATA_SEMBENCH.md. Local package versions and remote image pins are recorded alongside the results/code. Results from the original and parent-disjoint splits must not be mixed.

## Remaining submission decisions

Author identity/affiliation and the venue's review anonymity requirements still require confirmation. No reviewer-accessible artifact URL has been published. The local package is ready for inspection; external publication and submission have not been performed. A single benchmark, attribute groups, small failed LLM pilot, benchmark labels and no faithful Evergreen/SUPG system head-to-head remain limitations.
