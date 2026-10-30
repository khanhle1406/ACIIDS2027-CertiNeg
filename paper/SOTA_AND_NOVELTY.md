# CertiNeg — SOTA and novelty notes (checked 26 Oct 2026)

## Closest systems/statistical neighbors

- **Evergreen (2026)** — quantified/grouped semantic claim verification, witness stopping, finite-population sampling, confidence sequences, and multiplicity handling. Closest systems neighbor, but its natural operating unit is claim verification rather than selection of a returned group set under a nonzero FDP target.
- **SUPG (PVLDB 2020)** — proxy-driven approximate selection with high-probability precision/recall guarantees under limited oracle labels.
- **BARGAIN (SIGMOD 2026)** — adaptive cheap/expensive LLM cascades with guarantees on accuracy, precision, or recall.
- **Semantic Views (2026)** — semantic-filter precision/recall certification using free ranking signals and fresh stratified oracle samples.
- **Learn-then-Test (AOAS 2025)** — risk-control via multiple testing, including fixed-sequence procedures for ordered candidate families.
- **Selection by Prediction with Conformal p-values (JMLR 2023)** — false-selection control under exchangeability.
- **Cochran (1977)** — classical finite-population sampling without replacement.
- **ALPHA / weighted risk-limiting audits** — sequential and side-information-aware finite-population auditing.
- **LOTUS / SemBench** — semantic query processing and benchmarking context.

## What is explicitly not novel

Do not claim novelty for hypergeometric inversion, SRSWOR, Bonferroni/weighted alpha, fixed-sequence testing, confidence sequences/e-processes, weighted audit sampling, proxy ordering, witness-first stopping, or generic semantic-filter certification.

## Defensible contribution boundary

1. group-level FDP for the whole frozen output of semantic `NOT EXISTS`;
2. exact trusted group truth with asymmetric inner cost;
3. design-based finite-corpus guarantee without calibrated model probabilities;
4. selected-set certification over a predeclared family using standard multiple-testing tools;
5. empirical characterization of tightness, census-relative cost, audit budget, family granularity, risk ranking, complete predicate screening, target sensitivity, and split variability;
6. end-to-end SemBench Cars learned-predicate workloads showing that strong row metrics can coexist with unsafe group-level absence outputs.

## Remaining strongest empirical gap

A true LLM/VLM workload and a second benchmark/grouping scheme are still missing. A stronger systems extension should compare model/prompt variance, oracle-label acquisition latency/cost, and sequential/weighted audits at matched `(epsilon, delta)`.

## v8 framing note

The manuscript now concentrates the prior-tool/novelty boundary in Related Work and Discussion. Abstract and Results state the positive contribution directly: grouped-absence FDP as the output-set target, exact group-truth auditing, and empirical validity/utility/cost characterization.

## v9 clarification

Group FDP is one minus group precision for nonempty output. The manuscript now acknowledges this reduction explicitly. Additional grouping evidence and a failed small-LLM pilot extend the cost/utility characterization; they do not establish superiority over a faithful SUPG or Evergreen implementation. See CITATION_AUDIT.md for source-version and metadata corrections.
