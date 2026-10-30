# Citation audit, 28 October 2026

Primary comparisons checked: Evergreen arXiv:2604.26180v2, SUPG (PVLDB paper), Semantic Views arXiv:2610.12768v1, LOTUS arXiv:2407.11418v3, SemBench (PVLDB paper), and Learn then Test (DOI metadata). Snapshot metadata for LTT, SemBench and BARGAIN is in `../data/results/completion/citation_metadata.json`.

Corrections: BARGAIN's cited DOI identifies a **2025 PACMMOD journal article, volume 3 issue 6**, rather than a 2026 SIGMOD proceedings entry. Its existing citation key is retained for compatibility. SUPG and SemBench now include publisher DOIs. Version identifiers are explicit for Evergreen and Semantic Views. Qwen's technical report is cited for the additional model.

The new Related Work paragraph acknowledges the reduction of nonempty-output group FDP to one minus group precision. Group labeling cost and the frozen-corpus experiment are the contribution context; different error-unit terminology alone does not establish methodological novelty. No empirical superiority over a faithful Evergreen or SUPG implementation is claimed.

This is a focused audit of central comparisons, not a claim that every cited passage in every source has been independently audited. The official ACIIDS submission page does not establish an anonymity or artifact-hosting policy; these remain pre-submit metadata decisions. No artifact has been published externally.
