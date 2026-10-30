# Submission readiness — v9

The revised manuscript is 15 LNCS pages and has additional grouping evidence, a fixed-output baseline, and a disclosed degenerate small-LLM pilot. It is ready for author inspection, subject to the remaining metadata and artifact-hosting decisions below.

Validation: five tests pass; all changed Python files pass Ruff and format checks; original audit/baseline results reproduce within 1e-12; the complete five-seed training/audit pipeline reproduces exactly. PDF compilation succeeds with no undefined references/citations or overfull boxes. Underfull spacing and a PDF figure-version warning remain nonfatal build messages. See COMPLETION_REPORT.md and data/results/completion for evidence.

Before submission:

- Confirm author names, affiliations and the actual anonymity policy. Current source remains anonymous.
- Decide reviewer access to the artifact. No public URL has been created; the manuscript makes no public-release claim.
- Inspect the final PDF and local artifact package. Do not characterize the LLM pilot as successful output certification: it produces zero candidate negative groups.
- Preserve source/model/data hashes and seeds. Any final edit requires rebuilding and visually checking the exact submitted PDF.

The official submission page lists 12–15 pages including references and the extended 31 October 2026 deadline: https://aciids.pwr.edu.pl/2027/submit.html . The dates page specifies 23:59 AoE: https://aciids.pwr.edu.pl/2027/dates.html .
