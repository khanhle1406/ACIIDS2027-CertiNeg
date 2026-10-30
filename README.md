# CertiNeg

CertiNeg studies finite-population risk control for grouped semantic `NOT EXISTS` queries. It audits a frozen set of candidate groups and bounds the fraction that contain a positive record.

The paper is [paper/FINAL.pdf](paper/FINAL.pdf); its LaTeX source is [paper/latex/main.tex](paper/latex/main.tex). The paper currently uses an anonymous author block. The small-LLM pilot produces no candidate negative groups; useful certified output is demonstrated for the TF–IDF/SGD executor.

## Repository layout

- `paper/src/`: certification implementation.
- `paper/experiments/`: experiment and figure scripts.
- `paper/tests/`: exhaustive and protocol tests.
- `paper/latex/` and `paper/figures/`: manuscript source and figures.
- `data/results/`: cached experiment outputs, including the completion experiments.
- `paper/COMPLETION_REPORT.md`: protocols, validation, cost accounting and limitations.
- `paper/DATA_SEMBENCH.md`: dataset provenance, download identifier and checksum.
- `output/FINAL/`: a manuscript snapshot and its checksums.

The root `src`, `experiments`, `tests`, `results` and `docs` entries are convenience links to these locations.

## Setup and verification

Install the dependencies in `paper/requirements.txt` in your Python environment, then run from the repository root:

```bash
python -m pytest -q
```

To build the manuscript with Tectonic:

```bash
cd paper/latex
tectonic main.tex
```

The source references figures in `paper/figures/`. `paper/FINAL.pdf` is the submission copy; update it separately when rebuilding.

## Data and reproducibility

The SemBench Cars raw archive is not included. Download and extract it under `data/raw/` using the provenance and SHA-256 in [paper/DATA_SEMBENCH.md](paper/DATA_SEMBENCH.md). Cached derived results are included in `data/results/`. See [paper/README.md](paper/README.md) and [paper/COMPLETION_REPORT.md](paper/COMPLETION_REPORT.md) for experiment commands and the frozen pilot protocol. Running the LLM inference on Modal can incur charges; the cache supports inspecting the reported results without rerunning inference.

Historical exports and raw data remain in the local workspace and are excluded from Git. The former v9 artifact predates the manuscript's humanizer edit. Author metadata and final submission decisions remain open.
