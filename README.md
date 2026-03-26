# Unsupervised Human Activity Recognition using Bayesian Hidden Markov Models

This repository is organized around the final project proposal steps instead of one monolithic pipeline.

## Proposal-aligned layout

- `data/toy/`
  - synthetic toy sequence used for the Step 2 sanity check
- `outputs/toy/`
  - saved Forward-Backward and Gibbs outputs for the toy experiment
- `outputs/har/`
  - saved HAR comparison summary
- `figures/`
  - figures used in the report
- `src/toy_data.py`
  - toy data generation
- `src/toy_forward_backward.py`
  - exact Forward-Backward implementation for the toy HMM
- `src/toy_gibbs_sampler.py`
  - Gibbs sampler for the toy Bayesian HMM
- `src/proposal_step2_toy_sanity.py`
  - utilities that summarize cached Step 2 results and regenerate toy figures
- `src/proposal_step3_classical_hmm.py`
  - classical Gaussian HMM with EM for UCI HAR
- `src/proposal_step4_bayesian_hmm.py`
  - Bayesian Gaussian HMM with Gibbs sampling for UCI HAR
- `src/proposal_step5_analysis.py`
  - HAR comparison, robustness analysis, and result serialization
- `src/common.py`
  - shared math, metrics, inference, and dataset dataclasses
- `src/har_data.py`
  - UCI HAR loading and PCA preprocessing
- `src/plotting.py`
  - plotting helpers for Steps 2 to 5
- `scripts/`
  - runnable entry points for each proposal step
- `report/`
  - sectioned LaTeX sources and compiled PDF

## Main commands

Run the toy sanity-check summary from cached Step 2 outputs:

```bash
python3 scripts/run_step2_toy_summary.py
```

Run the classical HMM on UCI HAR:

```bash
python3 scripts/run_step3_classical_hmm.py
```

Run the Bayesian HMM on UCI HAR:

```bash
python3 scripts/run_step4_bayesian_hmm.py
```

Run the Step 5 comparison analysis:

```bash
MPLCONFIGDIR='./outputs/.mplconfig' python3 scripts/run_step5_analysis.py
```

Compile the report:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```
