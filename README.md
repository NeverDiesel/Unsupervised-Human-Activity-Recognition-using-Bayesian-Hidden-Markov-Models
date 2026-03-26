# Bayesian HMM HAR Project

This folder contains the finished STA2104 project assets based on the proposal:

- `final_report.pdf`: compiled final report
- `final_report.tex`: LaTeX source for the report
- `run_analysis.py`: end-to-end experiment runner
- `src/project_pipeline.py`: HMM implementation, plotting, and report generation helpers
- `figures/`: generated figures used in the report
- `outputs/results_summary.json`: saved experiment metrics

To rerun everything from this folder:

```bash
MPLCONFIGDIR='./outputs/.mplconfig' python3 run_analysis.py
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
```
