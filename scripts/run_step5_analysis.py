from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.har_data import load_har_dataset
from src.proposal_step5_analysis import run_har_experiment, save_har_summary


def main() -> None:
    dataset_dir = PROJECT_ROOT.parent / "UCI HAR Dataset"
    figures_dir = PROJECT_ROOT / "figures"
    summary_path = PROJECT_ROOT / "outputs" / "har" / "results_summary.json"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_har_dataset(dataset_dir, pca_dim=8)
    results, _, _ = run_har_experiment(dataset, figures_dir)
    save_har_summary(results, summary_path)
    print("Saved Step 5 analysis summary to", summary_path)


if __name__ == "__main__":
    main()
