from __future__ import annotations

from pathlib import Path

from src.project_pipeline import (
    load_har_dataset,
    run_har_experiment,
    run_toy_experiment,
    write_report,
    write_results,
)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir.parent / "UCI HAR Dataset"
    figures_dir = project_dir / "figures"
    outputs_dir = project_dir / "outputs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HAR dataset...")
    dataset = load_har_dataset(dataset_dir, pca_dim=8)
    print("Running toy experiment...")
    toy_results = run_toy_experiment(figures_dir)
    print("Running HAR experiment...")
    har_results, _, _ = run_har_experiment(dataset, figures_dir)

    results = {
        "toy": toy_results,
        "har": har_results,
    }

    print("Writing outputs...")
    write_results(outputs_dir / "results_summary.json", results)
    write_report(project_dir / "final_report.tex", figures_dir, results)
    print("Done.")


if __name__ == "__main__":
    main()
