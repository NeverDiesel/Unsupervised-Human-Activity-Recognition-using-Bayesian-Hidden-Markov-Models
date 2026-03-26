from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.har_data import load_har_dataset
from src.proposal_step4_bayesian_hmm import BayesianGaussianHMM


def main() -> None:
    dataset_dir = PROJECT_ROOT.parent / "UCI HAR Dataset"
    dataset = load_har_dataset(dataset_dir, pca_dim=8)
    model = BayesianGaussianHMM(n_states=6, random_state=1).fit(
        dataset.X_train, dataset.train_lengths, n_iter=34, burn_in=16
    )
    print("Step 4 Bayesian HMM complete.")
    print("Train log-likelihood:", model.score(dataset.X_train, dataset.train_lengths))
    print("Test log-likelihood:", model.score(dataset.X_test, dataset.test_lengths))


if __name__ == "__main__":
    main()
