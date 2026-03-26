from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.har_data import load_har_dataset
from src.proposal_step3_classical_hmm import GaussianHMM


def main() -> None:
    dataset_dir = PROJECT_ROOT.parent / "UCI HAR Dataset"
    dataset = load_har_dataset(dataset_dir, pca_dim=8)
    model = GaussianHMM(n_states=6, random_state=0).fit(dataset.X_train, dataset.train_lengths, n_iter=20)
    print("Step 3 classical HMM complete.")
    print("Train log-likelihood:", model.score(dataset.X_train, dataset.train_lengths))
    print("Test log-likelihood:", model.score(dataset.X_test, dataset.test_lengths))


if __name__ == "__main__":
    main()
