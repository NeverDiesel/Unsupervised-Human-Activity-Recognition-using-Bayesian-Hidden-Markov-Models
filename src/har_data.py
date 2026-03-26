from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import EPS, PCATransformer, SequenceDataset, lengths_from_group_ids


def load_har_dataset(base_path: Path, pca_dim: int = 8) -> SequenceDataset:
    activity_names = {}
    with open(base_path / "activity_labels.txt") as handle:
        for line in handle:
            idx, name = line.strip().split()
            activity_names[int(idx)] = name.replace("_", " ")

    X_train = np.loadtxt(base_path / "train" / "X_train.txt")
    y_train = np.loadtxt(base_path / "train" / "y_train.txt", dtype=int)
    subjects_train = np.loadtxt(base_path / "train" / "subject_train.txt", dtype=int)

    X_test = np.loadtxt(base_path / "test" / "X_test.txt")
    y_test = np.loadtxt(base_path / "test" / "y_test.txt", dtype=int)
    subjects_test = np.loadtxt(base_path / "test" / "subject_test.txt", dtype=int)

    mean = X_train.mean(axis=0)
    X_train_centered = X_train - mean
    X_test_centered = X_test - mean

    covariance = np.einsum("ni,nj->ij", X_train_centered, X_train_centered) / (len(X_train_centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    components = eigenvectors[:, :pca_dim].T
    explained_variance_ratio = eigenvalues[:pca_dim] / np.clip(eigenvalues.sum(), EPS, None)

    pca = PCATransformer(
        mean_=mean,
        components_=components,
        explained_variance_ratio_=explained_variance_ratio,
    )

    return SequenceDataset(
        X_train=pca.transform(X_train),
        y_train=y_train,
        train_lengths=lengths_from_group_ids(subjects_train),
        X_test=pca.transform(X_test),
        y_test=y_test,
        test_lengths=lengths_from_group_ids(subjects_test),
        train_subjects=subjects_train,
        test_subjects=subjects_test,
        activity_names=activity_names,
        pca=pca,
    )
