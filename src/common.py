from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


EPS = 1e-12


@dataclass
class PCATransformer:
    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_ratio_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        centered = X - self.mean_
        return np.einsum("nd,dk->nk", centered, self.components_.T)


@dataclass
class SequenceDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    train_lengths: list[int]
    X_test: np.ndarray
    y_test: np.ndarray
    test_lengths: list[int]
    train_subjects: np.ndarray
    test_subjects: np.ndarray
    activity_names: dict[int, str]
    pca: PCATransformer


@dataclass
class GaussianParams:
    pi: np.ndarray
    A: np.ndarray
    means: np.ndarray
    covs: np.ndarray


def lengths_from_group_ids(group_ids: np.ndarray) -> list[int]:
    changes = np.flatnonzero(np.r_[True, group_ids[1:] != group_ids[:-1], True])
    return np.diff(changes).tolist()


def stable_normalize(probabilities: np.ndarray) -> np.ndarray:
    total = probabilities.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full_like(probabilities, 1.0 / len(probabilities))
    return probabilities / total


def sample_from_log_probs(log_probs: np.ndarray, rng: np.random.Generator) -> int:
    shifted = log_probs - np.max(log_probs)
    probabilities = np.exp(shifted)
    probabilities = stable_normalize(probabilities)
    return int(rng.choice(len(probabilities), p=probabilities))


def multivariate_logpdf_matrix(
    X: np.ndarray, means: np.ndarray, covs: np.ndarray
) -> np.ndarray:
    log_probs = np.empty((X.shape[0], means.shape[0]))
    d = X.shape[1]
    constant = d * np.log(2.0 * np.pi)
    for k in range(means.shape[0]):
        covariance = covs[k]
        jitter = 1e-6
        while True:
            try:
                chol = np.linalg.cholesky(covariance)
                break
            except np.linalg.LinAlgError:
                covariance = covariance + jitter * np.eye(d)
                jitter *= 10
        centered = X - means[k]
        solved = np.linalg.solve(chol, centered.T)
        mahalanobis = np.sum(solved**2, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        log_probs[:, k] = -0.5 * (constant + log_det + mahalanobis)
    return log_probs


def run_forward_backward(
    log_emissions: np.ndarray, pi: np.ndarray, A: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    T, K = log_emissions.shape
    log_pi = np.log(np.clip(pi, EPS, 1.0))
    log_A = np.log(np.clip(A, EPS, 1.0))

    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_emissions[0]
    for t in range(1, T):
        log_alpha[t] = log_emissions[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_A, axis=0
        )

    log_likelihood = float(logsumexp(log_alpha[-1]))

    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_A + log_emissions[t + 1][None, :] + log_beta[t + 1][None, :],
            axis=1,
        )

    log_gamma = log_alpha + log_beta - log_likelihood
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        log_xi_t = (
            log_alpha[t][:, None]
            + log_A
            + log_emissions[t + 1][None, :]
            + log_beta[t + 1][None, :]
            - log_likelihood
        )
        xi_t = np.exp(log_xi_t)
        xi_sum += xi_t / np.clip(xi_t.sum(), EPS, None)
    return log_likelihood, gamma, xi_sum


def forward_filter_backward_sample(
    log_emissions: np.ndarray, pi: np.ndarray, A: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    T, K = log_emissions.shape
    log_pi = np.log(np.clip(pi, EPS, 1.0))
    log_A = np.log(np.clip(A, EPS, 1.0))

    log_alpha = np.empty((T, K))
    scales = np.empty(T)
    log_alpha[0] = log_pi + log_emissions[0]
    scales[0] = logsumexp(log_alpha[0])
    log_alpha[0] -= scales[0]

    for t in range(1, T):
        log_alpha[t] = log_emissions[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_A, axis=0
        )
        scales[t] = logsumexp(log_alpha[t])
        log_alpha[t] -= scales[t]

    states = np.empty(T, dtype=int)
    states[-1] = sample_from_log_probs(log_alpha[-1], rng)
    for t in range(T - 2, -1, -1):
        states[t] = sample_from_log_probs(log_alpha[t] + log_A[:, states[t + 1]], rng)

    return states, float(scales.sum())


def empirical_transition_counts(
    states: np.ndarray, lengths: Iterable[int], n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    init_counts = np.zeros(n_states)
    transition_counts = np.zeros((n_states, n_states))
    start = 0
    for length in lengths:
        stop = start + length
        seq_states = states[start:stop]
        init_counts[seq_states[0]] += 1
        for t in range(length - 1):
            transition_counts[seq_states[t], seq_states[t + 1]] += 1
        start = stop
    return init_counts, transition_counts


def simple_kmeans(
    X: np.ndarray, n_clusters: int, rng: np.random.Generator, n_iter: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    initial_indices = rng.choice(len(X), size=n_clusters, replace=False)
    centroids = X[initial_indices].copy()
    labels = np.zeros(len(X), dtype=int)

    for _ in range(n_iter):
        squared_distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(squared_distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            members = X[labels == k]
            if len(members) == 0:
                centroids[k] = X[rng.integers(0, len(X))]
            else:
                centroids[k] = members.mean(axis=0)
    return labels, centroids


def align_predicted_states(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, float]:
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    size = max(len(true_labels), len(pred_labels))
    confusion = np.zeros((size, size))
    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            confusion[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {}
    for row, col in zip(row_ind, col_ind):
        if col < len(pred_labels) and row < len(true_labels):
            mapping[pred_labels[col]] = true_labels[row]
    aligned = np.array([mapping.get(label, label) for label in y_pred])
    accuracy = float(np.mean(aligned == y_true))
    return aligned, accuracy


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    _, aligned_accuracy = align_predicted_states(y_true, y_pred)
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "aligned_accuracy": aligned_accuracy,
    }
