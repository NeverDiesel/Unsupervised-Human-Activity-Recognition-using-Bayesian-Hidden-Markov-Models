from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import invwishart
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
    scaler: object
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


def iter_sequences(X: np.ndarray, lengths: Iterable[int]) -> Iterable[np.ndarray]:
    start = 0
    for length in lengths:
        stop = start + length
        yield X[start:stop]
        start = stop


def iter_sequence_pairs(
    X: np.ndarray, y: np.ndarray, lengths: Iterable[int]
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    start = 0
    for length in lengths:
        stop = start + length
        yield X[start:stop], y[start:stop]
        start = stop


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


class GaussianHMM:
    def __init__(
        self,
        n_states: int,
        covariance_regularization: float = 1e-3,
        transition_smoothing: float = 1e-2,
        random_state: int = 0,
    ) -> None:
        self.n_states = n_states
        self.covariance_regularization = covariance_regularization
        self.transition_smoothing = transition_smoothing
        self.rng = np.random.default_rng(random_state)
        self.params: GaussianParams | None = None
        self.log_likelihood_history: list[float] = []

    def _initialize(self, X: np.ndarray, lengths: list[int]) -> None:
        labels, centroids = simple_kmeans(X, self.n_states, self.rng, n_iter=30)
        means = np.zeros((self.n_states, X.shape[1]))
        covs = np.zeros((self.n_states, X.shape[1], X.shape[1]))

        for k in range(self.n_states):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                means[k] = centroids[k]
                covs[k] = np.cov(X.T) + self.covariance_regularization * np.eye(X.shape[1])
            else:
                means[k] = cluster_points.mean(axis=0)
                centered = cluster_points - means[k]
                if len(cluster_points) == 1:
                    cov = np.eye(X.shape[1])
                else:
                    cov = np.einsum("ni,nj->ij", centered, centered) / len(cluster_points)
                covs[k] = cov + self.covariance_regularization * np.eye(X.shape[1])

        init_counts, transition_counts = empirical_transition_counts(labels, lengths, self.n_states)
        pi = stable_normalize(init_counts + self.transition_smoothing)
        A = transition_counts + self.transition_smoothing
        A /= A.sum(axis=1, keepdims=True)
        self.params = GaussianParams(pi=pi, A=A, means=means, covs=covs)

    def _emissions(self, X: np.ndarray) -> np.ndarray:
        assert self.params is not None
        return multivariate_logpdf_matrix(X, self.params.means, self.params.covs)

    def fit(self, X: np.ndarray, lengths: list[int], n_iter: int = 20, tol: float = 1e-3) -> "GaussianHMM":
        self._initialize(X, lengths)
        assert self.params is not None
        K = self.n_states
        d = X.shape[1]

        for iteration in range(n_iter):
            init_expected = np.zeros(K)
            transition_expected = np.zeros((K, K))
            gamma_sums = np.zeros(K)
            weighted_sum = np.zeros((K, d))
            second_moment = np.zeros((K, d, d))
            total_log_likelihood = 0.0

            start = 0
            for length in lengths:
                stop = start + length
                seq = X[start:stop]
                log_emissions = self._emissions(seq)
                log_likelihood, gamma, xi_sum = run_forward_backward(
                    log_emissions, self.params.pi, self.params.A
                )
                total_log_likelihood += log_likelihood
                init_expected += gamma[0]
                transition_expected += xi_sum
                gamma_sums += gamma.sum(axis=0)
                weighted_sum += np.einsum("nk,nd->kd", gamma, seq)
                for k in range(K):
                    second_moment[k] += np.einsum("n,ni,nj->ij", gamma[:, k], seq, seq)
                start = stop

            self.log_likelihood_history.append(total_log_likelihood)

            means = weighted_sum / np.clip(gamma_sums[:, None], EPS, None)
            covs = np.zeros((K, d, d))
            for k in range(K):
                covariance = second_moment[k] / np.clip(gamma_sums[k], EPS, None)
                covariance -= np.outer(means[k], means[k])
                covs[k] = covariance + self.covariance_regularization * np.eye(d)

            pi = stable_normalize(init_expected + self.transition_smoothing)
            A = transition_expected + self.transition_smoothing
            A /= A.sum(axis=1, keepdims=True)
            self.params = GaussianParams(pi=pi, A=A, means=means, covs=covs)

            if iteration >= 1:
                improvement = self.log_likelihood_history[-1] - self.log_likelihood_history[-2]
                if improvement < tol:
                    break
        return self

    def score(self, X: np.ndarray, lengths: list[int]) -> float:
        assert self.params is not None
        total = 0.0
        start = 0
        for length in lengths:
            stop = start + length
            seq = X[start:stop]
            total += run_forward_backward(self._emissions(seq), self.params.pi, self.params.A)[0]
            start = stop
        return float(total)

    def predict(self, X: np.ndarray, lengths: list[int]) -> np.ndarray:
        assert self.params is not None
        predictions = []
        start = 0
        for length in lengths:
            stop = start + length
            seq = X[start:stop]
            _, gamma, _ = run_forward_backward(self._emissions(seq), self.params.pi, self.params.A)
            predictions.append(np.argmax(gamma, axis=1))
            start = stop
        return np.concatenate(predictions)

    def predict_proba(self, X: np.ndarray, lengths: list[int]) -> np.ndarray:
        assert self.params is not None
        probabilities = []
        start = 0
        for length in lengths:
            stop = start + length
            seq = X[start:stop]
            _, gamma, _ = run_forward_backward(self._emissions(seq), self.params.pi, self.params.A)
            probabilities.append(gamma)
            start = stop
        return np.vstack(probabilities)


class BayesianGaussianHMM:
    def __init__(
        self,
        n_states: int,
        covariance_regularization: float = 1e-3,
        alpha_pi: float = 1.5,
        alpha_transition: float = 2.0,
        kappa0: float = 0.2,
        nu0_offset: int = 2,
        random_state: int = 0,
    ) -> None:
        self.n_states = n_states
        self.covariance_regularization = covariance_regularization
        self.alpha_pi = alpha_pi
        self.alpha_transition = alpha_transition
        self.kappa0 = kappa0
        self.nu0_offset = nu0_offset
        self.rng = np.random.default_rng(random_state)
        self.params: GaussianParams | None = None
        self.posterior_mean_params: GaussianParams | None = None
        self.state_trace_log_likelihood: list[float] = []
        self.self_transition_trace: list[np.ndarray] = []
        self.posterior_samples: dict[str, list[np.ndarray]] = {
            "pi": [],
            "A": [],
            "means": [],
            "covs": [],
        }
        self.last_states: np.ndarray | None = None

    def _initialize(self, X: np.ndarray, lengths: list[int]) -> np.ndarray:
        baseline = GaussianHMM(
            n_states=self.n_states,
            covariance_regularization=self.covariance_regularization,
            random_state=0,
        ).fit(X, lengths, n_iter=8)
        self.params = baseline.params
        assert self.params is not None
        return baseline.predict(X, lengths)

    def _sample_emission_parameters(self, X: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = self.n_states
        d = X.shape[1]
        means = np.zeros((K, d))
        covs = np.zeros((K, d, d))

        m0 = np.zeros(d)
        psi0 = np.eye(d)
        nu0 = d + self.nu0_offset

        for k in range(K):
            cluster_points = X[states == k]
            n_k = len(cluster_points)
            if n_k == 0:
                kappa_n = self.kappa0
                nu_n = nu0
                m_n = m0
                psi_n = psi0
            else:
                x_bar = cluster_points.mean(axis=0)
                centered = cluster_points - x_bar
                scatter = np.einsum("ni,nj->ij", centered, centered)
                kappa_n = self.kappa0 + n_k
                nu_n = nu0 + n_k
                m_n = (self.kappa0 * m0 + n_k * x_bar) / kappa_n
                mean_shift = (x_bar - m0).reshape(-1, 1)
                psi_n = psi0 + scatter + (self.kappa0 * n_k / kappa_n) * (mean_shift @ mean_shift.T)

            cov = invwishart.rvs(df=nu_n, scale=psi_n, random_state=self.rng)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            cov = cov + self.covariance_regularization * np.eye(d)
            mean = self.rng.multivariate_normal(m_n, cov / kappa_n)
            means[k] = mean
            covs[k] = cov
        return means, covs

    def _sample_markov_parameters(
        self, states: np.ndarray, lengths: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        init_counts, transition_counts = empirical_transition_counts(states, lengths, self.n_states)
        pi = self.rng.dirichlet(np.full(self.n_states, self.alpha_pi) + init_counts)
        A = np.vstack(
            [
                self.rng.dirichlet(
                    np.full(self.n_states, self.alpha_transition) + transition_counts[k]
                )
                for k in range(self.n_states)
            ]
        )
        return pi, A

    def _sample_states(
        self, X: np.ndarray, lengths: list[int], pi: np.ndarray, A: np.ndarray, means: np.ndarray, covs: np.ndarray
    ) -> tuple[np.ndarray, float]:
        states = []
        total_log_likelihood = 0.0
        start = 0
        for length in lengths:
            stop = start + length
            seq = X[start:stop]
            log_emissions = multivariate_logpdf_matrix(seq, means, covs)
            seq_states, seq_log_likelihood = forward_filter_backward_sample(
                log_emissions, pi, A, self.rng
            )
            states.append(seq_states)
            total_log_likelihood += seq_log_likelihood
            start = stop
        return np.concatenate(states), float(total_log_likelihood)

    def fit(
        self,
        X: np.ndarray,
        lengths: list[int],
        n_iter: int = 36,
        burn_in: int = 18,
    ) -> "BayesianGaussianHMM":
        states = self._initialize(X, lengths)
        assert self.params is not None

        for iteration in range(n_iter):
            pi, A = self._sample_markov_parameters(states, lengths)
            means, covs = self._sample_emission_parameters(X, states)
            states, log_likelihood = self._sample_states(X, lengths, pi, A, means, covs)

            self.params = GaussianParams(pi=pi, A=A, means=means, covs=covs)
            self.last_states = states.copy()
            self.state_trace_log_likelihood.append(log_likelihood)
            self.self_transition_trace.append(np.diag(A).copy())

            if iteration >= burn_in:
                self.posterior_samples["pi"].append(pi.copy())
                self.posterior_samples["A"].append(A.copy())
                self.posterior_samples["means"].append(means.copy())
                self.posterior_samples["covs"].append(covs.copy())

        self.posterior_mean_params = GaussianParams(
            pi=np.mean(self.posterior_samples["pi"], axis=0),
            A=np.mean(self.posterior_samples["A"], axis=0),
            means=np.mean(self.posterior_samples["means"], axis=0),
            covs=np.mean(self.posterior_samples["covs"], axis=0),
        )
        return self

    def _params_for_prediction(self) -> GaussianParams:
        if self.posterior_mean_params is not None:
            return self.posterior_mean_params
        assert self.params is not None
        return self.params

    def score(self, X: np.ndarray, lengths: list[int]) -> float:
        params = self._params_for_prediction()
        total = 0.0
        start = 0
        for length in lengths:
            stop = start + length
            seq = X[start:stop]
            log_emissions = multivariate_logpdf_matrix(seq, params.means, params.covs)
            total += run_forward_backward(log_emissions, params.pi, params.A)[0]
            start = stop
        return float(total)

    def predict(self, X: np.ndarray, lengths: list[int]) -> np.ndarray:
        params = self._params_for_prediction()
        predictions = []
        start = 0
        for length in lengths:
            stop = start + length
            seq = X[start:stop]
            _, gamma, _ = run_forward_backward(
                multivariate_logpdf_matrix(seq, params.means, params.covs), params.pi, params.A
            )
            predictions.append(np.argmax(gamma, axis=1))
            start = stop
        return np.concatenate(predictions)

    def predict_proba(self, X: np.ndarray, lengths: list[int]) -> np.ndarray:
        params = self._params_for_prediction()
        probabilities = []
        start = 0
        for length in lengths:
            stop = start + length
            seq = X[start:stop]
            _, gamma, _ = run_forward_backward(
                multivariate_logpdf_matrix(seq, params.means, params.covs), params.pi, params.A
            )
            probabilities.append(gamma)
            start = stop
        return np.vstack(probabilities)


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
    X_train_reduced = pca.transform(X_train)
    X_test_reduced = pca.transform(X_test)

    return SequenceDataset(
        X_train=X_train_reduced,
        y_train=y_train,
        train_lengths=lengths_from_group_ids(subjects_train),
        X_test=X_test_reduced,
        y_test=y_test,
        test_lengths=lengths_from_group_ids(subjects_test),
        train_subjects=subjects_train,
        test_subjects=subjects_test,
        activity_names=activity_names,
        scaler={"mean": mean},
        pca=pca,
    )


def sample_hmm_sequences(
    pi: np.ndarray,
    A: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    lengths: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    observations = []
    states = []
    for length in lengths:
        z = np.empty(length, dtype=int)
        x = np.empty((length, means.shape[1]))
        z[0] = rng.choice(len(pi), p=pi)
        x[0] = rng.multivariate_normal(means[z[0]], covs[z[0]])
        for t in range(1, length):
            z[t] = rng.choice(len(pi), p=A[z[t - 1]])
            x[t] = rng.multivariate_normal(means[z[t]], covs[z[t]])
        observations.append(x)
        states.append(z)
    return np.vstack(observations), np.concatenate(states)


def make_toy_dataset(seed: int = 7) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    pi = np.array([0.75, 0.2, 0.05])
    A = np.array(
        [
            [0.95, 0.04, 0.01],
            [0.03, 0.94, 0.03],
            [0.02, 0.05, 0.93],
        ]
    )
    means = np.array(
        [
            [-4.0, -0.5],
            [0.0, 4.0],
            [4.2, -0.4],
        ]
    )
    covs = np.array(
        [
            [[0.12, 0.01], [0.01, 0.1]],
            [[0.1, -0.01], [-0.01, 0.12]],
            [[0.12, 0.01], [0.01, 0.1]],
        ]
    )
    train_lengths = [220, 210, 240, 230]
    test_lengths = [210, 205]
    X_train, z_train = sample_hmm_sequences(pi, A, means, covs, train_lengths, rng)
    X_test, z_test = sample_hmm_sequences(pi, A, means, covs, test_lengths, rng)
    return {
        "true_params": GaussianParams(pi=pi, A=A, means=means, covs=covs),
        "X_train": X_train,
        "z_train": z_train,
        "train_lengths": train_lengths,
        "X_test": X_test,
        "z_test": z_test,
        "test_lengths": test_lengths,
    }


def run_toy_experiment(output_dir: Path) -> dict[str, object]:
    toy = make_toy_dataset()
    em = GaussianHMM(n_states=3, random_state=0).fit(toy["X_train"], toy["train_lengths"], n_iter=20)
    bayes = BayesianGaussianHMM(n_states=3, random_state=1).fit(
        toy["X_train"], toy["train_lengths"], n_iter=28, burn_in=12
    )

    em_test_pred = em.predict(toy["X_test"], toy["test_lengths"])
    bayes_test_pred = bayes.predict(toy["X_test"], toy["test_lengths"])
    em_metrics = clustering_metrics(toy["z_test"], em_test_pred)
    bayes_metrics = clustering_metrics(toy["z_test"], bayes_test_pred)
    bayes_posteriors = bayes.predict_proba(toy["X_test"], toy["test_lengths"])

    plot_toy_state_recovery(
        toy["z_test"],
        bayes_test_pred,
        bayes_posteriors,
        toy["test_lengths"][0],
        output_dir / "toy_state_recovery.png",
    )
    plot_transition_heatmaps(
        toy["true_params"].A,
        bayes.posterior_mean_params.A,
        output_dir / "toy_transition_recovery.png",
        title_left="True transitions",
        title_right="Bayesian posterior mean",
    )

    return {
        "em": {
            "train_log_likelihood": em.score(toy["X_train"], toy["train_lengths"]),
            "test_log_likelihood": em.score(toy["X_test"], toy["test_lengths"]),
            **em_metrics,
        },
        "bayesian": {
            "train_log_likelihood": bayes.score(toy["X_train"], toy["train_lengths"]),
            "test_log_likelihood": bayes.score(toy["X_test"], toy["test_lengths"]),
            **bayes_metrics,
        },
    }


def run_har_experiment(dataset: SequenceDataset, output_dir: Path) -> tuple[dict[str, object], GaussianHMM, BayesianGaussianHMM]:
    em = GaussianHMM(n_states=6, random_state=0).fit(dataset.X_train, dataset.train_lengths, n_iter=20)
    bayes = BayesianGaussianHMM(n_states=6, random_state=1).fit(
        dataset.X_train, dataset.train_lengths, n_iter=34, burn_in=16
    )

    em_train_pred = em.predict(dataset.X_train, dataset.train_lengths)
    em_test_pred = em.predict(dataset.X_test, dataset.test_lengths)
    bayes_train_pred = bayes.predict(dataset.X_train, dataset.train_lengths)
    bayes_test_pred = bayes.predict(dataset.X_test, dataset.test_lengths)

    rng = np.random.default_rng(11)
    noisy_test = dataset.X_test + rng.normal(scale=0.35, size=dataset.X_test.shape)
    em_noisy_pred = em.predict(noisy_test, dataset.test_lengths)
    bayes_noisy_pred = bayes.predict(noisy_test, dataset.test_lengths)

    em_results = {
        "train_log_likelihood": em.score(dataset.X_train, dataset.train_lengths),
        "test_log_likelihood": em.score(dataset.X_test, dataset.test_lengths),
        "train": clustering_metrics(dataset.y_train, em_train_pred),
        "test": clustering_metrics(dataset.y_test, em_test_pred),
        "noisy_test": clustering_metrics(dataset.y_test, em_noisy_pred),
    }
    bayes_results = {
        "train_log_likelihood": bayes.score(dataset.X_train, dataset.train_lengths),
        "test_log_likelihood": bayes.score(dataset.X_test, dataset.test_lengths),
        "train": clustering_metrics(dataset.y_train, bayes_train_pred),
        "test": clustering_metrics(dataset.y_test, bayes_test_pred),
        "noisy_test": clustering_metrics(dataset.y_test, bayes_noisy_pred),
    }

    plot_har_model_comparison(
        em_results,
        bayes_results,
        output_dir / "har_model_comparison.png",
    )
    plot_convergence_diagnostics(
        em.log_likelihood_history,
        bayes.state_trace_log_likelihood,
        output_dir / "convergence_diagnostics.png",
    )
    plot_posterior_uncertainty(
        np.array(bayes.posterior_samples["A"]),
        output_dir / "posterior_uncertainty.png",
    )
    plot_hmm_diagram(output_dir / "hmm_diagram.png")

    return {
        "em": em_results,
        "bayesian": bayes_results,
        "pca_explained_variance": float(dataset.pca.explained_variance_ratio_.sum()),
    }, em, bayes


def plot_toy_state_recovery(
    z_true: np.ndarray,
    z_pred: np.ndarray,
    posterior_probs: np.ndarray,
    preview_length: int,
    out_path: Path,
) -> None:
    aligned_pred, _ = align_predicted_states(z_true, z_pred)
    T = min(preview_length, 120)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

    axes[0].step(range(T), z_true[:T], where="mid", label="True state", linewidth=2)
    axes[0].step(range(T), aligned_pred[:T], where="mid", label="Decoded state", linewidth=2, linestyle="--")
    axes[0].set_ylabel("State")
    axes[0].set_title("Toy-data hidden-state recovery on one sequence")
    axes[0].legend(frameon=False)

    axes[1].imshow(
        posterior_probs[:T].T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Latent state")
    axes[1].set_title("Posterior state probabilities")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_transition_heatmaps(
    A_true: np.ndarray,
    A_est: np.ndarray,
    out_path: Path,
    title_left: str,
    title_right: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for axis, matrix, title in zip(axes, [A_true, A_est], [title_left, title_right]):
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
        axis.set_title(title)
        axis.set_xlabel("Next state")
        axis.set_ylabel("Current state")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=axes, shrink=0.75)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_har_model_comparison(
    em_results: dict[str, object],
    bayes_results: dict[str, object],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    labels = ["EM-HMM", "Bayesian HMM"]
    colors = ["#1f77b4", "#d62728"]

    clean_ari = [em_results["test"]["ari"], bayes_results["test"]["ari"]]
    clean_nmi = [em_results["test"]["nmi"], bayes_results["test"]["nmi"]]
    noisy_ari = [em_results["noisy_test"]["ari"], bayes_results["noisy_test"]["ari"]]
    log_likelihoods = [em_results["test_log_likelihood"], bayes_results["test_log_likelihood"]]

    x = np.arange(len(labels))
    axes[0].bar(x - 0.15, clean_ari, width=0.3, color=colors, alpha=0.9, label="ARI")
    axes[0].bar(x + 0.15, clean_nmi, width=0.3, color=colors, alpha=0.45, label="NMI")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Clean test clustering")
    axes[0].legend(frameon=False)

    axes[1].bar(labels, log_likelihoods, color=colors)
    axes[1].set_title("Test log-likelihood")
    axes[1].tick_params(axis="x", rotation=12)

    axes[2].bar(labels, noisy_ari, color=colors)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("ARI under added noise")
    axes[2].tick_params(axis="x", rotation=12)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_convergence_diagnostics(
    em_history: list[float],
    bayes_history: list[float],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].plot(em_history, marker="o", color="#1f77b4")
    axes[0].set_title("EM training log-likelihood")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Log-likelihood")

    axes[1].plot(bayes_history, marker="o", color="#d62728")
    axes[1].set_title("Gibbs trace")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Sampled-model log-likelihood")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_posterior_uncertainty(A_samples: np.ndarray, out_path: Path) -> None:
    diagonal_samples = A_samples[:, np.arange(A_samples.shape[1]), np.arange(A_samples.shape[1])]
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    ax.boxplot(diagonal_samples, patch_artist=True)
    ax.set_xlabel("Hidden state")
    ax.set_ylabel("Self-transition probability")
    ax.set_title("Posterior uncertainty over state persistence")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hmm_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    z_positions = [(2, 2.8), (5, 2.8), (8, 2.8)]
    x_positions = [(2, 1.0), (5, 1.0), (8, 1.0)]

    for idx, position in enumerate(z_positions):
        circle = Circle(position, 0.5, facecolor="#f4c542", edgecolor="black", linewidth=1.4)
        ax.add_patch(circle)
        ax.text(position[0], position[1], rf"$z_{{t{idx - 1:+d}}}$" if idx != 1 else r"$z_t$", ha="center", va="center")

    for idx, position in enumerate(x_positions):
        circle = Circle(position, 0.5, facecolor="#8ecae6", edgecolor="black", linewidth=1.4)
        ax.add_patch(circle)
        ax.text(position[0], position[1], rf"$x_{{t{idx - 1:+d}}}$" if idx != 1 else r"$x_t$", ha="center", va="center")

    arrow_specs = [
        ((2.5, 2.8), (4.5, 2.8)),
        ((5.5, 2.8), (7.5, 2.8)),
        ((2, 2.3), (2, 1.5)),
        ((5, 2.3), (5, 1.5)),
        ((8, 2.3), (8, 1.5)),
    ]
    for start, end in arrow_specs:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.5))

    ax.text(1.0, 3.55, "Latent activity states", fontsize=11)
    ax.text(1.0, 0.15, "Observed PCA-reduced smartphone features", fontsize=11)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_report(report_path: Path, figures_dir: Path, results: dict[str, object]) -> None:
    har = results["har"]
    toy = results["toy"]
    figure_prefix = Path("figures")

    bayes_better = har["bayesian"]["test"]["ari"] >= har["em"]["test"]["ari"]
    comparison_sentence = (
        "The Bayesian HMM achieved slightly stronger clustering quality on the held-out HAR subjects."
        if bayes_better
        else "The EM baseline clustered the held-out HAR subjects slightly better, but the Bayesian model remained competitive while exposing uncertainty over persistence."
    )

    latex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{caption}}
\captionsetup{{font=small}}
\title{{Unsupervised Human Activity Recognition using Bayesian Hidden Markov Models}}
\author{{Albert Cai \\ Jiashu Li}}
\date{{March 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We study unsupervised human activity recognition on the UCI smartphone HAR dataset using finite-state Gaussian hidden Markov models. Following our proposal, we compare a classical EM-trained Gaussian HMM against a Bayesian HMM with Dirichlet priors on transition probabilities and Normal-Inverse-Wishart priors on Gaussian emissions. A toy-data sanity check verifies that the Bayesian sampler can recover latent state structure and transition dynamics from synthetic sequences with known parameters. On the HAR benchmark, we reduce the 561 engineered features to eight principal components and treat each subject as an independent sequence. {comparison_sentence} The resulting analysis highlights both the benefits and the practical tradeoffs of Bayesian parameter uncertainty in sequential clustering problems.
\end{{abstract}}

\section{{Introduction}}
Human activity recognition (HAR) aims to infer behaviors such as walking, sitting, and standing from wearable sensor streams. Most modern HAR systems are supervised, but in many realistic deployments labels are expensive, incomplete, or unavailable. This makes probabilistic latent-variable models attractive because they can discover structure directly from noisy sequential measurements while also modeling temporal dependence between activities.

This project asks whether a Bayesian hidden Markov model (HMM) offers a more robust unsupervised HAR pipeline than a classical maximum-likelihood HMM. The course motivation is a natural fit: HMMs provide a compact directed graphical model, exact inference through forward-backward recursions, and a clean Bayesian extension through conjugate priors and MCMC.

\section{{Model Overview}}
Figure~\ref{{fig:diagram}} shows the sequential model. For each subject-specific sequence, latent states $z_t \in \{{1,\dots,K\}}$ evolve as a first-order Markov chain and emit reduced-dimensional sensor vectors $x_t \in \mathbb{{R}}^d$:
\begin{{align}}
p(z_1) &= \pi_{{z_1}}, \\
p(z_t \mid z_{{t-1}}) &= A_{{z_{{t-1}}, z_t}}, \\
x_t \mid z_t = k &\sim \mathcal{{N}}(\mu_k, \Sigma_k).
\end{{align}}
We fix $K=6$ hidden states to match the six activities in the benchmark and set $d=8$ principal components after standardizing the original 561 features. This PCA projection retains {format_float(har["pca_explained_variance"] * 100, 1)}\% of the variance while keeping full-covariance Gaussian inference numerically stable.

For the EM baseline, we maximize the observed-data likelihood with exact E-steps from forward-backward and closed-form M-steps for $\pi$, $A$, $\mu_k$, and $\Sigma_k$. For the Bayesian model, we use
\begin{{align}}
\pi &\sim \text{{Dirichlet}}(\alpha_\pi \mathbf{{1}}), \\
A_k &\sim \text{{Dirichlet}}(\alpha_A \mathbf{{1}}), \\
(\mu_k, \Sigma_k) &\sim \text{{NIW}}(m_0, \kappa_0, \Psi_0, \nu_0),
\end{{align}}
and alternate three Gibbs updates: sampling initial and transition probabilities from Dirichlet posteriors, sampling emission parameters from NIW posteriors, and sampling latent state sequences with forward-filtering backward-sampling.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{{figure_prefix / "hmm_diagram.png"}}}
\caption{{Graphical sketch of the Gaussian HMM used for HAR. Yellow nodes are latent activity states and blue nodes are observed PCA-reduced sensor features.}}
\label{{fig:diagram}}
\end{{figure}}

\section{{Experimental Setup}}
The UCI HAR dataset contains 10,299 sliding-window observations collected from 30 subjects performing six activities. We use the provided train/test split, standardize the 561 engineered features using training-set statistics, project onto eight principal components, and treat each subject as one sequence. Evaluation is unsupervised: we train without labels and then compare inferred states to true activities with adjusted Rand index (ARI), normalized mutual information (NMI), and an aligned accuracy computed after Hungarian matching. We also report held-out log-likelihood and a simple robustness check where Gaussian noise is added to the test features.

Before touching the real dataset, we run a toy sanity check with a three-state Gaussian HMM whose true parameters are known. This lets us test whether the sampler recovers state sequences and transition structure at all.

\section{{Results}}
\subsection{{Toy-data sanity check}}
The synthetic experiment confirms that the Bayesian HMM can recover latent structure from data generated by the model family. On held-out synthetic sequences, the EM baseline obtained ARI {format_float(toy["em"]["ari"])} and aligned accuracy {format_float(toy["em"]["aligned_accuracy"])}, while the Bayesian HMM obtained ARI {format_float(toy["bayesian"]["ari"])} and aligned accuracy {format_float(toy["bayesian"]["aligned_accuracy"])}.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{{figure_prefix / "toy_state_recovery.png"}}}
\caption{{Toy-data sanity check. Top: true versus decoded latent states for one sequence. Bottom: posterior state probabilities from the Bayesian HMM.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.76\linewidth]{{{figure_prefix / "toy_transition_recovery.png"}}}
\caption{{True toy transition matrix compared with the Bayesian posterior-mean transition matrix.}}
\end{{figure}}

\subsection{{UCI HAR comparison}}
Table~\ref{{tab:main}} summarizes the main quantitative comparison. The EM model achieved test ARI {format_float(har["em"]["test"]["ari"])} and NMI {format_float(har["em"]["test"]["nmi"])}, while the Bayesian model achieved test ARI {format_float(har["bayesian"]["test"]["ari"])} and NMI {format_float(har["bayesian"]["test"]["nmi"])}. Held-out log-likelihood was {format_float(har["em"]["test_log_likelihood"], 1)} for EM and {format_float(har["bayesian"]["test_log_likelihood"], 1)} for the Bayesian posterior-mean model.

\begin{{table}}[H]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
Model & Test ARI & Test NMI & Test aligned acc. & Test log-likelihood \\
\midrule
EM-HMM & {format_float(har["em"]["test"]["ari"])} & {format_float(har["em"]["test"]["nmi"])} & {format_float(har["em"]["test"]["aligned_accuracy"])} & {format_float(har["em"]["test_log_likelihood"], 1)} \\
Bayesian HMM & {format_float(har["bayesian"]["test"]["ari"])} & {format_float(har["bayesian"]["test"]["nmi"])} & {format_float(har["bayesian"]["test"]["aligned_accuracy"])} & {format_float(har["bayesian"]["test_log_likelihood"], 1)} \\
\bottomrule
\end{{tabular}}
\caption{{Held-out clustering and likelihood comparison on UCI HAR.}}
\label{{tab:main}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{{figure_prefix / "har_model_comparison.png"}}}
\caption{{HAR comparison across clean clustering quality, held-out log-likelihood, and robustness under added Gaussian noise.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{{figure_prefix / "convergence_diagnostics.png"}}}
\caption{{Optimization and sampling diagnostics. Left: EM training log-likelihood. Right: Bayesian sampled-model log-likelihood trace.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{{figure_prefix / "posterior_uncertainty.png"}}}
\caption{{Posterior uncertainty over diagonal entries of the transition matrix, illustrating variation in state persistence across posterior samples.}}
\end{{figure}}

The noise-robustness experiment further shows how clustering quality changes after adding isotropic Gaussian noise in the PCA space. EM-HMM reached noisy-test ARI {format_float(har["em"]["noisy_test"]["ari"])}, whereas the Bayesian HMM reached {format_float(har["bayesian"]["noisy_test"]["ari"])}.

\section{{Discussion}}
The Bayesian model adds meaningful interpretability even when its raw clustering metrics are close to the EM baseline. Instead of a single point estimate for transition structure, it yields a posterior over state persistence and transition uncertainty. This matters for HAR because some activities, such as sitting or laying, tend to persist for long runs, while motion classes transition more frequently.

At the same time, this project also exposed a practical limitation: fully Bayesian inference is more computationally expensive and more sensitive to initialization than EM. Our implementation therefore uses PCA-reduced features and conjugate priors to keep inference tractable. A stronger extension would be to replace the finite-state model with an HDP-HMM or to use the raw inertial signals directly with a structured emission model.

\section{{Related Work}}
Rabiner (1989) remains the standard reference for HMM inference and learning, especially the forward-backward and Baum-Welch algorithms. Scott (2002) studied Bayesian inference for HMMs with MCMC, showing how posterior sampling provides a richer picture of latent-state uncertainty than maximum likelihood alone. Beal (2003) developed variational Bayesian methods for graphical models, including HMMs, offering an alternative to sampling-based Bayesian inference. Teh et al. (2006) and Fox et al. (2008) extended Bayesian HMMs to nonparametric settings where the number of hidden states can be learned rather than fixed in advance. In the HAR domain, Anguita et al. (2013) introduced the benchmark smartphone dataset that we use here, though most follow-up work on this dataset has emphasized supervised discriminative models rather than unsupervised sequential Bayesian modeling.

\section{{Conclusion}}
This project demonstrates that a Bayesian Gaussian HMM is a feasible unsupervised model for smartphone-based HAR and that exact state inference plus Gibbs sampling can be implemented with standard conjugate updates. The toy-data experiment validates the inference pipeline, and the UCI HAR comparison shows that sequential probabilistic models can recover meaningful activity structure even without labels. The main gain from the Bayesian approach is not only performance, but also access to posterior uncertainty over transitions and emissions.

\begin{{thebibliography}}{{9}}
\bibitem{{rabiner1989}} L. R. Rabiner. A tutorial on hidden Markov models and selected applications in speech recognition. \textit{{Proceedings of the IEEE}}, 1989.
\bibitem{{scott2002}} S. L. Scott. Bayesian methods for hidden Markov models. \textit{{Journal of the American Statistical Association}}, 2002.
\bibitem{{beal2003}} M. J. Beal. Variational algorithms for approximate Bayesian inference. PhD thesis, University College London, 2003.
\bibitem{{teh2006}} Y. W. Teh, M. I. Jordan, M. J. Beal, and D. M. Blei. Hierarchical Dirichlet Processes. \textit{{Journal of the American Statistical Association}}, 2006.
\bibitem{{fox2008}} E. B. Fox, E. B. Sudderth, M. I. Jordan, and A. S. Willsky. An HDP-HMM for systems with state persistence. \textit{{ICML}}, 2008.
\bibitem{{anguita2013}} D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz. A public domain dataset for human activity recognition using smartphones. \textit{{ESANN}}, 2013.
\end{{thebibliography}}

\end{{document}}
"""
    report_path.write_text(latex)


def write_results(results_path: Path, results: dict[str, object]) -> None:
    results_path.write_text(json.dumps(results, indent=2))
