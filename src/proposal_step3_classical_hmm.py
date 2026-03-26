from __future__ import annotations

import numpy as np

from .common import (
    EPS,
    GaussianParams,
    empirical_transition_counts,
    multivariate_logpdf_matrix,
    run_forward_backward,
    simple_kmeans,
    stable_normalize,
)


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
