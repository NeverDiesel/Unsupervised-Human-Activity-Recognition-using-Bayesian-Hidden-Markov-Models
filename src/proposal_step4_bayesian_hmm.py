from __future__ import annotations

import numpy as np
from scipy.stats import invwishart

from .common import (
    GaussianParams,
    empirical_transition_counts,
    forward_filter_backward_sample,
    multivariate_logpdf_matrix,
    run_forward_backward,
)
from .proposal_step3_classical_hmm import GaussianHMM


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
            if np.ndim(cov) == 0:
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
