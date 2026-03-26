from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import SequenceDataset, clustering_metrics
from .plotting import (
    plot_convergence_diagnostics,
    plot_har_model_comparison,
    plot_hmm_diagram,
    plot_posterior_uncertainty,
)
from .proposal_step3_classical_hmm import GaussianHMM
from .proposal_step4_bayesian_hmm import BayesianGaussianHMM


def run_har_experiment(
    dataset: SequenceDataset, output_dir: Path
) -> tuple[dict[str, object], GaussianHMM, BayesianGaussianHMM]:
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

    plot_har_model_comparison(em_results, bayes_results, output_dir / "har_model_comparison.png")
    plot_convergence_diagnostics(
        em.log_likelihood_history,
        bayes.state_trace_log_likelihood,
        output_dir / "convergence_diagnostics.png",
    )
    plot_posterior_uncertainty(np.array(bayes.posterior_samples["A"]), output_dir / "posterior_uncertainty.png")
    plot_hmm_diagram(output_dir / "hmm_diagram.png")

    return {
        "em": em_results,
        "bayesian": bayes_results,
        "pca_explained_variance": float(dataset.pca.explained_variance_ratio_.sum()),
    }, em, bayes


def save_har_summary(results: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(results, indent=2))
