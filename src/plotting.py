from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from .common import align_predicted_states


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
    title_left: str = "True transitions",
    title_right: str = "Estimated transitions",
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
