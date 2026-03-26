from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import clustering_metrics
from .plotting import plot_toy_state_recovery, plot_transition_heatmaps


def _posterior_mode_states(Z_samples: np.ndarray) -> np.ndarray:
    modes = []
    for t in range(Z_samples.shape[1]):
        states, counts = np.unique(Z_samples[:, t], return_counts=True)
        modes.append(states[np.argmax(counts)])
    return np.asarray(modes, dtype=int)


def summarize_cached_toy_results(project_root: Path) -> dict[str, object]:
    toy_dir = project_root / "outputs" / "toy"
    fb = np.load(toy_dir / "forward_backward_results.npz")
    gibbs = np.load(toy_dir / "gibbs_results.npz")

    z_true = fb["Z_true"]
    z_fb = fb["Z_hat"]
    z_gibbs = _posterior_mode_states(gibbs["Z_samples"])

    summary = {
        "forward_backward": {
            **clustering_metrics(z_true, z_fb),
            "log_likelihood": float(fb["log_likelihood"]),
        },
        "gibbs": {
            **clustering_metrics(z_true, z_gibbs),
            "posterior_mean_pi": np.mean(gibbs["pi_samples"], axis=0).tolist(),
            "posterior_mean_A": np.mean(gibbs["A_samples"], axis=0).tolist(),
        },
    }

    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_toy_state_recovery(
        z_true=z_true,
        z_pred=z_fb,
        posterior_probs=fb["gamma"],
        preview_length=min(120, len(z_true)),
        out_path=figures_dir / "toy_state_recovery.png",
    )
    plot_transition_heatmaps(
        np.asarray(gibbs["true_A"]),
        np.mean(gibbs["A_samples"], axis=0),
        figures_dir / "toy_transition_recovery.png",
        title_left="True transitions",
        title_right="Bayesian posterior mean",
    )

    summary_path = toy_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
