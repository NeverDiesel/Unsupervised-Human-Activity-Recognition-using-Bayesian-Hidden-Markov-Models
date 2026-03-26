import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath(os.path.join(PROJECT_ROOT, "..")))

from toy_sanity_check.src.toy_gibbs_sampler import load_toy_data, gibbs_sample_toy_hmm, posterior_mean


def main():
    data_path = os.path.join(PROJECT_ROOT, "data", "sequence.npz")
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    X, Z_true, true_params = load_toy_data(data_path)

    results = gibbs_sample_toy_hmm(
        X,
        K=3,
        n_iter=300,
        burn_in=100,
        thin=2,
        seed=42,
    )

    pi_post = posterior_mean(results["pi_samples"])
    A_post = posterior_mean(results["A_samples"])
    means_post = posterior_mean(results["means_samples"])

    print("\n=== True vs Posterior Mean ===")
    print("True pi:\n", true_params["pi"])
    print("Posterior mean pi:\n", pi_post)

    print("\nTrue A:\n", true_params["A"])
    print("Posterior mean A:\n", A_post)

    print("\nTrue means:\n", true_params["means"])
    print("Posterior mean means:\n", means_post)

    save_path = os.path.join(output_dir, "gibbs_results.npz")
    np.savez(
        save_path,
        pi_samples=results["pi_samples"],
        A_samples=results["A_samples"],
        means_samples=results["means_samples"],
        covs_samples=results["covs_samples"],
        Z_samples=results["Z_samples"],
        loglik_trace=results["loglik_trace"],
        last_pi=results["last_pi"],
        last_A=results["last_A"],
        last_means=results["last_means"],
        last_covs=results["last_covs"],
        last_Z=results["last_Z"],
        Z_true=Z_true,
        true_pi=true_params["pi"],
        true_A=true_params["A"],
        true_means=true_params["means"],
        true_covs=true_params["covs"],
    )

    print("\nSaved to:")
    print(save_path)


if __name__ == "__main__":
    main()
