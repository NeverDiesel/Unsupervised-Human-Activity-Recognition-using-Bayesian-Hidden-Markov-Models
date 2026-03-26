import os
import sys
import numpy as np

# Make sure Python can find src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

from toy_forward_backward import load_toy_data, forward_backward


def main():
    # Paths
    data_path = os.path.join(PROJECT_ROOT, "data", "toy", "sequence.npz")
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "toy")
    os.makedirs(output_dir, exist_ok=True)

    # Load toy data
    X, Z_true, params = load_toy_data(data_path)

    # Run forward-backward
    results = forward_backward(
        X=X,
        pi=params["pi"],
        A=params["A"],
        means=params["means"],
        covs=params["covs"]
    )

    gamma = results["gamma"]          # posterior state probabilities
    xi = results["xi"]                # pairwise posterior
    alpha = results["alpha"]
    beta = results["beta"]
    log_likelihood = results["log_likelihood"]

    # Convert posterior probabilities to hard state predictions
    Z_hat = np.argmax(gamma, axis=1)

    # Simple accuracy against true hidden states
    accuracy = np.mean(Z_hat == Z_true)

    # Print summary
    print("=== Run Toy Forward-Backward ===")
    print("Data path:", data_path)
    print("X shape:", X.shape)
    print("Z_true shape:", Z_true.shape)
    print("alpha shape:", alpha.shape)
    print("beta shape:", beta.shape)
    print("gamma shape:", gamma.shape)
    print("xi shape:", xi.shape)
    print("log_likelihood:", log_likelihood)
    print("state accuracy:", accuracy)

    print("\nFirst 20 true states:")
    print(Z_true[:20])

    print("\nFirst 20 inferred states:")
    print(Z_hat[:20])

    print("\nFirst 5 rows of gamma:")
    print(gamma[:5])

    # Save outputs for later plotting / report use
    save_path = os.path.join(output_dir, "forward_backward_results.npz")
    np.savez(
        save_path,
        X=X,
        Z_true=Z_true,
        Z_hat=Z_hat,
        gamma=gamma,
        xi=xi,
        alpha=alpha,
        beta=beta,
        log_likelihood=log_likelihood
    )

    print("\nResults saved to:")
    print(save_path)


if __name__ == "__main__":
    main()