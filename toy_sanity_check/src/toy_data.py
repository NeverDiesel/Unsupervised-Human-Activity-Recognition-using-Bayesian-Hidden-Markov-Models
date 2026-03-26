import numpy as np


def sample_categorical(probs, rng):
    """
    Sample one category index from a probability vector.
    probs: shape (K,)
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(len(probs), p=probs)


def sample_hidden_states(T, pi, A, rng=None):
    """
    Sample hidden state sequence Z from an HMM.

    Parameters
    ----------
    T : int
        Sequence length
    pi : array-like, shape (K,)
        Initial state distribution
    A : array-like, shape (K, K)
        Transition matrix, A[i, j] = P(z_t = j | z_{t-1} = i)
    rng : np.random.Generator or None

    Returns
    -------
    Z : np.ndarray, shape (T,)
        Hidden state sequence
    """
    if rng is None:
        rng = np.random.default_rng()

    pi = np.asarray(pi, dtype=float)
    A = np.asarray(A, dtype=float)

    K = len(pi)
    Z = np.zeros(T, dtype=int)

    Z[0] = sample_categorical(pi, rng)

    for t in range(1, T):
        Z[t] = sample_categorical(A[Z[t - 1]], rng)

    return Z


def sample_gaussian_emissions(Z, means, covs, rng=None):
    """
    Sample Gaussian observations X given hidden states Z.

    Parameters
    ----------
    Z : np.ndarray, shape (T,)
        Hidden states
    means : np.ndarray, shape (K, D)
        Mean vector for each state
    covs : np.ndarray, shape (K, D, D)
        Covariance matrix for each state
    rng : np.random.Generator or None

    Returns
    -------
    X : np.ndarray, shape (T, D)
        Observation sequence
    """
    if rng is None:
        rng = np.random.default_rng()

    Z = np.asarray(Z, dtype=int)
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)

    T = len(Z)
    D = means.shape[1]
    X = np.zeros((T, D), dtype=float)

    for t in range(T):
        k = Z[t]
        X[t] = rng.multivariate_normal(mean=means[k], cov=covs[k])

    return X


def generate_hmm_sequence(T=200, pi=None, A=None, means=None, covs=None, seed=42):
    """
    Generate a full synthetic HMM sequence: hidden states Z and observations X.

    Returns
    -------
    X : np.ndarray, shape (T, D)
    Z : np.ndarray, shape (T,)
    params : dict
        Dictionary containing pi, A, means, covs
    """
    rng = np.random.default_rng(seed)

    # Default: 3-state, 2D Gaussian HMM
    if pi is None:
        pi = np.array([0.6, 0.3, 0.1])

    if A is None:
        A = np.array([
            [0.85, 0.10, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.15, 0.80]
        ])

    if means is None:
        means = np.array([
            [0.0, 0.0],
            [3.0, 3.0],
            [0.0, 4.0]
        ])

    if covs is None:
        covs = np.array([
            [[0.4, 0.0],
             [0.0, 0.4]],

            [[0.5, 0.1],
             [0.1, 0.5]],

            [[0.4, -0.1],
             [-0.1, 0.6]]
        ])

    Z = sample_hidden_states(T=T, pi=pi, A=A, rng=rng)
    X = sample_gaussian_emissions(Z=Z, means=means, covs=covs, rng=rng)

    params = {
        "pi": pi,
        "A": A,
        "means": means,
        "covs": covs
    }

    return X, Z, params


def save_toy_data(filepath, X, Z, params):
    """
    Save toy dataset to .npz
    """
    np.savez(
        filepath,
        X=X,
        Z=Z,
        pi=params["pi"],
        A=params["A"],
        means=params["means"],
        covs=params["covs"]
    )


def load_toy_data(filepath):
    """
    Load toy dataset from .npz
    """
    data = np.load(filepath)
    X = data["X"]
    Z = data["Z"]
    params = {
        "pi": data["pi"],
        "A": data["A"],
        "means": data["means"],
        "covs": data["covs"]
    }
    return X, Z, params


if __name__ == "__main__":
    X, Z, params = generate_hmm_sequence(T=200, seed=42)

    print("X shape:", X.shape)
    print("Z shape:", Z.shape)
    print("First 20 hidden states:", Z[:20])
    print("Initial distribution pi:\n", params["pi"])
    print("Transition matrix A:\n", params["A"])
    print("Means:\n", params["means"])