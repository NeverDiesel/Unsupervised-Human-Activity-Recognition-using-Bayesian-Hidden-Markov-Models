import numpy as np


def log_multivariate_normal_pdf(x, mean, cov):
    """
    Compute log N(x | mean, cov) for one observation x.

    Parameters
    ----------
    x : np.ndarray, shape (D,)
    mean : np.ndarray, shape (D,)
    cov : np.ndarray, shape (D, D)

    Returns
    -------
    float
        Log-density value
    """
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    d = x.shape[0]
    diff = x - mean

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")

    inv_cov = np.linalg.inv(cov)

    return -0.5 * (
        d * np.log(2 * np.pi)
        + logdet
        + diff.T @ inv_cov @ diff
    )


def compute_emission_likelihoods(X, means, covs):
    """
    Compute emission likelihood matrix B where:
    B[t, k] = p(x_t | z_t = k)

    Parameters
    ----------
    X : np.ndarray, shape (T, D)
    means : np.ndarray, shape (K, D)
    covs : np.ndarray, shape (K, D, D)

    Returns
    -------
    B : np.ndarray, shape (T, K)
        Emission likelihood matrix
    """
    X = np.asarray(X, dtype=float)
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)

    T = X.shape[0]
    K = means.shape[0]

    B = np.zeros((T, K), dtype=float)

    for t in range(T):
        for k in range(K):
            logp = log_multivariate_normal_pdf(X[t], means[k], covs[k])
            B[t, k] = np.exp(logp)

    return B


def forward_pass(pi, A, B):
    """
    Scaled forward algorithm.

    Parameters
    ----------
    pi : np.ndarray, shape (K,)
        Initial state distribution
    A : np.ndarray, shape (K, K)
        Transition matrix
    B : np.ndarray, shape (T, K)
        Emission likelihood matrix

    Returns
    -------
    alpha : np.ndarray, shape (T, K)
        Scaled forward probabilities
    c : np.ndarray, shape (T,)
        Scaling factors
    log_likelihood : float
        Log p(X)
    """
    pi = np.asarray(pi, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    T, K = B.shape
    alpha = np.zeros((T, K), dtype=float)
    c = np.zeros(T, dtype=float)

    # t = 0
    alpha[0] = pi * B[0]
    c[0] = alpha[0].sum()
    if c[0] == 0:
        raise ValueError("Scaling factor c[0] is zero.")
    alpha[0] /= c[0]

    # t = 1, ..., T-1
    for t in range(1, T):
        for k in range(K):
            alpha[t, k] = B[t, k] * np.sum(alpha[t - 1] * A[:, k])

        c[t] = alpha[t].sum()
        if c[t] == 0:
            raise ValueError(f"Scaling factor c[{t}] is zero.")
        alpha[t] /= c[t]

    log_likelihood = np.sum(np.log(c))
    return alpha, c, log_likelihood


def backward_pass(A, B, c):
    """
    Scaled backward algorithm.

    Parameters
    ----------
    A : np.ndarray, shape (K, K)
    B : np.ndarray, shape (T, K)
    c : np.ndarray, shape (T,)

    Returns
    -------
    beta : np.ndarray, shape (T, K)
        Scaled backward probabilities
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    c = np.asarray(c, dtype=float)

    T, K = B.shape
    beta = np.zeros((T, K), dtype=float)

    # initialize
    beta[T - 1] = 1.0

    # backward recursion
    for t in range(T - 2, -1, -1):
        for i in range(K):
            beta[t, i] = np.sum(A[i, :] * B[t + 1, :] * beta[t + 1, :])

        beta[t] /= c[t + 1]

    return beta


def compute_gamma(alpha, beta):
    """
    Compute posterior state probabilities:
    gamma[t, k] = P(z_t = k | X)

    Parameters
    ----------
    alpha : np.ndarray, shape (T, K)
    beta : np.ndarray, shape (T, K)

    Returns
    -------
    gamma : np.ndarray, shape (T, K)
    """
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def compute_xi(alpha, beta, A, B):
    """
    Compute pairwise posterior:
    xi[t, i, j] = P(z_t = i, z_{t+1} = j | X)

    Parameters
    ----------
    alpha : np.ndarray, shape (T, K)
    beta : np.ndarray, shape (T, K)
    A : np.ndarray, shape (K, K)
    B : np.ndarray, shape (T, K)

    Returns
    -------
    xi : np.ndarray, shape (T-1, K, K)
    """
    T, K = alpha.shape
    xi = np.zeros((T - 1, K, K), dtype=float)

    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = alpha[t, i] * A[i, j] * B[t + 1, j] * beta[t + 1, j]

        denom = xi[t].sum()
        if denom == 0:
            raise ValueError(f"xi normalization failed at t={t}")
        xi[t] /= denom

    return xi


def forward_backward(X, pi, A, means, covs):
    """
    Run full Forward-Backward on toy Gaussian HMM.

    Parameters
    ----------
    X : np.ndarray, shape (T, D)
    pi : np.ndarray, shape (K,)
    A : np.ndarray, shape (K, K)
    means : np.ndarray, shape (K, D)
    covs : np.ndarray, shape (K, D, D)

    Returns
    -------
    results : dict
        Contains B, alpha, beta, gamma, xi, c, log_likelihood
    """
    B = compute_emission_likelihoods(X, means, covs)
    alpha, c, log_likelihood = forward_pass(pi, A, B)
    beta = backward_pass(A, B, c)
    gamma = compute_gamma(alpha, beta)
    xi = compute_xi(alpha, beta, A, B)

    return {
        "B": B,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "xi": xi,
        "c": c,
        "log_likelihood": log_likelihood
    }


def load_toy_data(filepath):
    """
    Load toy dataset saved by toy_data.py
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
    X, Z_true, params = load_toy_data("toy_sanity_check/data/sequence.npz")

    results = forward_backward(
        X=X,
        pi=params["pi"],
        A=params["A"],
        means=params["means"],
        covs=params["covs"]
    )

    gamma = results["gamma"]
    z_hat = np.argmax(gamma, axis=1)

    print("=== Toy Forward-Backward ===")
    print("X shape:", X.shape)
    print("gamma shape:", gamma.shape)
    print("xi shape:", results["xi"].shape)
    print("log_likelihood:", results["log_likelihood"])
    print("First 20 true states:     ", Z_true[:20])
    print("First 20 inferred states: ", z_hat[:20])
    print("First 5 gamma rows:\n", gamma[:5])
