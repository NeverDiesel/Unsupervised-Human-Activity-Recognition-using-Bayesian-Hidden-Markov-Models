import numpy as np


def sample_categorical(probs, rng):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(len(probs), p=probs)


def logsumexp(a):
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


def log_multivariate_normal_pdf(x, mean, cov):
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
        d * np.log(2.0 * np.pi)
        + logdet
        + diff.T @ inv_cov @ diff
    )


def compute_log_emissions(X, means, covs):
    X = np.asarray(X, dtype=float)
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)

    T = X.shape[0]
    K = means.shape[0]

    logB = np.zeros((T, K), dtype=float)
    for t in range(T):
        for k in range(K):
            logB[t, k] = log_multivariate_normal_pdf(X[t], means[k], covs[k])
    return logB


def forward_filter_log(X, pi, A, means, covs):
    """
    Compute filtered probabilities needed for FFBS.

    Returns
    -------
    filtered : (T, K)
        filtered[t, k] = p(z_t = k | x_1:t)
    loglik : float
        log p(X | params)
    """
    X = np.asarray(X, dtype=float)
    pi = np.asarray(pi, dtype=float)
    A = np.asarray(A, dtype=float)

    T = X.shape[0]
    K = len(pi)

    logB = compute_log_emissions(X, means, covs)
    log_alpha = np.zeros((T, K), dtype=float)

    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)

    log_alpha[0] = log_pi + logB[0]

    for t in range(1, T):
        for k in range(K):
            log_alpha[t, k] = logB[t, k] + logsumexp(log_alpha[t - 1] + log_A[:, k])

    filtered = np.zeros((T, K), dtype=float)
    for t in range(T):
        row = log_alpha[t] - logsumexp(log_alpha[t])
        filtered[t] = np.exp(row)

    loglik = logsumexp(log_alpha[T - 1])
    return filtered, loglik


def sample_states_ffbs(X, pi, A, means, covs, rng):
    """
    Sample latent state sequence Z | X, params
    using Forward-Filtering Backward-Sampling.
    """
    filtered, loglik = forward_filter_log(X, pi, A, means, covs)
    T, K = filtered.shape

    Z = np.zeros(T, dtype=int)

    # sample z_T from p(z_T | x_1:T)
    Z[T - 1] = sample_categorical(filtered[T - 1], rng)

    # sample backward
    for t in range(T - 2, -1, -1):
        probs = filtered[t] * A[:, Z[t + 1]]
        probs = probs / probs.sum()
        Z[t] = sample_categorical(probs, rng)

    return Z, loglik


def transition_counts(Z, K):
    counts = np.zeros((K, K), dtype=int)
    for t in range(len(Z) - 1):
        counts[Z[t], Z[t + 1]] += 1
    return counts


def sample_initial_distribution(Z, alpha_pi, rng):
    """
    pi | z_1 ~ Dirichlet(alpha_pi + one_hot(z_1))
    """
    K = len(alpha_pi)
    counts = np.zeros(K, dtype=float)
    counts[Z[0]] = 1.0
    return rng.dirichlet(alpha_pi + counts)


def sample_transition_matrix(Z, alpha_A, rng):
    """
    Each row A_i | Z ~ Dirichlet(alpha_A[i] + transition_counts[i])
    """
    alpha_A = np.asarray(alpha_A, dtype=float)
    K = alpha_A.shape[0]
    counts = transition_counts(Z, K)

    A = np.zeros((K, K), dtype=float)
    for i in range(K):
        A[i] = rng.dirichlet(alpha_A[i] + counts[i])
    return A, counts


def scatter_matrix(X):
    """
    Sum of outer products around the sample mean:
    S = sum (x_i - xbar)(x_i - xbar)^T
    """
    n, d = X.shape
    if n == 0:
        return np.zeros((d, d), dtype=float)
    xbar = X.mean(axis=0)
    centered = X - xbar
    return centered.T @ centered


def sample_wishart(df, scale, rng):
    """
    Sample W ~ Wishart(df, scale) using Bartlett decomposition.
    """
    scale = np.asarray(scale, dtype=float)
    d = scale.shape[0]

    L = np.linalg.cholesky(scale)

    A = np.zeros((d, d), dtype=float)
    for i in range(d):
        A[i, i] = np.sqrt(rng.chisquare(df - i))
        for j in range(i):
            A[i, j] = rng.normal()

    LA = L @ A
    W = LA @ LA.T
    return W


def sample_inverse_wishart(df, scale, rng):
    """
    Sigma ~ InvWishart(df, scale)
    by sampling W ~ Wishart(df, scale^{-1}) and returning W^{-1}.
    """
    scale = np.asarray(scale, dtype=float)
    inv_scale = np.linalg.inv(scale)
    W = sample_wishart(df, inv_scale, rng)
    Sigma = np.linalg.inv(W)
    return Sigma


def niw_posterior_params(Xk, m0, kappa0, nu0, Psi0):
    """
    Posterior NIW parameters for one state's assigned observations.
    """
    d = m0.shape[0]
    n = Xk.shape[0]

    if n == 0:
        return m0, kappa0, nu0, Psi0

    xbar = Xk.mean(axis=0)
    S = scatter_matrix(Xk)

    kappa_n = kappa0 + n
    m_n = (kappa0 * m0 + n * xbar) / kappa_n
    nu_n = nu0 + n

    diff = (xbar - m0).reshape(d, 1)
    Psi_n = Psi0 + S + (kappa0 * n / kappa_n) * (diff @ diff.T)

    return m_n, kappa_n, nu_n, Psi_n


def sample_gaussian_params_niw(Xk, m0, kappa0, nu0, Psi0, rng):
    """
    Sample:
      Sigma ~ InvWishart(nu_n, Psi_n)
      mu | Sigma ~ N(m_n, Sigma / kappa_n)
    """
    m_n, kappa_n, nu_n, Psi_n = niw_posterior_params(Xk, m0, kappa0, nu0, Psi0)

    Sigma = sample_inverse_wishart(nu_n, Psi_n, rng)
    mu = rng.multivariate_normal(mean=m_n, cov=Sigma / kappa_n)

    return mu, Sigma


def initialize_params(X, K, rng):
    """
    Simple random initialization for toy data.
    """
    T, D = X.shape

    Z = rng.integers(low=0, high=K, size=T)

    pi = np.ones(K) / K

    A = np.zeros((K, K), dtype=float)
    for i in range(K):
        A[i] = rng.dirichlet(np.ones(K))

    means = np.zeros((K, D), dtype=float)
    global_cov = np.cov(X.T) + 1e-3 * np.eye(D)
    covs = np.zeros((K, D, D), dtype=float)

    for k in range(K):
        idx = np.where(Z == k)[0]
        if len(idx) > 0:
            means[k] = X[idx].mean(axis=0)
        else:
            means[k] = X[rng.integers(0, T)]
        covs[k] = global_cov.copy()

    return Z, pi, A, means, covs


def default_priors(X, K):
    """
    Create weakly informative priors for toy Gaussian HMM.
    """
    X = np.asarray(X, dtype=float)
    _, D = X.shape

    alpha_pi = np.ones(K)
    alpha_A = np.ones((K, K))

    m0 = X.mean(axis=0)
    kappa0 = 0.5
    nu0 = D + 2
    Psi0 = np.cov(X.T) + 1.0 * np.eye(D)

    return {
        "alpha_pi": alpha_pi,
        "alpha_A": alpha_A,
        "m0": m0,
        "kappa0": kappa0,
        "nu0": nu0,
        "Psi0": Psi0,
    }


def gibbs_sample_toy_hmm(
    X,
    K=3,
    n_iter=300,
    burn_in=100,
    thin=1,
    seed=42,
    priors=None,
):
    """
    Block Gibbs sampler for toy Bayesian Gaussian HMM.

    Updates:
      1) Z | X, pi, A, means, covs    via FFBS
      2) pi | Z                       via Dirichlet
      3) A | Z                        via row-wise Dirichlet
      4) (mu_k, Sigma_k) | X, Z       via NIW

    Returns
    -------
    results : dict
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    T, D = X.shape

    if priors is None:
        priors = default_priors(X, K)

    alpha_pi = priors["alpha_pi"]
    alpha_A = priors["alpha_A"]
    m0 = priors["m0"]
    kappa0 = priors["kappa0"]
    nu0 = priors["nu0"]
    Psi0 = priors["Psi0"]

    # initialize
    Z, pi, A, means, covs = initialize_params(X, K, rng)

    saved_pi = []
    saved_A = []
    saved_means = []
    saved_covs = []
    saved_Z = []
    loglik_trace = []

    for it in range(n_iter):
        # 1) sample latent states
        Z, loglik = sample_states_ffbs(X, pi, A, means, covs, rng)

        # 2) sample pi
        pi = sample_initial_distribution(Z, alpha_pi, rng)

        # 3) sample A
        A, counts = sample_transition_matrix(Z, alpha_A, rng)

        # 4) sample Gaussian params for each state
        new_means = np.zeros((K, D), dtype=float)
        new_covs = np.zeros((K, D, D), dtype=float)

        for k in range(K):
            Xk = X[Z == k]
            mu_k, Sigma_k = sample_gaussian_params_niw(
                Xk, m0, kappa0, nu0, Psi0, rng
            )
            new_means[k] = mu_k
            new_covs[k] = Sigma_k

        means = new_means
        covs = new_covs

        loglik_trace.append(loglik)

        # save after burn-in and thinning
        if it >= burn_in and ((it - burn_in) % thin == 0):
            saved_pi.append(pi.copy())
            saved_A.append(A.copy())
            saved_means.append(means.copy())
            saved_covs.append(covs.copy())
            saved_Z.append(Z.copy())

        if (it + 1) % 25 == 0 or it == 0:
            print(
                f"Iter {it+1:>3d} | loglik = {loglik:.3f} | "
                f"state counts = {np.bincount(Z, minlength=K)}"
            )

    results = {
        "pi_samples": np.array(saved_pi),
        "A_samples": np.array(saved_A),
        "means_samples": np.array(saved_means),
        "covs_samples": np.array(saved_covs),
        "Z_samples": np.array(saved_Z),
        "loglik_trace": np.array(loglik_trace),
        "last_pi": pi,
        "last_A": A,
        "last_means": means,
        "last_covs": covs,
        "last_Z": Z,
        "priors": priors,
    }
    return results


def posterior_mean(arr):
    return np.mean(arr, axis=0)


def load_toy_data(filepath):
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
    X, Z_true, params = load_toy_data("data/toy/toy_sequence.npz")

    results = gibbs_sample_toy_hmm(
        X,
        K=3,
        n_iter=200,
        burn_in=80,
        thin=2,
        seed=42,
    )

    print("\n=== Posterior means ===")
    print("pi mean:\n", posterior_mean(results["pi_samples"]))
    print("A mean:\n", posterior_mean(results["A_samples"]))
    print("means mean:\n", posterior_mean(results["means_samples"]))