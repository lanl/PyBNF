"""Convergence diagnostics for MCMC samplers — pure R-hat / ESS math.

Extracted from ``algorithms/samplers/base.py`` (M2.2 move 5, ADR-0009): the
convergence diagnostics are substantial, already shared by all four samplers,
project-specific, and oracle-testable, so their pure numerical core lives here
as free functions — a peer of ``objective.py``, navigable, and importable by the
benchmark harness (M2.5) without reaching into ``samplers/``. The
instance-coupled glue (``report_convergence_diagnostics`` /
``check_convergence`` / ``_write_diagnostics``) and the PSet→array bridge
(``_param_vec``) stay on :class:`BayesianAlgorithm`, which delegates the math
here.

Implements the rank-normalized split-R-hat and bulk/tail effective sample size
of Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021), *Bayesian Analysis*
(the same conventions as Stan / ArviZ).

The data layout throughout: ``chain_history`` is a list of ``num_parallel``
per-chain histories, each a list of ``(n_dim,)`` parameter vectors in the
sampling space (see ``BayesianAlgorithm._param_vec``). The internal ``chains``
arrays are ``(n_chains, n_draws, n_dim)`` (or ``(n_chains, n_draws)`` for the
single-parameter ESS kernel).
"""

import numpy as np
from scipy import stats


def split_chains(chain_history, num_parallel):
    """Build the split-chains array used for R-hat and ESS.

    Uses the last 50% of each chain, then splits that window in two (doubling the
    chain count, catching within-chain non-stationarity). Returns an
    ``(2 * num_parallel, half, n_dim)`` array, or ``None`` if there is too little
    history (fewer than 20 recorded steps, or fewer than 5 per split half).
    """
    min_len = min(len(h) for h in chain_history)
    if min_len < 20:
        return None

    start = min_len // 2
    usable = min_len - start
    half = usable // 2
    if half < 5:
        return None

    chunks = []
    for j in range(num_parallel):
        chunk = chain_history[j][start:start + 2 * half]
        chunks.append(chunk[:half])
        chunks.append(chunk[half:2 * half])
    return np.array(chunks)  # (2N, half, n_dim)


def split_chain_rhat(chains):
    """
    Compute the potential scale reduction factor R-hat from an array of chains,
    using the Vehtari et al. (2021) convention R = sqrt(var_plus / W), where
    var_plus = ((n-1)/n) W + B/n. This is the form paired with the rank
    normalization and folding done by rhat() (and matches Stan / ArviZ).

    It deliberately omits the older Gelman & Rubin (1992) df-style correction
    sqrt((N+1)/N * (var_plus/W) - (n-1)/(N n)): that factor only inflates R-hat
    when the chains have not converged (it cancels to ~1 at convergence), making
    PyBNF report systematically higher R-hat than reference tools on the same
    chains -- harmful for cross-tool comparison -- without changing any
    convergence decision.

    chains: (N, n, d) array
    Returns: (d,) array of R-hat values
    """
    N, n, d = chains.shape
    mu_chains = np.mean(chains, axis=1)
    s2_chains = np.var(chains, axis=1, ddof=1)
    B = n * np.var(mu_chains, axis=0, ddof=1)
    W = np.mean(s2_chains, axis=0)
    var_plus = ((n - 1) / n) * W + (1.0 / n) * B
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(var_plus / W)


def rhat(chain_history, num_parallel):
    """
    Compute rank-normalized split-R-hat for each parameter (Vehtari, Gelman, Simpson,
    Carpenter & Burkner, 2021, Bayesian Analysis).

    Steps:
    1. Split each chain in half (doubles the number of chains, catches within-chain non-stationarity)
    2. Rank-normalize across all split chains (replaces values with normal scores of their ranks)
    3. Compute R-hat on both the ranked values and folded ranked values (detects scale differences)
    4. Return the element-wise maximum

    Returns a numpy array of shape (n_dim,) or None if insufficient data.
    """
    chains = split_chains(chain_history, num_parallel)
    if chains is None:
        return None

    N_split, n, d = chains.shape

    # Rank-normalize each parameter across all split chains
    ranked = np.empty_like(chains)
    for p in range(d):
        flat = chains[:, :, p].ravel()
        order = flat.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(flat) + 1)
        # Transform ranks to normal scores: Phi^{-1}((rank - 3/8) / (S + 1/4))
        z_scores = stats.norm.ppf((ranks - 0.375) / (len(flat) + 0.25))
        ranked[:, :, p] = z_scores.reshape(N_split, n)

    # R-hat on ranked values (detects location differences)
    rhat_rank = split_chain_rhat(ranked)

    # Folded R-hat: fold around the median to detect scale differences
    folded = np.abs(ranked - np.median(ranked, axis=(0, 1), keepdims=True))
    rhat_fold = split_chain_rhat(folded)

    # Return the element-wise max
    with np.errstate(invalid='ignore'):
        return np.fmax(rhat_rank, rhat_fold)


def ess_from_chains(chains):
    """
    Compute effective sample size from an (M, n) array of chains using
    FFT-based autocovariance and Geyer's initial positive sequence estimator.
    """
    M, n = chains.shape
    if n < 4:
        return float('nan')

    chain_means = np.mean(chains, axis=1)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    B_over_n = np.var(chain_means, ddof=1)
    var_hat = ((n - 1) / n) * W + B_over_n

    if var_hat < 1e-30:
        return float(M * n)

    # FFT autocovariance for each chain (biased estimator), averaged across chains
    npad = 1 << (2 * n - 1).bit_length()
    mean_acov = np.zeros(n)
    for m in range(M):
        x = chains[m] - chain_means[m]
        xpad = np.zeros(npad)
        xpad[:n] = x
        ft = np.fft.rfft(xpad)
        acov = np.fft.irfft(ft * np.conj(ft))[:n] / n
        mean_acov += acov
    mean_acov /= M

    # Combined autocorrelation: rho_t = 1 - (W - mean_acov[t]) / var_hat
    # Use the within-chain variance W (ddof=1) as the lag-0 anchor, per
    # Vehtari et al. (2021) / Stan. (Using mean_acov[0] == ((n-1)/n)*W instead
    # introduces an O(1/n) downward bias in ESS; negligible but non-standard.)
    # Geyer's initial positive sequence: sum consecutive pairs, stop at first negative pair
    tau = 0.0
    t = 1
    while t < n - 1:
        rho_t = 1.0 - (W - mean_acov[t]) / var_hat
        rho_t1 = 1.0 - (W - mean_acov[t + 1]) / var_hat
        P = rho_t + rho_t1
        if P < 0:
            break
        tau += P
        t += 2

    ess_val = M * n / max(1.0 + 2.0 * tau, 1.0)
    return max(ess_val, 1.0)


def ess(chain_history, num_parallel):
    """
    Compute bulk and tail effective sample size per Vehtari et al. (2021).

    Bulk ESS: computed on rank-normalized values (same transform as R-hat).
    Tail ESS: minimum ESS of the 5% and 95% quantile indicators.

    Returns (bulk_ess, tail_ess) arrays of shape (n_dim,) or (None, None).
    """
    chains = split_chains(chain_history, num_parallel)
    if chains is None:
        return None, None

    M, n, d = chains.shape
    bulk_ess = np.zeros(d)
    tail_ess = np.zeros(d)

    for p in range(d):
        param_chains = chains[:, :, p]  # (M, n)

        # Bulk ESS: rank-normalize then compute ESS
        flat = param_chains.ravel()
        order = flat.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(flat) + 1)
        z_scores = stats.norm.ppf((ranks - 0.375) / (len(flat) + 0.25))
        ranked_chains = z_scores.reshape(M, n)
        bulk_ess[p] = ess_from_chains(ranked_chains)

        # Tail ESS: ESS of quantile indicators
        q05 = np.quantile(flat, 0.05)
        q95 = np.quantile(flat, 0.95)
        ind_low = (param_chains <= q05).astype(float)
        ind_high = (param_chains >= q95).astype(float)
        ess_low = ess_from_chains(ind_low)
        ess_high = ess_from_chains(ind_high)
        tail_ess[p] = min(ess_low, ess_high)

    return bulk_ess, tail_ess
