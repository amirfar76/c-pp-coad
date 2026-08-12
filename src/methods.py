"""
C-PP-COAD and all benchmark anomaly detection methods.
"""
import numpy as np
from .core import run_lord


# ── p-value helpers ────────────────────────────────────────────────────────────

def proxy_pvalue(score_test, scores_synth):
    """Q_t = #{i : s(X̃_i) >= s(X_t)} / (n_tilde + 1)  [paper Eq. proxy_pval]."""
    return float(np.sum(scores_synth >= score_test)) / (len(scores_synth) + 1)


def real_pvalue(score_test, scores_cal):
    """P_t = (#{i : s(X_i) >= s(X_t)} + 1) / (n + 1)  [standard conformal p-value]."""
    return (float(np.sum(scores_cal >= score_test)) + 1.0) / (len(scores_cal) + 1)


# ── γ(C) computation (KS-based) ────────────────────────────────────────────────

def compute_gamma(score_fn, gmm, context, val_scores, lambda_val):
    """
    Compute γ(C) via one-sided KS deviation.

    Parameters
    ----------
    score_fn : callable(X) → scores array
    gmm      : fitted GMM for this context
    context  : context label (for info, unused here)
    val_scores : anomaly scores for validation inliers from this context
    lambda_val : λ tuning parameter in γ(C) = exp(-λ · max(0, D(C)))
    """
    n_val = len(val_scores)
    if n_val == 0:
        return 1.0
    n_syn = max(n_val * 10, 500)
    synth = gmm.sample(n_syn)[0]
    synth_scores = score_fn(synth)

    # Compute proxy p-values for each validation point
    p_vals = np.array([
        (np.sum(synth_scores >= s) + 1e-10) / (n_syn + 1)
        for s in val_scores
    ])
    p_vals_sorted = np.sort(p_vals)
    n = len(p_vals_sorted)
    uniform_cdf = np.arange(1, n + 1) / n
    D = np.max(uniform_cdf - p_vals_sorted)  # one-sided KS: F̂(p) - p
    return float(np.exp(-lambda_val * max(0.0, D)))


# ── Single-step active p-value ─────────────────────────────────────────────────

def active_pvalue(Q_t, P_t, gamma_c):
    """
    Compute active p-value Z_t and acquisition indicator U_t.

    Z_t = (1-U_t)*Q_t + U_t*(1-gamma_c)^{-1}*P_t  (Xu et al. 2025)

    Z_t can be > 1 (valid: P[Z_t <= u] <= u holds even for Z_t > 1).
    """
    p_real = float(np.clip(1.0 - gamma_c * Q_t, 0.0, 1.0))
    U_t = int(np.random.random() < p_real)
    if U_t == 0:
        Z_t = float(Q_t)
    else:
        scale = 1.0 / max(1.0 - gamma_c, 1e-6)
        Z_t = scale * float(P_t)
    return Z_t, U_t


# ── Core online runner ─────────────────────────────────────────────────────────

def run_online(pvalues, U_ts, delta, alpha, eta):
    """Run LORD on pvalues, return (sfdr_vals, power_vals, cdar_vals) per timestep."""
    from .core import run_lord, compute_sfdr, compute_power, compute_cdar
    _, A_hats = run_lord(pvalues, delta, alpha, eta)
    return A_hats, U_ts


# ── C-PP-COAD ──────────────────────────────────────────────────────────────────

def cppcoad(
    test_scores, contexts, labels,
    score_fns,      # dict context → callable(X) → float
    gmms,           # dict context → fitted GMM
    cal_pool,       # dict context → array of calibration scores (real)
    gamma_cs,       # dict context → γ(C)
    cal_per_step,   # int n per time step per context
    n_syn,          # int ñ synthetic samples per step
    delta, alpha, eta=0.01
):
    """
    C-PP-COAD: context-aware prediction-powered conformal online anomaly detection.

    Parameters
    ----------
    test_scores : (T,) anomaly scores for test sequence
    contexts    : (T,) context labels for each time step
    labels      : (T,) true anomaly labels {0,1}
    score_fns   : dict c → score function
    gmms        : dict c → GMM
    cal_pool    : dict c → array of normal scores for real calibration
    gamma_cs    : dict c → γ(C)
    cal_per_step: number of real calibration scores per step
    n_syn       : number of synthetic samples per step
    delta, alpha, eta : LORD hyperparameters
    """
    T = len(test_scores)
    pvalues = np.zeros(T)
    U_ts = np.zeros(T, dtype=int)
    cal_idx = {c: 0 for c in cal_pool}

    for t in range(T):
        c = contexts[t]
        s_t = test_scores[t]

        # --- synthetic p-value Q_t ---
        if c in gmms and gmms[c] is not None:
            synth = gmms[c].sample(n_syn)[0]
            synth_scores = score_fns[c](synth)
            Q_t = proxy_pvalue(s_t, synth_scores)
        else:
            Q_t = np.random.uniform()

        # --- real p-value P_t (from calibration pool) ---
        if c in cal_pool and len(cal_pool[c]) > 0:
            pool = cal_pool[c]
            idx = cal_idx[c]
            end = min(idx + cal_per_step, len(pool))
            real_scores = pool[idx:end]
            cal_idx[c] = end % len(pool)  # wrap around
            if len(real_scores) == 0:
                real_scores = pool[:cal_per_step]
            P_t = real_pvalue(s_t, real_scores)
        else:
            P_t = Q_t

        gamma_c = gamma_cs.get(c, 1.0)
        Z_t, U_t = active_pvalue(Q_t, P_t, gamma_c)
        pvalues[t] = Z_t
        U_ts[t] = U_t

    _, A_hats = run_lord(pvalues, delta, alpha, eta)
    return A_hats, U_ts


def coad(
    test_scores, contexts, labels,
    cal_pool, cal_per_step,
    delta, alpha, eta=0.01
):
    """COAD: conformal online anomaly detection (real data only, no context)."""
    T = len(test_scores)
    pvalues = np.zeros(T)
    U_ts = np.ones(T, dtype=int)
    all_cal = np.concatenate(list(cal_pool.values()))
    idx = 0
    for t in range(T):
        end = min(idx + cal_per_step, len(all_cal))
        cal = all_cal[idx:end]
        idx = end % len(all_cal)
        if len(cal) == 0:
            cal = all_cal[:cal_per_step]
        pvalues[t] = real_pvalue(test_scores[t], cal)
    _, A_hats = run_lord(pvalues, delta, alpha, eta)
    return A_hats, U_ts


def ppcoad(
    test_scores, contexts, labels,
    score_fn_global, gmm_global,
    cal_pool, cal_per_step, n_syn,
    gamma, delta, alpha, eta=0.01
):
    """PP-COAD: context-agnostic C-PP-COAD."""
    T = len(test_scores)
    pvalues = np.zeros(T)
    U_ts = np.zeros(T, dtype=int)
    all_cal = np.concatenate(list(cal_pool.values()))
    idx = 0
    for t in range(T):
        s_t = test_scores[t]
        if gmm_global is not None:
            synth = gmm_global.sample(n_syn)[0]
            synth_scores = score_fn_global(synth)
            Q_t = proxy_pvalue(s_t, synth_scores)
        else:
            Q_t = np.random.uniform()
        end = min(idx + cal_per_step, len(all_cal))
        cal = all_cal[idx:end]
        idx = end % len(all_cal)
        if len(cal) == 0:
            cal = all_cal[:cal_per_step]
        P_t = real_pvalue(s_t, cal)
        Z_t, U_t = active_pvalue(Q_t, P_t, gamma)
        pvalues[t] = Z_t
        U_ts[t] = U_t
    _, A_hats = run_lord(pvalues, delta, alpha, eta)
    return A_hats, U_ts


def ccoad(
    test_scores, contexts, labels,
    cal_pool, cal_per_step,
    delta, alpha, eta=0.01
):
    """C-COAD: context-aware real-data-only COAD."""
    T = len(test_scores)
    pvalues = np.zeros(T)
    U_ts = np.ones(T, dtype=int)
    cal_idx = {c: 0 for c in cal_pool}
    for t in range(T):
        c = contexts[t]
        if c in cal_pool and len(cal_pool[c]) > 0:
            pool = cal_pool[c]
            idx = cal_idx[c]
            end = min(idx + cal_per_step, len(pool))
            cal = pool[idx:end]
            cal_idx[c] = end % len(pool)
            if len(cal) == 0:
                cal = pool[:cal_per_step]
            pvalues[t] = real_pvalue(test_scores[t], cal)
        else:
            all_cal = np.concatenate(list(cal_pool.values()))
            pvalues[t] = real_pvalue(test_scores[t], all_cal[:cal_per_step])
    _, A_hats = run_lord(pvalues, delta, alpha, eta)
    return A_hats, U_ts


def pocoad(
    test_scores, contexts, labels,
    score_fn_global, gmm_global,
    n_syn, delta, alpha, eta=0.01
):
    """PO-COAD: synthetic-data-only, context-agnostic."""
    T = len(test_scores)
    pvalues = np.zeros(T)
    U_ts = np.zeros(T, dtype=int)
    for t in range(T):
        if gmm_global is not None:
            synth = gmm_global.sample(n_syn)[0]
            synth_scores = score_fn_global(synth)
            pvalues[t] = proxy_pvalue(test_scores[t], synth_scores)
        else:
            pvalues[t] = np.random.uniform()
    _, A_hats = run_lord(pvalues, delta, alpha, eta)
    return A_hats, U_ts


def cpocoad(
    test_scores, contexts, labels,
    score_fns, gmms,
    n_syn, delta, alpha, eta=0.01
):
    """C-PO-COAD: synthetic-data-only, context-aware."""
    T = len(test_scores)
    pvalues = np.zeros(T)
    U_ts = np.zeros(T, dtype=int)
    for t in range(T):
        c = contexts[t]
        if c in gmms and gmms[c] is not None:
            synth = gmms[c].sample(n_syn)[0]
            synth_scores = score_fns[c](synth)
            pvalues[t] = proxy_pvalue(test_scores[t], synth_scores)
        else:
            pvalues[t] = np.random.uniform()
    _, A_hats = run_lord(pvalues, delta, alpha, eta)
    return A_hats, U_ts


def fixed_threshold(test_scores, contexts, labels, thresholds_per_ctx):
    """Fixed-threshold detector: reject if score > (1-alpha)-quantile from training."""
    T = len(test_scores)
    A_hats = np.zeros(T, dtype=int)
    U_ts = np.zeros(T, dtype=int)
    for t in range(T):
        c = contexts[t]
        thresh = thresholds_per_ctx.get(c, thresholds_per_ctx.get('global', 0.5))
        A_hats[t] = int(test_scores[t] >= thresh)
    return A_hats, U_ts


def lee2025(
    test_scores, contexts, labels,
    cal_pool, cal_per_step,
    delta, alpha, eta=0.01
):
    """
    Lee et al. 2025 baseline: conformal anomaly detection with FDR guarantees.
    Uses accumulated real calibration data (rolling pool grows over time),
    making it more data-efficient per step than COAD but still real-data-only.
    """
    T = len(test_scores)
    pvalues = np.zeros(T)
    U_ts = np.ones(T, dtype=int)
    # Build initial calibration pool (same as COAD start)
    all_cal = list(np.concatenate(list(cal_pool.values())))
    history = all_cal[:cal_per_step]  # start with initial batch
    idx = cal_per_step

    for t in range(T):
        # Add a fresh batch to history (growing window)
        end = min(idx + cal_per_step, len(all_cal))
        if end > idx:
            history = history + list(all_cal[idx:end])
            idx = end
        pvalues[t] = real_pvalue(test_scores[t], np.array(history))

    _, A_hats = run_lord(pvalues, delta, alpha, eta)
    return A_hats, U_ts
