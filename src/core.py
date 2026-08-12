"""
LORD algorithm for decaying-memory sFDR control, plus metric computation.
"""
import numpy as np


def lord_zetas(T_max):
    """Compute the zeta sequence for LORD: zeta_t ∝ log(min(t,2)) / (t * exp(sqrt(log t))), normalized."""
    t = np.arange(1, T_max + 1, dtype=float)
    raw = np.log(np.minimum(t, 2)) / (t * np.exp(np.sqrt(np.log(t))))
    return raw / raw.sum()


def run_lord(pvalues, delta, alpha, eta=1.0):
    """
    Apply LORD with decaying-memory sFDR to a sequence of p-values.

    Formula: alpha_t = alpha*eta*zeta_tilde_t + alpha*sum_j delta^{t-rho_j} * zeta_{t-j}
    where rho_j is the time of the j-th rejection (1-indexed count).

    Returns
    -------
    alpha_ts : (T,) array of thresholds
    A_hats   : (T,) binary rejection decisions
    """
    T = len(pvalues)
    zetas = lord_zetas(T)
    zeta_tilde = np.maximum(zetas, 1 - delta)

    alpha_ts = np.zeros(T)
    A_hats = np.zeros(T, dtype=int)
    rejection_times = []  # list of rho_j (0-indexed times)

    for t in range(T):
        alpha_t = alpha * eta * zeta_tilde[t]
        for j_idx, rho_j in enumerate(rejection_times):
            j = j_idx + 1          # 1-indexed rejection count
            lag_time = t - rho_j   # time since j-th rejection
            idx_zeta = t - j       # zeta index for boost term
            if lag_time >= 0 and idx_zeta >= 0:
                alpha_t += alpha * (delta ** lag_time) * zetas[idx_zeta]
        alpha_ts[t] = alpha_t
        if pvalues[t] <= alpha_t:
            A_hats[t] = 1
            rejection_times.append(t)

    return alpha_ts, A_hats


def compute_sfdr(A_hats, A_true, delta, eta=1.0):
    """Compute decaying-memory sFDR at each time step (single run)."""
    T = len(A_hats)
    sfdr = np.zeros(T)
    for t in range(T):
        w = np.array([delta ** (t - tau) for tau in range(t + 1)])
        false_disc = float(w @ (A_hats[:t+1] * (1 - A_true[:t+1])))
        total_disc = float(w @ A_hats[:t+1])
        sfdr[t] = false_disc / (total_disc + eta)
    return sfdr


def compute_power(A_hats, A_true, delta, eta=1.0):
    """Compute decaying-memory power at each time step (single run)."""
    T = len(A_hats)
    power = np.zeros(T)
    for t in range(T):
        w = np.array([delta ** (t - tau) for tau in range(t + 1)])
        true_det = float(w @ (A_hats[:t+1] * A_true[:t+1]))
        total_anom = float(w @ A_true[:t+1])
        power[t] = true_det / (total_anom + eta)
    return power


def compute_cdar(U_ts, delta):
    """Compute cumulative data acquisition rate at each time step (single run)."""
    T = len(U_ts)
    cdar = np.zeros(T)
    for t in range(T):
        w = np.array([delta ** (t - tau) for tau in range(t + 1)])
        cdar[t] = float(w @ U_ts[:t+1])
    return cdar
