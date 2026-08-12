"""
Shared experiment runner: train models, split data, run all methods over T steps.

Data split per run:
  - 1/3 score training
  - 1/3 digital-twin training
  - 1/3 calibration+test pool

The calibration+test pool is randomly split into T test points and the
remaining N_cal points form the calibration pool. The FULL calibration pool
is available at every time step (for p-value validity). U_t ∈ {0,1} measures
whether real data was acquired (binary cost per step), matching the paper's CDAR.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .core import compute_sfdr, compute_power, compute_cdar, run_lord
from .methods import (
    proxy_pvalue, real_pvalue, active_pvalue, compute_gamma,
    fixed_threshold, lee2025
)


def _fit_rf(X_train, y_train, max_depth=10):
    if len(np.unique(y_train)) < 2:
        return _ConstantScorer()
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                 max_depth=max_depth, min_samples_leaf=5)
    clf.fit(X_train, y_train)
    return clf


class _ConstantScorer:
    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])


def _score_fn(clf):
    def fn(X):
        return clf.predict_proba(X)[:, 1]
    return fn


def _fit_gmm(X_normal, n_components=4):
    n_c = min(n_components, max(1, len(X_normal) // 20))
    gmm = GaussianMixture(n_components=n_c, covariance_type='diag',
                          max_iter=200, random_state=42, n_init=2,
                          reg_covar=1e-4)
    gmm.fit(X_normal)
    return gmm


def _fit_gmm_nc(X_normal, n_components):
    """Fit GMM with exact n_components (no per-sample cap) — for DT quality control."""
    n_c = min(n_components, max(1, len(X_normal) // 10))
    gmm = GaussianMixture(n_components=n_c, covariance_type='diag',
                          max_iter=200, random_state=42, n_init=2,
                          reg_covar=1e-4)
    gmm.fit(X_normal)
    return gmm


def run_experiment(
    X, y, contexts,
    T=300,
    delta=0.99,
    alpha=0.1,
    eta=1.0,
    lambda_val=5.0,
    n_syn=50,
    n_runs=100,
    seed=0,
    score_max_depth=10,
    dt_n_components=4,
    gmm_exclude_top_pct=0.0,
):
    """
    Run all methods on dataset for n_runs repetitions.

    Returns dict method_name -> {'sfdr': (n_runs,T), 'power': (n_runs,T), 'cdar': (n_runs,T)}
    """
    rng = np.random.default_rng(seed)
    N = len(X)
    unique_contexts = np.unique(contexts)

    method_names = ['COAD', 'PP-COAD', 'C-COAD', 'PO-COAD', 'C-PO-COAD',
                    'C-PP-COAD', 'Fixed', 'Lee2025']
    results = {m: {'sfdr': [], 'power': [], 'cdar': []} for m in method_names}

    for run in range(n_runs):
        idx = rng.permutation(N)
        n1 = N // 3

        score_idx = idx[:n1]
        dt_idx = idx[n1:2 * n1]
        cal_test_idx = idx[2 * n1:]

        X_score, y_score = X[score_idx], y[score_idx]
        X_dt, y_dt = X[dt_idx], y[dt_idx]
        ctx_score = contexts[score_idx]
        ctx_dt = contexts[dt_idx]

        X_ct, y_ct = X[cal_test_idx], y[cal_test_idx]
        ctx_ct = contexts[cal_test_idx]

        # Sample T test points
        n_ct = len(X_ct)
        if n_ct < T:
            test_local = rng.choice(n_ct, T, replace=True)
        else:
            test_local = rng.choice(n_ct, T, replace=False)

        X_test = X_ct[test_local]
        y_test = y_ct[test_local]
        ctx_test = ctx_ct[test_local]

        # Calibration pool = remaining (not used as test)
        remaining_mask = np.ones(n_ct, dtype=bool)
        remaining_mask[test_local] = False
        X_cal_all = X_ct[remaining_mask]
        y_cal_all = y_ct[remaining_mask]
        ctx_cal_all = ctx_ct[remaining_mask]

        # Only NORMAL calibration samples are valid for conformal p-values
        norm_mask = y_cal_all == 0
        X_cal_norm = X_cal_all[norm_mask]
        ctx_cal_norm = ctx_cal_all[norm_mask]

        # ── Train global score function ─────────────────────────────────────
        clf_global = _fit_rf(X_score, y_score, max_depth=score_max_depth)
        sf_global = _score_fn(clf_global)
        test_scores_global = sf_global(X_test)
        cal_scores_global = sf_global(X_cal_norm) if len(X_cal_norm) > 0 else np.array([0.5])

        # Global GMM (apply same top-score exclusion as per-context GMMs)
        X_dt_norm = X_dt[y_dt == 0]
        if gmm_exclude_top_pct > 0 and len(X_dt_norm) >= 50:
            sc_norm_g = sf_global(X_dt_norm)
            thr_g = np.percentile(sc_norm_g, 100.0 * (1.0 - gmm_exclude_top_pct))
            keep_g = sc_norm_g <= thr_g
            X_gmm_global = X_dt_norm[keep_g] if keep_g.sum() >= 20 else X_dt_norm
        else:
            X_gmm_global = X_dt_norm
        gmm_global = _fit_gmm_nc(X_gmm_global, dt_n_components) if len(X_gmm_global) >= 20 else None

        # ── Per-context models ──────────────────────────────────────────────
        clf_ctx, sf_ctx, gmm_ctx = {}, {}, {}
        gamma_ctx, thresh_ctx = {}, {}
        cal_pool_ctx = {}  # context → array of normal calibration scores

        for c in unique_contexts:
            # Score function
            mask_c = ctx_score == c
            if mask_c.sum() >= 20:
                clf_ctx[c] = _fit_rf(X_score[mask_c], y_score[mask_c],
                                     max_depth=score_max_depth)
            else:
                clf_ctx[c] = clf_global
            sf_ctx[c] = _score_fn(clf_ctx[c])

            # GMM per context.  When gmm_exclude_top_pct > 0, exclude the top
            # scored fraction of DT normals to simulate a DT trained on "typical"
            # operating conditions only, missing edge-case normals near the
            # anomaly boundary.  This creates the mismatch that causes C-PO-COAD
            # to produce biased proxy p-values while compute_gamma detects the
            # bias and protects C-PP-COAD via the active p-value gate.
            mask_dt_c = (ctx_dt == c) & (y_dt == 0)
            if mask_dt_c.sum() >= 20:
                X_dt_normal = X_dt[mask_dt_c]
                if gmm_exclude_top_pct > 0 and len(X_dt_normal) >= 50:
                    dt_sc = sf_ctx[c](X_dt_normal)
                    thr = np.percentile(dt_sc, 100.0 * (1.0 - gmm_exclude_top_pct))
                    keep = dt_sc <= thr
                    X_gmm = X_dt_normal[keep] if keep.sum() >= 20 else X_dt_normal
                else:
                    X_gmm = X_dt_normal
                gmm_ctx[c] = _fit_gmm_nc(X_gmm, dt_n_components)
            else:
                gmm_ctx[c] = gmm_global

            # Calibration pool per context (normal samples)
            mask_cal_c = ctx_cal_norm == c
            if mask_cal_c.sum() > 0:
                cal_pool_ctx[c] = sf_ctx[c](X_cal_norm[mask_cal_c])
            else:
                cal_pool_ctx[c] = cal_scores_global.copy()

            # γ(C) via KS deviation on held-out validation normal scores.
            val_scores_c = cal_pool_ctx[c]
            if len(val_scores_c) > 0 and gmm_ctx[c] is not None:
                gamma_ctx[c] = compute_gamma(sf_ctx[c], gmm_ctx[c], c,
                                              val_scores_c, lambda_val)
            else:
                gamma_ctx[c] = 0.5
            # Cap gamma so that scale * min_P_t < alpha_1 for the smallest calibration pool.
            # With gamma=0.5 (scale=2), contexts with n_cal>=100 satisfy this for delta<=0.99.
            gamma_ctx[c] = min(gamma_ctx[c], 0.50)

            # Fixed threshold: (1-alpha) quantile of normal training scores
            mask_train_norm = mask_c & (y_score == 0)
            if mask_train_norm.sum() > 0:
                train_sc = sf_ctx[c](X_score[mask_train_norm])
                thresh_ctx[c] = float(np.quantile(train_sc, 1 - alpha))
            else:
                thresh_ctx[c] = 0.5

        # Context-specific test scores
        test_scores_ctx = np.array([sf_ctx[ctx_test[t]](X_test[t:t+1])[0] for t in range(T)])

        # Global threshold
        norm_score_mask = y_score == 0
        if norm_score_mask.sum() > 0:
            thresh_ctx['global'] = float(np.quantile(sf_global(X_score[norm_score_mask]), 1 - alpha))
        else:
            thresh_ctx['global'] = 0.5

        # ── Step through T time steps ───────────────────────────────────────
        pval_store = {m: np.zeros(T) for m in method_names}
        U_store = {m: np.zeros(T, dtype=int) for m in method_names}

        for t in range(T):
            c = ctx_test[t]

            # --- C-PP-COAD ---
            s_t_ctx = test_scores_ctx[t]
            cal_c = cal_pool_ctx[c]
            if len(cal_c) == 0:
                cal_c = cal_scores_global
            P_t = real_pvalue(s_t_ctx, cal_c)
            if gmm_ctx[c] is not None:
                synth = gmm_ctx[c].sample(n_syn)[0]
                Q_t = proxy_pvalue(s_t_ctx, sf_ctx[c](synth))
            else:
                Q_t = float(rng.uniform())
            gam = gamma_ctx[c]
            Z_t, U_t = active_pvalue(Q_t, P_t, gam)
            pval_store['C-PP-COAD'][t] = Z_t
            U_store['C-PP-COAD'][t] = U_t

            # --- COAD (context-agnostic, real data always) ---
            P_t_global = real_pvalue(test_scores_global[t], cal_scores_global)
            pval_store['COAD'][t] = P_t_global
            U_store['COAD'][t] = 1

            # --- PP-COAD (context-agnostic, active p-value) ---
            if gmm_global is not None:
                synth_g = gmm_global.sample(n_syn)[0]
                Q_t_g = proxy_pvalue(test_scores_global[t], sf_global(synth_g))
            else:
                Q_t_g = float(rng.uniform())
            gam_g = min(float(np.mean(list(gamma_ctx.values()))), 0.999) if gamma_ctx else 0.5
            Z_t_g, U_t_g = active_pvalue(Q_t_g, P_t_global, gam_g)
            pval_store['PP-COAD'][t] = Z_t_g
            U_store['PP-COAD'][t] = U_t_g

            # --- C-COAD (context-aware real data only) ---
            pval_store['C-COAD'][t] = P_t
            U_store['C-COAD'][t] = 1

            # --- PO-COAD (synthetic only, context-agnostic) ---
            pval_store['PO-COAD'][t] = Q_t_g
            U_store['PO-COAD'][t] = 0

            # --- C-PO-COAD (synthetic only, context-aware) ---
            pval_store['C-PO-COAD'][t] = Q_t
            U_store['C-PO-COAD'][t] = 0

            # --- Fixed threshold ---
            thresh = thresh_ctx.get(c, thresh_ctx.get('global', 0.5))
            pval_store['Fixed'][t] = 0.0 if s_t_ctx >= thresh else 1.0
            U_store['Fixed'][t] = 0

            # --- Lee 2025 (accumulating real calibration) ---
            pval_store['Lee2025'][t] = P_t_global  # same pool as COAD
            U_store['Lee2025'][t] = 1

        # ── Run LORD + compute metrics for each method ──────────────────────
        for name in method_names:
            if name == 'Lee2025':
                # Lee et al. (2025) uses a raw per-step threshold without
                # sequential FDR control — reject whenever p-value < alpha.
                A_hats = (pval_store[name][:T] < alpha).astype(int)
            else:
                _, A_hats = run_lord(pval_store[name], delta, alpha, eta)
            sfdr = compute_sfdr(A_hats, y_test, delta, eta)
            power = compute_power(A_hats, y_test, delta, eta)
            cdar = compute_cdar(U_store[name], delta)
            results[name]['sfdr'].append(sfdr)
            results[name]['power'].append(power)
            results[name]['cdar'].append(cdar)

    for name in method_names:
        for metric in ['sfdr', 'power', 'cdar']:
            results[name][metric] = np.stack(results[name][metric])

    return results


def run_classifier_comparison(
    X, y, contexts,
    T=50,
    delta=0.95,
    alpha=0.1,
    eta=1.0,
    lambda_val=5.0,
    n_syn=50,
    n_runs=100,
    seed=0,
):
    """
    Compare C-PP-COAD and Fixed threshold with three score functions:
    Random Forest (supervised), One-Class SVM (semi-supervised), K-Means (unsupervised).

    Returns dict with keys 'C-PP-COAD (RF)', 'Fixed (RF)', 'C-PP-COAD (SVM)', etc.
    """
    rng = np.random.default_rng(seed)
    N = len(X)
    unique_contexts = np.unique(contexts)

    score_fn_names = ['RF', 'SVM', 'KMeans']
    method_suffixes = ['C-PP-COAD', 'Fixed']
    all_keys = [f'{m} ({s})' for s in score_fn_names for m in method_suffixes]
    results = {k: {'sfdr': [], 'power': [], 'cdar': []} for k in all_keys}

    for run in range(n_runs):
        idx = rng.permutation(N)
        n1 = N // 3
        score_idx = idx[:n1]
        dt_idx = idx[n1:2 * n1]
        cal_test_idx = idx[2 * n1:]

        X_score, y_score = X[score_idx], y[score_idx]
        X_dt, y_dt = X[dt_idx], y[dt_idx]
        ctx_score = contexts[score_idx]
        ctx_dt = contexts[dt_idx]
        X_ct, y_ct = X[cal_test_idx], y[cal_test_idx]
        ctx_ct = contexts[cal_test_idx]

        n_ct = len(X_ct)
        if n_ct < T:
            test_local = rng.choice(n_ct, T, replace=True)
        else:
            test_local = rng.choice(n_ct, T, replace=False)

        X_test = X_ct[test_local]
        y_test = y_ct[test_local]
        ctx_test = ctx_ct[test_local]

        remaining_mask = np.ones(n_ct, dtype=bool)
        remaining_mask[test_local] = False
        X_cal_norm = X_ct[remaining_mask][y_ct[remaining_mask] == 0]
        ctx_cal_norm = ctx_ct[remaining_mask][y_ct[remaining_mask] == 0]

        for score_name in score_fn_names:
            # Fit score function
            X_norm_score = X_score[y_score == 0]

            if score_name == 'RF':
                if len(np.unique(y_score)) >= 2:
                    clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                                 random_state=42, n_jobs=-1)
                    clf.fit(X_score, y_score)
                    def sf(X, _clf=clf):
                        return _clf.predict_proba(X)[:, 1]
                else:
                    def sf(X):
                        return np.full(len(X), 0.5)

            elif score_name == 'SVM':
                if len(X_norm_score) >= 10:
                    svm = OneClassSVM(kernel='rbf', nu=0.1)
                    svm.fit(X_norm_score)
                    def sf(X, _svm=svm):
                        return -_svm.decision_function(X)
                else:
                    def sf(X):
                        return np.full(len(X), 0.5)

            else:  # KMeans
                k = min(4, max(1, len(X_norm_score) // 20))
                km = KMeans(n_clusters=k, random_state=42, n_init=3)
                km.fit(X_norm_score)
                def sf(X, _km=km):
                    dists = _km.transform(X)
                    return dists.min(axis=1)

            # Context-aware models
            gmm_ctx = {}
            cal_pool_ctx = {}
            gamma_ctx = {}
            thresh_ctx = {}

            for c in unique_contexts:
                mask_dt_c = (ctx_dt == c) & (y_dt == 0)
                if mask_dt_c.sum() >= 20:
                    n_c = min(2, max(1, mask_dt_c.sum() // 20))
                    gmm = GaussianMixture(n_components=n_c,
                                         covariance_type='diag', random_state=42,
                                         reg_covar=1e-3, max_iter=200, n_init=2)
                    try:
                        gmm.fit(X_dt[mask_dt_c])
                        gmm_ctx[c] = gmm
                    except Exception:
                        gmm_ctx[c] = None
                else:
                    gmm_ctx[c] = None

                mask_cal_c = ctx_cal_norm == c
                if mask_cal_c.sum() > 0:
                    cal_pool_ctx[c] = sf(X_cal_norm[mask_cal_c])
                else:
                    cal_pool_ctx[c] = sf(X_cal_norm) if len(X_cal_norm) > 0 else np.array([0.5])

                if gmm_ctx[c] is not None and len(cal_pool_ctx[c]) > 0:
                    gamma_ctx[c] = compute_gamma(sf, gmm_ctx[c], c,
                                                  cal_pool_ctx[c], lambda_val)
                    gamma_ctx[c] = min(gamma_ctx[c], 0.50)
                else:
                    gamma_ctx[c] = 0.5

                mask_train_norm = (ctx_score == c) & (y_score == 0)
                if mask_train_norm.sum() > 0:
                    train_sc = sf(X_score[mask_train_norm])
                    thresh_ctx[c] = float(np.quantile(train_sc, 1 - alpha))
                else:
                    thresh_ctx[c] = 0.5

            test_scores = np.array([sf(X_test[t:t+1])[0] for t in range(T)])

            # Run C-PP-COAD and Fixed
            pval_cppoad = np.zeros(T)
            pval_fixed = np.zeros(T)
            U_cppoad = np.zeros(T, dtype=int)

            for t in range(T):
                c = ctx_test[t]
                s_t = test_scores[t]
                cal_c = cal_pool_ctx.get(c, np.array([0.5]))

                P_t = real_pvalue(s_t, cal_c)
                if gmm_ctx[c] is not None:
                    synth = gmm_ctx[c].sample(n_syn)[0]
                    Q_t = proxy_pvalue(s_t, sf(synth))
                else:
                    Q_t = float(rng.uniform())
                gam = gamma_ctx.get(c, 0.5)
                Z_t, U_t = active_pvalue(Q_t, P_t, gam)
                pval_cppoad[t] = Z_t
                U_cppoad[t] = U_t

                thresh = thresh_ctx.get(c, 0.5)
                pval_fixed[t] = 0.0 if s_t >= thresh else 1.0

            for method, pvals, U_arr in [
                (f'C-PP-COAD ({score_name})', pval_cppoad, U_cppoad),
                (f'Fixed ({score_name})',      pval_fixed,  np.zeros(T, dtype=int)),
            ]:
                _, A_hats = run_lord(pvals, delta, alpha, eta)
                results[method]['sfdr'].append(compute_sfdr(A_hats, y_test, delta, eta))
                results[method]['power'].append(compute_power(A_hats, y_test, delta, eta))
                results[method]['cdar'].append(compute_cdar(U_arr, delta))

    for k in all_keys:
        for metric in ['sfdr', 'power', 'cdar']:
            results[k][metric] = np.stack(results[k][metric])

    return results
