"""
Experiment 3: Network Intrusion Detection on 5G-NIDD Dataset.

Context = protocol family (UDP/TCP/ICMP/other).
Anomaly = malicious traffic flow.
Methods: COAD, PP-COAD, C-COAD, PO-COAD, C-PO-COAD, C-PP-COAD, Stat-AD, FC-COAD.
Produces two figures (context-agnostic benchmarks, context-aware benchmarks).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from src.datasets import load_5gnidd
from src.runner import run_experiment
from src.plotting import (
    make_figure, CONTEXT_AGNOSTIC_METHODS, CONTEXT_AWARE_METHODS
)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/5g_nidd/Combined.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')

T = 300
N_RUNS = 100
DELTA = 0.95
ALPHA = 0.1
ETA = 1.0
LAMBDA = 5.0
N_SYN = 50


def main():
    print('Loading 5G-NIDD dataset...')
    X, y, contexts = load_5gnidd(
        DATA_PATH,
        subsample_malicious=True,
        target_anomaly_rate=0.10,
        max_samples=18000,
        seed=0,
    )
    print(f'  Dataset: N={len(X)}, anomaly_rate={y.mean():.3f}, '
          f'contexts={np.unique(contexts, return_counts=True)}')

    print(f'Running experiments ({N_RUNS} runs × T={T} steps)...')
    results = run_experiment(
        X, y, contexts,
        T=T, delta=DELTA, alpha=ALPHA, eta=ETA,
        lambda_val=LAMBDA, n_syn=N_SYN,
        n_runs=N_RUNS, seed=42,
        score_max_depth=1,
        dt_n_components=1,
        gmm_exclude_top_pct=0.20,
    )

    np.save(os.path.join(OUT_DIR, '5gnidd_results.npy'), results)
    print('Results saved.')

    # Figure: context-agnostic benchmarks
    make_figure(
        results, CONTEXT_AGNOSTIC_METHODS,
        alpha=ALPHA, T=T,
        outpath=os.path.join(OUT_DIR, '5GNIDD_no_context.pdf'),
    )
    # Figure: context-aware benchmarks
    make_figure(
        results, CONTEXT_AWARE_METHODS,
        alpha=ALPHA, T=T,
        outpath=os.path.join(OUT_DIR, '5GNIDD_context.pdf'),
    )
    print('Figures saved.')


if __name__ == '__main__':
    main()
