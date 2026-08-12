"""
Experiment 4: UE Throughput Degradation Detection on ColO-RAN Dataset.

Context = O-RAN scheduling policy (0=Round-Robin, 1=Proportional Fair, 2=Max-Throughput).
Anomaly = UE throughput below 10th-percentile of normal baseline for that scheduling policy
          (models SLA violations due to resource starvation).
Methods: COAD, PP-COAD, C-COAD, PO-COAD, C-PO-COAD, C-PP-COAD, Stat-AD, FC-COAD.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from src.datasets import load_coloran
from src.runner import run_experiment
from src.plotting import (
    make_figure, CONTEXT_AGNOSTIC_METHODS, CONTEXT_AWARE_METHODS
)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/coloran')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')

T = 300
N_RUNS = 100
DELTA = 0.95
ALPHA = 0.1
ETA = 1.0
LAMBDA = 5.0
N_SYN = 50


def main():
    print('Loading ColO-RAN dataset...')
    X, y, contexts, thresholds = load_coloran(DATA_DIR, max_per_sched=8000, seed=0)
    print(f'  Dataset: N={len(X)}, anomaly_rate={y.mean():.3f}, '
          f'contexts={np.unique(contexts, return_counts=True)}')
    print(f'  Anomaly thresholds (dl_brate 10th pct): {thresholds}')

    print(f'Running experiments ({N_RUNS} runs × T={T} steps)...')
    results = run_experiment(
        X, y, contexts,
        T=T, delta=DELTA, alpha=ALPHA, eta=ETA,
        lambda_val=LAMBDA, n_syn=N_SYN,
        n_runs=N_RUNS, seed=42,
        score_max_depth=3,
        dt_n_components=1,
        gmm_exclude_top_pct=0.3,
    )

    np.save(os.path.join(OUT_DIR, 'coloran_results.npy'), results)
    print('Results saved.')

    make_figure(
        results, CONTEXT_AWARE_METHODS,
        alpha=ALPHA, T=T,
        outpath=os.path.join(OUT_DIR, 'ColORAN_context.pdf'),
    )
    print('Figures saved.')


if __name__ == '__main__':
    main()
