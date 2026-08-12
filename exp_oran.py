"""
Synthetic O-RAN conflict detection experiment.

Produces:
  - ORAN_0.95.pdf : 3-panel, context-aware benchmark comparison (T=300, delta=0.95)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import shutil
from src.datasets import generate_oran
from src.runner import run_experiment
from src.plotting import make_figure, CONTEXT_AWARE_METHODS

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'UAI_C_PP_COAD', 'Figures')

N_RUNS = 100
ALPHA  = 0.1


def main():
    print('Generating synthetic O-RAN dataset...')
    X, y, contexts = generate_oran(n_samples=10000, anomaly_rate=0.1, seed=0)
    print(f'  Dataset: N={len(X)}, anomaly_rate={y.mean():.3f}, '
          f'contexts={np.unique(contexts, return_counts=True)}')

    print(f'Running experiments ({N_RUNS} runs × T=300)...')
    results = run_experiment(
        X, y, contexts,
        T=300, delta=0.95, alpha=ALPHA, eta=1.0,
        lambda_val=5.0, n_syn=50,
        n_runs=N_RUNS, seed=42,
        score_max_depth=5,
        dt_n_components=1,
        gmm_exclude_top_pct=0.30,
    )
    np.save(os.path.join(OUT_DIR, 'oran_results.npy'), results)

    outpath = os.path.join(OUT_DIR, 'ORAN_0.95.pdf')
    make_figure(results, CONTEXT_AWARE_METHODS, alpha=ALPHA, T=300,
                outpath=outpath)
    shutil.copy(outpath, os.path.join(FIG_DIR, 'ORAN_0.95.pdf'))
    print(f'Saved ORAN_0.95.pdf')


if __name__ == '__main__':
    main()
