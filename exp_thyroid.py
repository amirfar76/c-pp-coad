"""
Thyroid disease experiments.

Produces:
  - classifier.pdf   : 2-panel, compares RF/SVM/k-means score functions (T=50, delta=0.95)
  - Thyroid_context.pdf : 3-panel, context-aware benchmark comparison (T=300, delta=0.99)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.datasets import load_thyroid
from src.runner import run_experiment, run_classifier_comparison
from src.plotting import (
    make_figure, make_two_panel_figure, CONTEXT_AWARE_METHODS
)

CACHE_PATH = os.path.join(os.path.dirname(__file__), 'data/thyroid/thyroid0387.data')
OUT_DIR    = os.path.join(os.path.dirname(__file__), 'figures')
FIG_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          'UAI_C_PP_COAD', 'Figures')

N_RUNS = 100
ALPHA  = 0.1


def main():
    print('Loading thyroid dataset...')
    X, y, contexts = load_thyroid(
        cache_path=CACHE_PATH,
        target_n=7200,
        target_anomaly_rate=0.07,
        seed=0,
    )
    print(f'  Dataset: N={len(X)}, anomaly_rate={y.mean():.3f}, '
          f'contexts={np.unique(contexts, return_counts=True)}')

    # ── Classifier comparison (T=50, delta=0.95) ─────────────────────────────
    print(f'\nClassifier comparison ({N_RUNS} runs × T=50)...')
    clf_results = run_classifier_comparison(
        X, y, contexts,
        T=50, delta=0.95, alpha=ALPHA, eta=1.0,
        lambda_val=5.0, n_syn=50,
        n_runs=N_RUNS, seed=42,
    )
    np.save(os.path.join(OUT_DIR, 'thyroid_clf_results.npy'), clf_results)

    # Method list: C-PP-COAD and Fixed for each score function
    clf_methods = [
        ('C-PP-COAD (RF)',    '#d62728', '-'),
        ('Fixed (RF)',        '#1f77b4', '--'),
        ('C-PP-COAD (SVM)',   '#ff7f0e', '-'),
        ('Fixed (SVM)',       '#2ca02c', '--'),
        ('C-PP-COAD (KMeans)','#9467bd', '-'),
        ('Fixed (KMeans)',    '#8c564b', '--'),
    ]
    outpath = os.path.join(OUT_DIR, 'classifier.pdf')
    make_two_panel_figure(clf_results, clf_methods, alpha=ALPHA, T=50,
                          outpath=outpath)
    # Copy to Figures directory
    import shutil
    shutil.copy(outpath, os.path.join(FIG_DIR, 'classifier.pdf'))
    print(f'  Saved classifier.pdf')

    # ── Context-aware benchmark comparison (T=300, delta=0.99) ───────────────
    print(f'\nBenchmark comparison ({N_RUNS} runs × T=300)...')
    results = run_experiment(
        X, y, contexts,
        T=300, delta=0.99, alpha=ALPHA, eta=1.0,
        lambda_val=5.0, n_syn=50,
        n_runs=N_RUNS, seed=42,
        score_max_depth=10,
        dt_n_components=4,
    )
    np.save(os.path.join(OUT_DIR, 'thyroid_results.npy'), results)

    outpath = os.path.join(OUT_DIR, 'Thyroid_context.pdf')
    make_figure(results, CONTEXT_AWARE_METHODS, alpha=ALPHA, T=300,
                outpath=outpath)
    shutil.copy(outpath, os.path.join(FIG_DIR, 'Thyroid_context.pdf'))
    print(f'  Saved Thyroid_context.pdf')


if __name__ == '__main__':
    main()
