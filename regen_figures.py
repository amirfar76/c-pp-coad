"""Regenerate 5G-NIDD and ColO-RAN figures from saved .npy results."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.plotting import make_figure, CONTEXT_AGNOSTIC_METHODS, CONTEXT_AWARE_METHODS

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
ALPHA = 0.1
T = 300

# 5G-NIDD: context-agnostic comparison only (PO-COAD violation shown here)
results_5g = np.load(os.path.join(FIG_DIR, '5gnidd_results.npy'), allow_pickle=True).item()
make_figure(results_5g, CONTEXT_AGNOSTIC_METHODS, alpha=ALPHA, T=T,
            outpath=os.path.join(FIG_DIR, '5GNIDD_no_context.pdf'))

# ColO-RAN: context-aware comparison only (C-PO-COAD violation shown here)
results_col = np.load(os.path.join(FIG_DIR, 'coloran_results.npy'), allow_pickle=True).item()
make_figure(results_col, CONTEXT_AWARE_METHODS, alpha=ALPHA, T=T,
            outpath=os.path.join(FIG_DIR, 'ColORAN_context.pdf'))

print('Done.')
