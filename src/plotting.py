"""
Plotting functions matching the paper figure style exactly:
  - 3-panel figures: sFDR, power, CDAR over time
  - Thick line = FDR satisfied on average; thin = violated
  - α target shown as dashed horizontal line
  - Colors/markers matching existing paper figures
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'lines.linewidth': 1.2,
    'figure.dpi': 150,
})

# Method display config: (label, color, linestyle)
CONTEXT_AGNOSTIC_METHODS = [
    ('COAD',      '#1f77b4', '-'),
    ('PP-COAD',   '#ff7f0e', '--'),
    ('PO-COAD',   '#2ca02c', ':'),
    ('Fixed',     '#9467bd', '-.'),
    ('Lee2025',   '#e377c2', '--'),
    ('C-PP-COAD', '#d62728', '-'),
]

CONTEXT_AWARE_METHODS = [
    ('COAD',      '#1f77b4', '-'),
    ('C-COAD',    '#ff7f0e', '--'),
    ('C-PO-COAD', '#2ca02c', ':'),
    ('C-PP-COAD', '#d62728', '-'),
    ('Lee2025',   '#e377c2', '--'),
]


BASE_LW = 2.2
THIN_LW = 0.5
LEGEND_LW = 1.2


def make_figure(results, method_list, alpha=0.1, T=300, title='', outpath=None):
    """
    3-panel figure: sFDR | Power | CDAR over T steps.

    Parameters
    ----------
    results     : dict method_name → {sfdr: (n_runs,T), power: (n_runs,T), cdar: (n_runs,T)}
    method_list : list of (name, color, linestyle) tuples
    alpha       : FDR target level
    T           : number of time steps
    title       : figure title (used as suptitle)
    outpath     : save path (.pdf); if None, uses plt.show()
    """
    ts = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2))

    panel_labels = ['(a)', '(b)', '(c)']
    ylabels = [
        r'Average sFDR',
        r'Average Power',
        r'Average CDAR',
    ]

    for ax, panel_lbl, ylabel in zip(axes, panel_labels, ylabels):
        ax.set_xlabel('Time step $t$')
        ax.set_ylabel(ylabel)
        ax.text(0.03, 0.97, panel_lbl, transform=ax.transAxes,
                va='top', ha='left', fontsize=9)

    # sFDR panel: draw α target line
    axes[0].axhline(alpha, color='black', linewidth=0.8, linestyle='--')

    legend_handles = []
    legend_labels = []

    for name, color, ls in method_list:
        if name not in results:
            continue
        sfdr_mat = results[name]['sfdr']   # (n_runs, T)
        power_mat = results[name]['power']
        cdar_mat = results[name]['cdar']

        sfdr_mean = sfdr_mat.mean(axis=0)
        power_mean = power_mat.mean(axis=0)
        cdar_mean = cdar_mat.mean(axis=0)

        label = name.replace('Lee2025', 'FC-COAD').replace('Fixed', 'Stat-AD')
        # Per-time-step mask: True where sFDR is violated
        violated = sfdr_mean[:T] > alpha

        for ax, values in zip(axes, [sfdr_mean[:T], power_mean[:T], cdar_mean[:T]]):
            # Thin pass covers the entire trace (including violated segments)
            ax.plot(ts, values, color=color, ls=ls, lw=THIN_LW)
            # Thick overlay only where sFDR ≤ alpha (non-violated segments)
            vals_thick = np.ma.masked_where(violated, values)
            ax.plot(ts, vals_thick, color=color, ls=ls, lw=BASE_LW)

        legend_handles.append(
            mlines.Line2D([], [], color=color, ls=ls, lw=LEGEND_LW, label=label))
        legend_labels.append(label)

    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(bottom=-0.5)

    # Shared legend below figure (manually built to avoid duplicate handles)
    h_alpha = mlines.Line2D([], [], color='black', lw=0.8, ls='--', label=r'$\alpha$')
    all_handles = [h_alpha] + legend_handles
    all_labels = [r'$\alpha$'] + legend_labels
    fig.legend(all_handles, all_labels, loc='lower center',
               ncol=min(len(all_labels), 6),
               bbox_to_anchor=(0.5, -0.18),
               frameon=True, fontsize=7.5,
               handlelength=2.5)

    # Thick/thin legend note in sFDR panel
    thick_line = mlines.Line2D([], [], color='gray', lw=BASE_LW, ls='-', label='FDR satisfied')
    thin_line = mlines.Line2D([], [], color='gray', lw=THIN_LW, ls='-', label='FDR violated')
    axes[0].legend(handles=[thick_line, thin_line], loc='upper right',
                   fontsize=6.5, frameon=True)

    if title:
        fig.suptitle(title, fontsize=9, y=1.02)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=150)
        print(f'Saved figure: {outpath}')
    else:
        plt.show()
    plt.close(fig)


def make_two_panel_figure(results, method_list, alpha=0.1, T=300,
                          title='', outpath=None):
    """2-panel figure: sFDR | Power (used for classifier comparison, T=50)."""
    ts = np.arange(1, T + 1)

    # Larger fonts to compensate for wider panels in 2-panel layout
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2))

    panel_labels = ['(a)', '(b)']
    ylabels = [r'Average sFDR', r'Average Power']

    for ax, panel_lbl, ylabel in zip(axes, panel_labels, ylabels):
        ax.set_xlabel('Time step $t$')
        ax.set_ylabel(ylabel)
        ax.text(0.03, 0.97, panel_lbl, transform=ax.transAxes,
                va='top', ha='left', fontsize=11)

    axes[0].axhline(alpha, color='black', linewidth=0.8, linestyle='--')

    legend_handles = []
    legend_labels = []

    for name, color, ls in method_list:
        if name not in results:
            continue
        sfdr_mean = results[name]['sfdr'].mean(axis=0)
        power_mean = results[name]['power'].mean(axis=0)
        violated = sfdr_mean[:T] > alpha

        for ax, values in zip(axes, [sfdr_mean[:T], power_mean[:T]]):
            ax.plot(ts, values, color=color, ls=ls, lw=THIN_LW)
            vals_thick = np.ma.masked_where(violated, values)
            ax.plot(ts, vals_thick, color=color, ls=ls, lw=BASE_LW)

        label = name.replace('Lee2025', 'FC-COAD').replace('Fixed', 'Stat-AD')
        legend_handles.append(
            mlines.Line2D([], [], color=color, ls=ls, lw=LEGEND_LW, label=label))
        legend_labels.append(label)

    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1)

    h_alpha = mlines.Line2D([], [], color='black', lw=0.8, ls='--', label=r'$\alpha$')
    all_handles = [h_alpha] + legend_handles
    all_labels = [r'$\alpha$'] + legend_labels
    fig.legend(all_handles, all_labels, loc='lower center',
               ncol=min(len(all_labels), 4),
               bbox_to_anchor=(0.5, -0.28),
               frameon=True, fontsize=9,
               handlelength=2.5)

    thick_line = mlines.Line2D([], [], color='gray', lw=BASE_LW, ls='-', label='FDR satisfied')
    thin_line = mlines.Line2D([], [], color='gray', lw=THIN_LW, ls='-', label='FDR violated')
    axes[0].legend(handles=[thick_line, thin_line], loc='upper right',
                   fontsize=8.5, frameon=True)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=150)
        print(f'Saved figure: {outpath}')
    else:
        plt.show()
    plt.close(fig)

    # Restore default rcParams for other figures
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8,
    })
