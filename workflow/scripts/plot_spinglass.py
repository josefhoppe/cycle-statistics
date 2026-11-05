import pandas as pd
import networkx as nx
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np
import scipy

import script_utils as su
from spinglass import utils, generator

snakemake = su.fix_smk()

GRAPH = snakemake.wildcards.kind
SHORT = snakemake.params.short_plot
with open(snakemake.input.exact) as f:
    opt_h = np.array(literal_eval(f.read()))
df_results = pd.read_csv(snakemake.input.results, index_col=0)

G, J, pos_dict = generator.get_spinglass(GRAPH)

eig, eigv = scipy.linalg.eig(J.todense())
non_frust_energy = utils.energy(abs(J), np.ones(J.shape[0]))

MAX_LEN = 40

# df_results is expected to be present in the notebook (read earlier)
df = df_results.copy()

temps = df.loc['T']
finite_cols = temps[temps.notna() & np.isfinite(temps)].index.tolist()
zero_cols = temps[temps == 0].index.tolist()
inf_cols = temps[temps == np.inf].index.tolist()

# collect integer cycle lengths <= 12 that appear in the index
lengths = []
for idx in df.index:
    try:
        n = int(idx)
        if n <= MAX_LEN:
            lengths.append(idx)
    except Exception:
        pass

labels_plot = ['E'] + lengths
# filter labels that actually exist (E might be missing)
labels_plot = [lab for lab in labels_plot if lab in df.index]

n_plots = len(labels_plot)
cols = 3
rows = int(np.ceil(n_plots / cols)) if n_plots > 0 else 1
if SHORT:
    rows = 1
if rows > 4:
    rows = 4
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
axes = axes.flatten()

# compute x position for 'inf' (random) as max(finite_T)*5, fallback 1.0
x_inf = temps[finite_cols].max() * 5

for ax, label in zip(axes, labels_plot):
    # plot finite-temperature replicates individually as points
    for c in finite_cols:
        try:
            x = temps[c]
            y = pd.to_numeric(df.loc[label, c], errors='coerce')
        except Exception:
            continue
        ax.plot(x, y, marker='o', linestyle='None', color='C0', alpha=0.7)
    # plot inf (random) replicates at x_inf with slight jitter
    if len(inf_cols) > 0:
        for i, c in enumerate(inf_cols):
            y = pd.to_numeric(df.loc[label, c], errors='coerce')
            ax.plot(x_inf, y, marker='s', linestyle='None', color='C3', alpha=0.9)

    y_opt = pd.to_numeric(df.loc[label, 'opt'], errors='coerce')
    ax.axhline(y_opt, color='C2', linestyle='-', linewidth=1.5, label='opt (ref)')
    # horizontal reference lines: 'frust' and 'opt' columns (T=0 style)
    if label != 'E':
        y_fr = pd.to_numeric(df.loc[label, 'frust'], errors='coerce')
        ax.axhline(y_fr, color='C1', linestyle='--', linewidth=1.5, label='frust (ref)')
        ax.axhline(0, color='C3', linestyle='--', linewidth=1.5, label='not frust')
        ax.set_title(f'Cycle frust. l={label}')
    else:
        y_spect = -max(eig).real * len(G) / 2
        ax.axhline(y_spect, color='C1', linestyle='--', linewidth=1.5, label='spectral bound')
        ax.axhline(non_frust_energy, color='C3', linestyle='--', linewidth=1.5, label='not frust')

        # secondary y-axis for F = (E - non_frust_energy) / 2
        def e_to_f(y):
            return (y - non_frust_energy) / 2

        def f_to_e(y):
            return 2 * y + non_frust_energy

        secax = ax.secondary_yaxis('right', functions=(e_to_f, f_to_e))
        secax.set_ylabel('# frust. edges')
        ax.set_title('Energy')

    # formatting and ticks
    ax.set_xscale('log')
    # xticks: unique finite temps > 0 plus the 'inf' position
    xticks = sorted({float(t) for t in ((temps[c]) for c in finite_cols) if not np.isnan(t) and t > 0})

    ax.set_xticks(xticks + [x_inf])
    ax.set_xticklabels([str(t) for t in xticks] + ['inf'])

    ax.set_xlabel('T (log)')
    ax.set_ylabel('E' if label == 'E' else 'fraction of frust. cycles')
    ax.grid(True, which='both', ls=':', lw=0.5)
    ax.legend(loc='best', fontsize='small')

# hide any unused axes
for ax in axes[n_plots:]:  
    ax.axis('off')

plt.tight_layout()

plt.savefig(snakemake.output[0])


