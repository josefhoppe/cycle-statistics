"""
Summarizes the balance estimator results as a LaTeX table (partial .tex file).

For each graph, the balance at cycle length l = ceil(sqrt(n)) is compared across estimators.
Standard deviations are printed in red if the estimator's mean differs significantly from the
HT-exact mean (alpha = 0.05, Holm-corrected over the whole table). Estimators sharing the
reference's spanning trees are compared with a paired t-test, others with Welch's t-test.
"""

from math import ceil, sqrt
import pandas as pd
from significance import compare_to_reference, holm
import script_utils as su

snakemake = su.fix_smk()

REFERENCE = 'HT-exact'
ALPHA = 0.05
# estimators that do not use the same spanning trees as the reference
UNPAIRED = ['Giscard']
estimators = snakemake.params['estimators']
graphs = snakemake.params['graphs']

df = pd.concat([pd.read_csv(f, index_col=0) for f in snakemake.input], ignore_index=True)

graph_info = {}
stats = {}
p_values = {}
for graph in graphs:
    df_graph = df[df.kind == graph]
    n, n_pos, n_neg = df_graph[['n', 'n_pos', 'n_neg']].iloc[0]
    l = ceil(sqrt(n))
    graph_info[graph] = (n, l, n_pos, n_neg)
    df_l = df_graph[df_graph.l == l]
    stats[graph] = df_l.groupby('estimator').balance.agg(['mean', 'std'])
    for estimator in estimators:
        if estimator != REFERENCE and estimator in stats[graph].index:
            p_values[graph, estimator] = compare_to_reference(df_l, estimator, REFERENCE, 'balance', 'sample',
                paired=estimator not in UNPAIRED)
p_adjusted = holm(p_values)

rows = []
for graph in graphs:
    n, l, n_pos, n_neg = graph_info[graph]
    cells = [graph, str(n), str(l), str(n_pos), str(n_neg), f'{stats[graph].loc[REFERENCE, "mean"]:.2f}']
    for estimator in estimators:
        if estimator not in stats[graph].index or pd.isna(stats[graph].loc[estimator, 'std']):
            cells.append('--')
            continue
        cell = f'{stats[graph].loc[estimator, "std"]:.2f}'
        if p_adjusted.get((graph, estimator), 1) < ALPHA:
            cell = f'\\textcolor{{red}}{{{cell}}}'
        cells.append(cell)
    rows.append(' & '.join(cells) + r' \\')

fixed_cols = ['Graph', '$n$', '$l$', '$|E^+|$', '$|E^-|$', 'Balance']

# Two-tiered header: approach (e.g. HT) on top, exactness (e.g. exact/approx) below
groups = [] # list of [approach, first column index, number of columns]
for i, estimator in enumerate(estimators):
    approach = estimator.split('-')[0]
    if groups and groups[-1][0] == approach:
        groups[-1][2] += 1
    else:
        groups.append([approach, len(fixed_cols) + i, 1])

header_top = ' & '.join([''] * len(fixed_cols) + [f'\\multicolumn{{{width}}}{{c}}{{{approach}}}' for approach, _, width in groups]) + r' \\'
header_rules = ' '.join(f'\\cmidrule(lr){{{start + 1}-{start + width}}}' for _, start, width in groups)
header_bottom = ' & '.join(fixed_cols + [estimator.split('-')[1] if '-' in estimator else '' for estimator in estimators]) + r' \\'

with open(snakemake.output[0], 'w') as f:
    f.write('\\begin{tabular}{l' + 'r' * (5 + len(estimators)) + '}\n')
    f.write('\\toprule\n')
    f.write(header_top + '\n')
    f.write(header_rules + '\n')
    f.write(header_bottom + '\n')
    f.write('\\midrule\n')
    for row in rows:
        f.write(row + '\n')
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}\n')
