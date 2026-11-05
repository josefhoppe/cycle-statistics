"""
Summarizes the count estimator results as a LaTeX table (partial .tex file).

For each graph, the cycle count at length l = ceil(sqrt(n)) is compared across estimators.
For each estimator, the relative standard error of the mean (SEM / mean) and the bias
(mean / mean of HT-exact) are shown, as well as the average number of occurring cycles
of length l per spanning tree. The bias is replaced by an en-dash if the
estimator's mean does not differ significantly from the HT-exact mean
(paired t-test over the shared spanning trees, alpha = 0.05, Holm-corrected over the whole table).
"""

from math import ceil, floor, log10, sqrt
import pandas as pd
from significance import compare_to_reference, holm
import script_utils as su

snakemake = su.fix_smk()

REFERENCE = 'HT-exact'
ALPHA = 0.05
estimators = snakemake.params['estimators']
graphs = snakemake.params['graphs']

def format_sci(x: float) -> str:
    exponent = floor(log10(abs(x)))
    return f'${x / 10**exponent:.2f} \\cdot 10^{{{exponent}}}$'

graph_info = {}
stats = {}
p_values = {}
for graph in graphs:
    df_graph = pd.concat([pd.read_csv(f'results/count-estimator/{estimator}/{graph}.csv', index_col=0) for estimator in estimators], ignore_index=True)
    kind, n = df_graph[['kind', 'n']].iloc[0]
    l = ceil(sqrt(n))
    df_l = df_graph[df_graph.l == l]
    # average number of occurring cycles of length l per spanning tree (not recorded by IS)
    cycles_per_tree = df_l[df_l.estimator == REFERENCE]['count'].mean()
    graph_info[graph] = (kind, n, l, cycles_per_tree)
    stats[graph] = df_l.groupby('estimator').result.agg(['mean', 'std', 'sem'])
    for estimator in estimators:
        if estimator != REFERENCE and estimator in stats[graph].index:
            # all count estimators use the same spanning trees
            p_values[graph, estimator] = compare_to_reference(df_l, estimator, REFERENCE, 'result', 'tree', paired=True)
p_adjusted = holm(p_values)

rows = []
for graph in graphs:
    kind, n, l, cycles_per_tree = graph_info[graph]
    ref_mean = stats[graph].loc[REFERENCE, 'mean']
    cells = [kind, str(n), str(l), format_sci(ref_mean), f'{cycles_per_tree:.1f}']
    for estimator in estimators:
        if estimator not in stats[graph].index or pd.isna(stats[graph].loc[estimator, 'std']):
            cells += ['--', '--']
            continue
        mean, sem = stats[graph].loc[estimator, ['mean', 'sem']]
        cells.append(f'{sem / mean:.2f}')
        cells.append(f'{mean / ref_mean:.2f}' if p_adjusted.get((graph, estimator), 1) < ALPHA else '--')
    rows.append(' & '.join(cells) + r' \\')

fixed_cols = ['Graph', '$n$', '$l$', 'Count', 'Occ. cycles']

# Three-tiered header: approach (e.g. HT) on top, exactness (e.g. exact/approx) below, then the metric
groups = [] # list of [approach, first column index, number of columns]
for i, estimator in enumerate(estimators):
    approach = estimator.split('-')[0]
    if groups and groups[-1][0] == approach:
        groups[-1][2] += 2
    else:
        groups.append([approach, len(fixed_cols) + 2 * i, 2])

header_top = ' & '.join([''] * len(fixed_cols) + [f'\\multicolumn{{{width}}}{{c}}{{{approach}}}' for approach, _, width in groups]) + r' \\'
header_top_rules = ' '.join(f'\\cmidrule(lr){{{start + 1}-{start + width}}}' for _, start, width in groups)
header_mid = ' & '.join([''] * len(fixed_cols) + [f'\\multicolumn{{2}}{{c}}{{{estimator.split("-")[1] if "-" in estimator else ""}}}' for estimator in estimators]) + r' \\'
header_mid_rules = ' '.join(f'\\cmidrule(lr){{{len(fixed_cols) + 2 * i + 1}-{len(fixed_cols) + 2 * i + 2}}}' for i in range(len(estimators)))
header_bottom = ' & '.join(fixed_cols + ['$\\mathrm{SEM}/\\mu$', 'bias'] * len(estimators)) + r' \\'

with open(snakemake.output[0], 'w') as f:
    f.write('\\begin{tabular}{lrrlr' + 'rc' * len(estimators) + '}\n')
    f.write('\\toprule\n')
    f.write(header_top + '\n')
    f.write(header_top_rules + '\n')
    f.write(header_mid + '\n')
    f.write(header_mid_rules + '\n')
    f.write(header_bottom + '\n')
    f.write('\\midrule\n')
    for row in rows:
        f.write(row + '\n')
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}\n')
