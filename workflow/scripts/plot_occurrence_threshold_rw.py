import pandas as pd
import seaborn as sns
import sys
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
import script_utils as su

if len(sys.argv) > 1:
    # run for partial results: python workflow/scripts/plot_occurrence_threshold_rw.py results/occurrence-rw/*/*.csv occurrence_threshold_rw.pdf
    df_data = pd.concat([pd.read_csv(file) for file in sys.argv[1:-1]]).sort_values('n')
    output = sys.argv[-1]
else:
    snakemake = su.fix_smk()

    df_data = pd.concat([pd.read_csv(file) for file in snakemake.input]).sort_values('n')
    output = snakemake.output[0]



# For each run (identified by run, n, p), find the biggest l where occurred > samples
def compute_threshold(group):
    samples = group['samples'].iloc[0]
    above_threshold = group[group['occurred'] > samples]
    if len(above_threshold) == 0:
        return 0
    return above_threshold['l'].max()

# Group by n, p, and run to compute threshold for each run
threshold_data = df_data.groupby(['gt_kind', 'n', 'n_eff', 'run']).apply(compute_threshold).reset_index()
threshold_data.columns = ['gt_kind', 'n', 'n_eff', 'run', 'threshold_l']
threshold_data.sort_values('n_eff', ascending=True, inplace=True)

sns.set_theme()

plot = sns.relplot(
    data=threshold_data, kind="scatter", x="n_eff", y="threshold_l",
    height=4.4, aspect=1.61803, hue='gt_kind', palette='tab20'
) 

plot.set(xscale='log')
plot.set(yscale='log')
plot.set(xlabel=r'size ($n_{eff}$)')
plot.set(ylabel='threshold length $l$')

plot.savefig(output)