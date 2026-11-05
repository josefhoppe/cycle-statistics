import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import script_utils as su

snakemake = su.fix_smk()

# Load and concatenate all input files
dfs = [pd.read_csv(f) for f in snakemake.input]
df_all = pd.concat(dfs, ignore_index=True)

# For each run (identified by run, n, p), find the biggest l where occurred > samples
def compute_threshold(group):
    samples = group['samples'].iloc[0]
    above_threshold = group[group['occurred'] > samples]
    if len(above_threshold) == 0:
        return 0
    return above_threshold['l'].max()

# Group by n, p, and run to compute threshold for each run
threshold_data = df_all.groupby(['n', 'p', 'run']).apply(compute_threshold).reset_index()
threshold_data.columns = ['n', 'p', 'run', 'threshold_l']

sns.set_theme('talk')

# Create loglog plot with seaborn relplot
g = sns.relplot(
    data=threshold_data,
    x='n',
    y='threshold_l',
    hue='p',
    palette='tab10',
    kind='line',
    errorbar='sd',
    aspect=1.4,
    height=5
)

# g.set(xscale='log', yscale='log')
g.set_axis_labels('graph size $n$', 'threshold length $l$')

plt.savefig(snakemake.output[0], bbox_inches='tight')
