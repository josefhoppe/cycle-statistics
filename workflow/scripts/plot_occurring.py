"""
Plots the number of cycles per length that occur in the sampled spanning trees (one line per run).
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import script_utils as su

snakemake = su.fix_smk()

x = snakemake.params.get('x', 'l')
y = snakemake.params.get('y', 'occurred')
x_label = snakemake.params.get("x_label", None)
y_label = snakemake.params.get("y_label", None)
red_line_data = snakemake.params.get("red_line", None)

df_est = pd.read_csv(snakemake.input[0])

sns.set_theme('talk')

fig = plt.figure(figsize=(10,7))
ax = fig.add_axes([0.12, 0.12, 0.82, 0.82])

plot = sns.lineplot(
    data=df_est, x=x, y=y, units='run', legend=False,
    estimator=None, alpha=0.2,
    ax=ax, size=10
)
plot.set(yscale='log')
if y_label is not None:
    plot.set_ylabel(y_label)
if x_label is not None:
    plot.set_xlabel(x_label)

if red_line_data is not None:
    ax.plot(df_est[df_est.run == 0][x], df_est[df_est.run == 0][red_line_data], 'r-')

fig.tight_layout(pad=0.2)
fig.savefig(snakemake.output[0])
