import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gmean
import script_utils as su

snakemake = su.fix_smk()

df_data = pd.concat([pd.read_csv(file) for file in snakemake.input])
y_axis = snakemake.params.get("y_axis", "estimated")
y_label = snakemake.params.get("y_label", None)
x_label = snakemake.params.get("x_label", None)
ratio = snakemake.params.get("ratio", "false") == 'True'
relative_inset = snakemake.params.get("relative_inset", "false") == 'True'
relative_inset_pos = snakemake.params.get("relative_inset_pos", 'lower right')
y_log = snakemake.params.get("y_log", "false") == 'True'

mean_per_length = df_data.groupby('len').agg({'correct': 'mean'}).rename({'correct': 'mean_per_length'},axis=1)
df_data = df_data.join(mean_per_length, 'len')


sns.set_theme('talk')
def plot_on_ax(df_data, ax, y_axis, ratio, y_log, x_label=None, y_label=None, title=None, cbar=False, kde=False):
    max_prob = df_data.correct.max() * 2
    min_prob = df_data.correct.min() / 2

    if ratio:
        df_data[y_axis] /= df_data['correct']
        if type(y_log) is not bool:
            y_log = df_data[y_axis].max() / df_data[y_axis].min() > y_log
    
    if len(df_data.same_cluster.unique()) > 1:
        df_data.sort_values('same_cluster', inplace=True)
    
    plot = sns.scatterplot(
        data=df_data, x="correct", y=y_axis, hue='len', palette='viridis', legend=False,
        style='same_cluster' if len(df_data.same_cluster.unique()) > 1 else None,
        ax=ax, rasterized=True,
    )

    if kde:
        df_data['weight'] = 0.1
        len_counts = df_data['len'].value_counts()
        for l in len_counts.index:
            if len_counts[l] > 10:
                df_data.loc[df_data['len'] == l, 'weight'] = 1 / len_counts[l]
        plot2 = sns.kdeplot(data=df_data, x="correct", y=y_axis,
                            log_scale=(True, bool(y_log)), cmap='Reds',
                            linewidths=1, ax=ax, weights='weight')


    normalize = mpl.colors.Normalize(df_data.len.min(), df_data.len.max())

    if cbar:
        cbar = plt.colorbar(cm.ScalarMappable(normalize, 'viridis'), ax=plot)
        cbar.set_label('cycle length')

    if ratio:
        plot.plot([min_prob, max_prob], [1, 1], 'r-')
        plot.set(xscale='log')
        if y_log:
            plot.set(yscale='log')
            if df_data[y_axis].max() < 10 and df_data[y_axis].min() > 0.1:
                coarse_ticks = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
                fine_ticks = [0.5, 0.7, 1.0, 1.5, 2.0]
                coarse_ticks = [f for f in coarse_ticks if f > df_data[y_axis].min() * 0.8 and f < df_data[y_axis].max() * 1.2]
                fine_ticks = [f for f in fine_ticks if f > df_data[y_axis].min() * 0.8 and f < df_data[y_axis].max() * 1.2]
                ticks = coarse_ticks if len(fine_ticks) > 4 else fine_ticks
                plot.set_yticks(ticks)
                plot.set_yticklabels(ticks)
        elif df_data[y_axis].max() < 1.1 and df_data[y_axis].min() > 0.9:
            ticks = [0.9, 0.95, 1.0, 1.05, 1.1]
            plot.set_yticks(ticks)
            plot.set_yticklabels(ticks)
                    
    else:
        plot.plot([min_prob, max_prob], [min_prob, max_prob], 'r-')
        plot.set(yscale='log', xscale='log')

    if y_label is not None:
        plot.set_ylabel(y_label)
    if x_label is not None:
        plot.set_xlabel(x_label)
    if title is not None:
        plot.set_title(title)
    return plot

if not relative_inset:
    fig, ax = plt.subplots()

    plot = plot_on_ax(df_data.copy(), ax, y_axis, ratio, y_log, x_label, y_label, cbar=True)
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.25,3.6))

    plot = plot_on_ax(df_data.copy(), ax1, y_axis, ratio, y_log, x_label, y_label, cbar=False)
    inset_plot = plot_on_ax(df_data.copy(), ax2, y_axis, True, 4, '', '', 'rel. error', cbar=True, kde=True)

fig.tight_layout(pad=0.2)
fig.savefig(snakemake.output[0], dpi=300)