import pandas as pd
import seaborn as sns
import sys
import script_utils as su

if len(sys.argv) > 1:
    # run for partial results: python workflow/scripts/plot_time.py results/time/*.csv time.pdf
    df_data = pd.concat([pd.read_csv(file) for file in sys.argv[1:-1]]).sort_values('n')
    output = sys.argv[-1]
else:
    snakemake = su.fix_smk()

    df_data = pd.concat([pd.read_csv(file) for file in snakemake.input])
    output = snakemake.output[0]

sns.set_theme()

plot = sns.relplot(
    data=df_data, kind="line", x="n", y="runtime", marker='o', style='p', hue='method',
    height=2.2, aspect=1.61803, errorbar='sd'
)

plot.set(xscale='log')
plot.set(yscale='log')
plot.set(xlabel=r'size ($n$)')
plot.set(ylabel='time [s]')

plot.savefig(output)