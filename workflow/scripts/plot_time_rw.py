import pandas as pd
import seaborn as sns
import sys
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
import script_utils as su

if len(sys.argv) > 1:
    # run for partial results: python workflow/scripts/plot_time_rw.py results/time-rw/*/*.csv time_rw.pdf
    df_data = pd.concat([pd.read_csv(file) for file in sys.argv[1:-1]]).sort_values('m')
    output = sys.argv[-1]
else:
    snakemake = su.fix_smk()

    df_data = pd.concat([pd.read_csv(file) for file in snakemake.input]).sort_values('m')
    output = snakemake.output[0]

sns.set_theme()

plot = sns.relplot(
    data=df_data, kind="scatter", x="m", y="runtime", style='method',
    height=4.4, aspect=1.61803, hue='dataset', palette='tab20'
) 

plot.set(xscale='log')
plot.set(yscale='log')
plot.set(xlabel=r'size ($m$)')
plot.set(ylabel='time [s]')

plot.savefig(output)