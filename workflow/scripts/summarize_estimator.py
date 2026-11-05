import pandas as pd
import seaborn as sns
import script_utils as su

snakemake = su.fix_smk()

dfs = [pd.read_csv(f, index_col=0) for f in snakemake.input]
dfs_proc = [df.groupby('l').agg({'result':['mean','std'],'estimator':'first', 'metric':'first', 'kind':'first', 'n':'first', 'q':'first', 'p':'first', 'deg_std':'first', 'count': 'mean'}) for df in dfs]

for df in dfs_proc:
    df.reset_index(inplace=True)
    df.columns = ['l', 'mean', 'std', 'estimator', 'metric', 'kind', 'n', 'q', 'p',  'deg_std', 'count']

df_proc = pd.concat(dfs_proc, ignore_index=True)

df_proc['esti'] = df_proc.estimator.apply(lambda s: s.split('-')[0])
df_proc['exactness'] = df_proc.estimator.apply(lambda s: s.split('-')[1])

df_proc['rel_std'] = df_proc['std'] / df_proc['mean']
df_proc['bias'] = 0
for l in df_proc.l.unique():
    mean = df_proc['mean'][(df_proc.l == l) & (df_proc.estimator == 'HT-exact')].item()
    df_proc.loc[df_proc.l == l, 'bias'] = df_proc.loc[df_proc.l == l, 'mean'] / mean

df_proc.to_csv(snakemake.output[0])


