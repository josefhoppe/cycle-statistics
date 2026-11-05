"""
Estimates the balance (fraction of balanced cycles) per length of a spin glass with one of the estimators (HT / OC / IS, exact or approximate occurrence probabilities, or Giscard).
"""

from math import ceil, sqrt
import random
from collections import defaultdict
import script_utils as su
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from random_ccs.sampling import spanning_trees_with_occ_prob, get_induced_cycle
from random_ccs.accurate import occurrence_probability_exact
from spinglass.generator import get_spinglass
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

snakemake = su.fix_smk()

SEED = su.BASE_SEED + int(snakemake.wildcards.get('seed', 0))

kind = snakemake.wildcards['kind']
estimator = snakemake.wildcards['estimator']
metric = snakemake.wildcards.get('metric', 'count')

G, J, *_ = get_spinglass(kind)

def one_sample(G, J, estimator, seed, run_info):
    n = len(G.nodes)
    p = len(G.edges) * 2 / (n*n - n)
    A = G


    # Prepare data for estimation
    edge_idx, J_data, _, _ = su.edge_index(J)
    edge_data = np.array(J_data == -1, dtype=np.int32)

    length_balanced = defaultdict(lambda: np.zeros(1000, dtype=np.float64))
    length_total = defaultdict(lambda: np.zeros(1000, dtype=np.float64))
    length_counts = defaultdict(lambda: np.zeros(1000, dtype=np.float64))
    rnd = np.random.default_rng(seed)

    if estimator == 'Giscard':
        random.seed(seed)
        #def balance_ratio(G, length, exact=False, n_samples=100000, accuracy=None, sampling_func=nrsampling, parallel=True):
        max_len = ceil(sqrt(n) * 2)
        balances = np.array(su.giscard_balance_estimate(J.todense(), max_len, 100)).T
        if run_info == 4:
            print(balances[11], np.nanmean(balances[11]))
        balance = {}
        for l in range(3, max_len + 1):
            balance[l] = np.nanmean(balances[l - 1]) if not np.all(np.isnan(balances[l - 1])) else np.nan
        return run_info, defaultdict(lambda: 0), defaultdict(lambda: 0), balance
        
    for i, (parent, tree_root, depth, us, vs, lcas, p_cs) in enumerate(spanning_trees_with_occ_prob(G, p, seed, trees=100)):
        lengths = depth[us] + depth[vs] - 2 * depth[lcas] + 1
            
        frust_depth = -1 * np.ones(depth.shape[0], depth.dtype)
        
        for j in range(len(G.nodes)):
            su.calc_frust_depth(j, frust_depth, parent, edge_idx, edge_data)

        cycle_frust = frust_depth[us] + frust_depth[vs] - 2 * frust_depth[lcas] + (edge_data[edge_idx[us, vs]])

        if 'IS' in estimator: # Importance Sampling, select one cycle and apply
            sel_idx = rnd.integers(len(us))
            l = lengths[sel_idx]
            frust = cycle_frust[sel_idx] % 2
            if 'exact' in estimator:
                #edge: tuple[int, int], parent: np.ndarray, depth: np.ndarray
                len_p_c = occurrence_probability_exact(A, [get_induced_cycle((us[sel_idx], vs[sel_idx]), parent, depth)])[0]
            else:
                len_p_c = p_cs[sel_idx]
            length_total[l][i] = len(us) * np.exp2(-len_p_c)
            if frust % 2 == 0:
                length_balanced[l][i] = len(us) * np.exp2(-len_p_c)
            else:
                length_balanced[l][i] = 0
            for l in np.unique(lengths):
                _ = length_total[l] # still mark as 0
        else:
            for l in np.unique(lengths):
                length_idxs = np.flatnonzero(lengths == l)
                length_counts[l][i] = len(length_idxs)
                if 'HT' in estimator: # Horvitz-Thompson, use all cycles
                    if 'exact' in estimator:
                        #edge: tuple[int, int], parent: np.ndarray, depth: np.ndarray
                        len_p_cs = occurrence_probability_exact(A, [get_induced_cycle(edge, parent, depth) for edge in zip(us[length_idxs], vs[length_idxs])])
                        len_p_cs = np.array(len_p_cs)
                    else:
                        len_p_cs = p_cs[length_idxs]
                    len_frusts = cycle_frust[length_idxs] % 2
                    length_total[l][i] = np.sum(1 / np.exp2(len_p_cs))
                    length_balanced[l][i] = np.sum(1 / np.exp2(len_p_cs[len_frusts == 0]))
                else: # One Cycle subsampling of HT (per length)
                    sel_idx = rnd.choice(length_idxs, 1)[0]
                    frust = cycle_frust[sel_idx] % 2
                    if 'exact' in estimator:
                        #edge: tuple[int, int], parent: np.ndarray, depth: np.ndarray
                        len_p_c = occurrence_probability_exact(A, [get_induced_cycle((us[sel_idx], vs[sel_idx]), parent, depth)])[0]
                    else:
                        len_p_c = p_cs[sel_idx]
                    length_total[l][i] = len(length_idxs) * np.exp2(-len_p_c)
                    if frust == 0:
                        length_balanced[l][i] = len(length_idxs) * np.exp2(-len_p_c)
                    else:
                        length_balanced[l][i] = 0
    total_count = {}
    balanced_count = {}
    balance = {}
    for l in length_total:
        total = length_total[l].sum()
        total_count[l] = total
        balanced = length_balanced[l].sum()
        balanced_count[l] = balanced
        with np.errstate(invalid='ignore'):
            balance[l] = balanced / total
    return run_info, total_count, balanced_count, balance

runs = 100

with joblib_progress(f"Sampling estimates for {estimator} on {kind}", total=runs):
    samples = Parallel(n_jobs=snakemake.threads)(delayed(one_sample)(G, J, estimator, SEED + i, i) for i in range(runs))

result = []

for (i, length_total, length_balanced, balance) in samples:
    for l in balance:
        result.append((l, i, balance[l], length_total[l], length_balanced[l]))

df_result = pd.DataFrame(result, columns=['l', 'sample', 'balance', 'total_count', 'balanced_count'])
df_result['estimator'] = estimator
df_result['metric'] = metric
df_result['kind'] = kind
df_result['n'] = len(G.nodes)
J_upper = scipy.sparse.triu(J, k=1)
df_result['n_pos'] = int((J_upper == 1).sum())
df_result['n_neg'] = int((J_upper == -1).sum())

df_result.to_csv(snakemake.output[0])
