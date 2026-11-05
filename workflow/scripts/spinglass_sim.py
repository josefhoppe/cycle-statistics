"""
Estimates the frustration per cycle length of a spin glass for its ground state, simulated states, and random states.
"""

import script_utils as su
from spinglass.generator import get_spinglass
from spinglass.utils import simulate, energy, count_frust
import numpy as np
import pandas as pd
from collections import defaultdict
from ast import literal_eval
from tqdm import tqdm
import pickle

import random_ccs as rcc

snakemake = su.fix_smk()

def estimate_frustration(G, edge_idx, edge_data, acc=True, samples=1000):
    """
    edge_idx: matrix of [u, v] -> m
    edge_data: m * N matrix, where m = |E| and N = number of different data points to consider.
    edge_data[:,0] is for theoretical frustration, all others for actual frustration.
    """
    n = len(G.nodes)
    p = len(G.edges) * 2 / (n*n - n)
    ress = rcc.spanning_trees_with_occ_prob(G, p, seed=np.random.default_rng(), trees=samples)

    length_frust = [defaultdict(lambda: 0.0) for _ in range(edge_data.shape[1])]
    length_fr_count = [defaultdict(lambda: 0.0) for _ in range(edge_data.shape[1])]
    length_total = defaultdict(lambda: 0.0)

    for (parent, root, depth, us, vs, lcas, p_cs) in tqdm(ress, total=samples):
        if acc:
            p_cs = np.array(rcc.accurate.occurrence_probability_exact(G, tree=(parent, depth, us, vs)))
        
        frust_depth = -1 * np.ones((depth.shape[0], edge_data.shape[1]))
        
        for i in range(len(G.nodes)):
            su.calc_frust_depth(i, frust_depth, parent, edge_idx, edge_data)

        cycle_frust = frust_depth[us] + frust_depth[vs] - 2 * frust_depth[lcas] + (edge_data[edge_idx[us, vs]])
        # cycle_frust[:,0] %= 2
        lengths = depth[us] + depth[vs] - 2 * depth[lcas] + 1

        for l in np.unique(lengths):
            if np.sum((lengths == l) * (cycle_frust[:,0] % 2)) > 0:
                length_frust[0][l] += np.sum(np.exp2(-p_cs[(lengths == l) * ((cycle_frust[:,0] % 2) > 0)]))
            for i in range(1, edge_data.shape[1]):
                if np.sum((lengths == l) * cycle_frust[:,i]) > 0:
                    length_frust[i][l] += np.sum(np.exp2(-p_cs[(lengths == l) * (cycle_frust[:,i] > 0)]))
            for i in range(edge_data.shape[1]):
                length_fr_count[i][l] += np.sum(np.exp2(-p_cs[lengths == l]) * cycle_frust[lengths == l,i])
            length_total[l] += np.sum(np.exp2(-p_cs[(lengths == l)]))
    return [{k: v / samples for k,v in lf_act.items()} for lf_act in length_frust], [{k: v / samples for k,v in lf_cnt.items()} for lf_cnt in length_fr_count], {k: v / samples for k,v in length_total.items()}

SEED = su.BASE_SEED
rnd = np.random.default_rng(SEED)

kind = snakemake.wildcards['kind']
G, J, *_ = get_spinglass(kind)

with open(snakemake.input[0]) as f:
    opt_h = np.array(literal_eval(f.read()))

# prepare samples 
Ts = [5,2,1,.5,.2,.1]
STEPS = 10000

hs = [np.ones_like(opt_h), opt_h]
labels = ["frust", "opt"]
h_Ts = [.0, .0]

for i in tqdm(range(10), 'simulating spinglass'):
    h = rnd.choice([-1,1], J.shape[0])
    hs.append(h.copy())
    labels.append(f'random-{i}')
    h_Ts.append(np.inf)
    for T in Ts:
        h = simulate(J, h, T, STEPS)
        hs.append(h.copy())
        labels.append(f'sim-{T}-{i}')
        h_Ts.append(T)

# Prepare data for estimation
J = J.tocoo()
edge_index, J_data, coo_x, coo_y = su.edge_index(J)

edge_data = np.array([(J_data * h[coo_x] * h[coo_y]) == -1 for h in hs], dtype=np.int32).T

print('Estimate Frustration')
frust, frust_count, total = estimate_frustration(G, edge_index, edge_data, samples=1000)
cols = {'frust': frust[0], 'total': total}
for lbl, fr_act in zip(labels[1:], frust[1:]):
    cols[lbl] = fr_act
df_count = pd.DataFrame(cols).sort_index()
df_count.index = df_count.index.map(str)
df_count.to_csv(snakemake.output[1])

df_rel = pd.DataFrame()
df_rel['frust'] = df_count['frust'] / df_count.total
for lbl, T, h in zip(labels, h_Ts, hs):
    df_rel[lbl] = df_count[lbl] / df_count.total
    df_rel.loc['E', lbl] = -.5 * h @ (J @ h) # Energy
    df_rel.loc['T', lbl] = T # Temperature
    df_rel.loc['F', lbl] = count_frust(J, h)

df_rel.to_csv(snakemake.output[0])