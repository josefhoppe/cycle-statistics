"""
Estimates the number of cycles per length with one of the estimators (HT / OC / IS, exact or approximate occurrence probabilities).
"""

from collections import defaultdict
import script_utils as su
import graphs
import networkx as nx
import numpy as np
import pandas as pd
from random_ccs.sampling import spanning_trees_with_occ_prob, get_induced_cycle
from random_ccs.accurate import occurrence_probability_exact
import graph_tool.all as gt

snakemake = su.fix_smk()

SEED = su.BASE_SEED + int(snakemake.wildcards.get('seed', 0))

kind = snakemake.wildcards['kind']
n = int(snakemake.wildcards.get('n', 0))
p = float(snakemake.wildcards.get('p', 0))
q = float(snakemake.wildcards.get('q', 0))
estimator = snakemake.wildcards['estimator']
metric = snakemake.wildcards.get('metric', 'count')

if kind == 'full':
    p = 1

rnd = np.random.default_rng(SEED)
G, n = graphs.make_graph(kind, n, p, q, rnd, graph_tool=True)

if type(G) is nx.Graph:
    if kind != 'er':
        p = len(G.edges) * 2 / (n*n - n)

    degs = list(dict(G.degree).values())
    A = G
else:
    if kind != 'er':
        p = G.num_edges() * 2 / (n*n - n)

    degs = G.get_total_degrees(G.get_vertices())
    A = gt.adjacency(G)
    # print(A.sum(axis=1))

print("start")

length_results = defaultdict(lambda: np.zeros(1000, dtype=np.float64))
length_counts = defaultdict(lambda: np.zeros(1000, dtype=np.float64))
rnd = np.random.default_rng(SEED)
    
for i, (parent, tree_root, depth, us, vs, lcas, p_cs) in enumerate(spanning_trees_with_occ_prob(G, p, SEED, trees=1000)):
    lengths = depth[us] + depth[vs] - 2 * depth[lcas] + 1
    if 'IS' in estimator: # Importance Sampling, select one cycle and apply
        sel_idx = rnd.integers(len(us))
        l = lengths[sel_idx]
        if 'exact' in estimator:
            #edge: tuple[int, int], parent: np.ndarray, depth: np.ndarray
            len_p_c = occurrence_probability_exact(A, [get_induced_cycle((us[sel_idx], vs[sel_idx]), parent, depth)])[0]
        else:
            len_p_c = p_cs[sel_idx]
        length_results[l][i] = len(us) * np.exp2(-len_p_c)
        for l in np.unique(lengths):
            _ = length_results[l] # still mark as 0
    else:
        for l in np.unique(lengths):
            length_idxs = np.flatnonzero(lengths == l)
            length_counts[l][i] = len(length_idxs)
            if 'HT' in estimator: # Horvitz-Thompson, use all cycles
                if 'exact' in estimator:
                    #edge: tuple[int, int], parent: np.ndarray, depth: np.ndarray
                    len_p_cs = occurrence_probability_exact(A, [get_induced_cycle(edge, parent, depth) for edge in zip(us[length_idxs], vs[length_idxs])])
                else:
                    len_p_cs = p_cs[length_idxs]
                length_results[l][i] = np.sum(1 / np.exp2(len_p_cs)) # TODO: this is only count est.
            else: # One Cycle subsampling of HT (per length)
                sel_idx = rnd.choice(length_idxs, 1)[0]
                if 'exact' in estimator:
                    #edge: tuple[int, int], parent: np.ndarray, depth: np.ndarray
                    len_p_c = occurrence_probability_exact(A, [get_induced_cycle((us[sel_idx], vs[sel_idx]), parent, depth)])[0]
                else:
                    len_p_c = p_cs[sel_idx]
                length_results[l][i] = len(length_idxs) * np.exp2(-len_p_c)

result = []

for l in length_results:
    for i, res in enumerate(length_results[l]):
        result.append((l, i, res, length_counts[l][i]))

df_result = pd.DataFrame(result, columns=['l', 'tree', 'result', 'count'])
df_result['estimator'] = estimator
df_result['metric'] = metric
df_result['kind'] = kind
df_result['n'] = n
df_result['p'] = p
df_result['q'] = q
# df_result['dia'] = nx.diameter(G)
df_result['deg_std'] = np.std(degs)

df_result.to_csv(snakemake.output[0])
