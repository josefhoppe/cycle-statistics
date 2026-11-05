"""
Measures the runtime of the estimators on real-world graphs from Netzschleuder.
"""

import script_utils as su
import graphs
import networkx as nx
import numpy as np
import pandas as pd
import time
from math import ceil, sqrt
from collections import defaultdict
from random_ccs.spanning_trees import normalize_cell
from random_ccs.sampling import spanning_trees_with_occ_prob, get_induced_cycle, estimate_cycle_count
from random_ccs.accurate import occurrence_probability_exact
import graph_tool.all as gt
import scipy

snakemake = su.fix_smk()
samples = int(snakemake.wildcards['samples'])
kind = snakemake.wildcards['kind']
dataset = graphs.netzschleuder_name(kind)
method = snakemake.wildcards['method']

seed = np.random.default_rng()

G = graphs.load_netzschleuder(kind)

# edge_list = [tuple(e) for e in GT.iter_edges()]
# print(f'{dataset} edge list: {len(edge_list)}')

# G = nx.Graph(edge_list)
# print(f'{dataset} NX Graph')
# G = nx.convert_node_labels_to_integers(G)

n = G.num_vertices()
p = G.num_edges() * 2 / (n*n - n)
m = G.num_edges()

print(f'{dataset} (n={n}, m={m})')

data = []

for run in range(10):
    print(f'start {method} {dataset} {run}/10')
    before = time.time()
    rnd = np.random.default_rng(su.TIME_SEEDS[run])
    rnd2 = np.random.default_rng(su.TIME_SEEDS[run])
    found = None
    if method == 'HT-approx':
        ress = spanning_trees_with_occ_prob(G, p, seed=rnd, trees=samples)
        for _ in ress: # generator -> have to empty it
            pass
    elif method == 'HT-exact':
        A = scipy.sparse.csr_array(gt.adjacency(G))
        ress = spanning_trees_with_occ_prob(G, p, seed=rnd, trees=samples)
        for (parent, tree_root, depth, us, vs, lcas, p_cs) in ress:
            cycles = [get_induced_cycle((u,v), parent, depth) for u, v in zip(us, vs)]
            _ = occurrence_probability_exact(A.copy(), cycles)
            cycles = []
    elif method == 'OC-exact':
        _, _, sample_occurred = estimate_cycle_count(G, samples, p)
        A = scipy.sparse.csr_array(gt.adjacency(G)).tocoo()
        lengths_to_est, = np.where(sample_occurred > 100)
        ress = spanning_trees_with_occ_prob(G, p, seed=rnd, trees=samples)
        for s, (parent, tree_root, depth, us, vs, lcas, p_cs) in enumerate(ress):
            length = depth[us] + depth[vs] - 2*depth[lcas] + 1
            for l in lengths_to_est:
                len_count = np.count_nonzero(length == l)
                if len_count > 0:
                    sel = rnd2.choice(len_count)
                    cycle = get_induced_cycle((us[length==l][sel], vs[length==l][sel]), parent, depth)
                    prob = occurrence_probability_exact(A, [cycle])
    elif method == 'IS-exact':
        _, _, sample_occurred = estimate_cycle_count(G, samples, p)
        A = scipy.sparse.csr_array(gt.adjacency(G)).tocoo()
        ress = spanning_trees_with_occ_prob(G, p, seed=rnd, trees=samples)
        for s, (parent, tree_root, depth, us, vs, lcas, p_cs) in enumerate(ress):
            sel = rnd2.choice(len(us))
            cycle = get_induced_cycle((us[sel], vs[sel]), parent, depth)
            prob = occurrence_probability_exact(A, [cycle])
    elif method == 'Giscard-const':
        max_len = 25
        l = 15
        A = np.asarray(gt.adjacency(G).todense())
        balances = su.giscard_balance_estimate(A, max_len, samples)
        found = np.mean([not np.isnan(b[l - 1]) for b in balances])
    elif method == 'Giscard-sqrt':
        max_len = ceil(2 * sqrt(n))
        l = ceil(sqrt(n))
        A = np.asarray(gt.adjacency(G).todense())
        balances = su.giscard_balance_estimate(A, max_len, samples)
        found = np.mean([not np.isnan(b[l - 1]) for b in balances])
    else:
        raise RuntimeError('Unknown method: ' + method)
    runtime = time.time() - before
    data.append({
        'dataset': dataset,
        'method': method, 'n': n, 'm': m, 'p': p, 'samples': samples, 'runtime': runtime,
        'found': found,
    })
    
df_result = pd.DataFrame(data)

df_result.to_csv(snakemake.output[0])
