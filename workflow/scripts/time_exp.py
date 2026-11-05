"""
Measures the runtime of the estimators on ER graphs.
"""

import script_utils as su
import networkx as nx
import numpy as np
import pandas as pd
import time
from random_ccs.sampling import spanning_trees_with_occ_prob, get_induced_cycle, estimate_cycle_count, NP_EDGE
from random_ccs.accurate import occurrence_probability_exact

snakemake = su.fix_smk()

n = int(snakemake.wildcards['n'])
p = float(snakemake.wildcards['p'])
N = n * float(snakemake.wildcards['N_factor'])
samples = int(snakemake.wildcards['samples'])
method = snakemake.wildcards['method']

seed = np.random.default_rng()

RUNS = 5 if 'Giscard' in method else 10

data = []

for run in range(RUNS):
    G = nx.gnp_random_graph(n, p, seed=su.TIME_SEEDS[run])
    j = 1
    while not nx.is_connected(G):
        G = nx.gnp_random_graph(n, p, seed=su.TIME_SEEDS[run] + j)
        j += 1
    undersample = 0
    overcorrelate = 0
    results = 0
    before = time.time()
    rnd = np.random.default_rng(su.TIME_SEEDS[run])
    rnd2 = np.random.default_rng(su.TIME_SEEDS[run])
    if method == 'HT-approx':
        ress = spanning_trees_with_occ_prob(G, p, seed=rnd, trees=samples)
        for _ in ress:
            pass
    elif method == 'HT-exact':
        ress = spanning_trees_with_occ_prob(G, p, seed=rnd, trees=samples)
        for (parent, tree_root, depth, us, vs, lcas, p_cs) in ress:
            cycles = [get_induced_cycle((u,v), parent, depth) for u, v in zip(us, vs)]
            c_probs = occurrence_probability_exact(G, cycles)
    elif method == 'OC-exact':
        _, _, sample_occurred = estimate_cycle_count(G, samples, p)
        lengths_to_est, = np.where(sample_occurred > 100)
        ress = spanning_trees_with_occ_prob(G, p, seed=rnd, trees=samples)
        for s, (parent, tree_root, depth, us, vs, lcas, p_cs) in enumerate(ress):
            length = depth[us] + depth[vs] - 2*depth[lcas] + 1
            for l in lengths_to_est:
                len_count = np.count_nonzero(length == l)
                if len_count > 0:
                    sel = rnd2.choice(len_count)
                    cycle = get_induced_cycle((us[length==l][sel], vs[length==l][sel]), parent, depth)
                    prob = occurrence_probability_exact(G, [cycle])
    elif method == 'Giscard-sqrt':
        J = nx.adjacency_matrix(G).todense()
        max_len = int(np.ceil(np.sqrt(n) * 2))
        su.giscard_balance_estimate(J, max_len, samples=samples)
    elif method == 'Giscard-const':
        J = nx.adjacency_matrix(G).todense()
        max_len = 25
        su.giscard_balance_estimate(J, max_len, samples=samples)
    else:
        raise RuntimeError('Unknown method: ' + method)
    runtime = time.time() - before
    data.append({
        'method': method, 'n': n, 'p': p, 'samples': samples, 'runtime': runtime
    })
    print(f"[Time Exp] Done with {method} (n={n},p={p},s={samples}) in {runtime:.2f}s [{run+1}/{RUNS}]")

df_result = pd.DataFrame(data)

df_result.to_csv(snakemake.output[0])
