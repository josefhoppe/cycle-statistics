"""
Compares the approximated occurrence probabilities of cycles (sampled from spanning trees) to the exact ones.
"""

from collections import defaultdict
import script_utils as su
import graphs
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from random_ccs.spanning_trees import normalize_cell
from random_ccs.sampling import spanning_trees_with_occ_prob, get_induced_cycle
from random_ccs.accurate import occurrence_probability_exact

snakemake = su.fix_smk()

def tau1(d,n,l):
    return 1 / (1 + (d-1)*(n-3)/(n-2) + 2*d/(n-1))

def taui(d,n,l):
    return 1 / (1 + (d-2)*(n-1)/(n-3))

def taui_c(d,n,i):
    """
    taui_corrected
    """
    f = (d-1)*(n-1)/d/(n-2)/(i+1)
    return 1 / (1 + (d-2)*(n-i-2 + 1/f)/(n-3))

def taul2(d,n,l):
    return 1 / (1 + (d-2)*(n-l)*(d-1)*(n-1)/(n-3)/l/d/(n-2))

def taul3(d,n,l):
    """
    tau l-2 actually, but whatever
    """
    f = 1/d + (d-2)*(n-l)/(l-1)/(n-3)/d
    return f / (f + (d-2)*((n-l)/(l-1) + 1)/(n-3))

def approx_probabilities(G: nx.Graph, cycles, q) -> dict[tuple, float]:
    res = {}
    n = len(G.nodes)

    # d_exp = (n - 1) * q
    d_exp = np.mean([G.degree[u] for u in G.nodes])

    for cycle in cycles:
        deg_sum = 0
        deg_prod = 1
        l = len(cycle)
        corr = taul2(d_exp, n, l) * taul3(d_exp, n, l) / taui(d_exp, n, l) / taui(d_exp, n, l)
        for i in range(2, l-2):
            corr *= taui_c(d_exp, n, i) / taui(d_exp, n, i)
        for u in cycle:
            deg_prod *= taui(G.degree[u], n, l)
        for u,v in zip(cycle, cycle[1:] + cycle[:1]):
            deg_sum += tau1(G.degree[u], n, l) / taui(G.degree[u], n, l) / taui(G.degree[v], n, l)
        res[cycle] = deg_sum * deg_prod * corr
    return res

def sample_probs(G, p, samples, seed):
    ress = spanning_trees_with_occ_prob(G, p=p, seed=seed, trees=samples)
    res_dict = {}
    for (parent, tree_root, depth, us, vs, lcas, p_cs) in ress:
        for u, v, p_c in zip(us, vs, p_cs):
            cycle = normalize_cell(get_induced_cycle((u,v), parent, depth))
            res_dict[cycle] = 2**p_c
    return res_dict

SEED = su.BASE_SEED + int(snakemake.wildcards.get('seed', 0))

kind = snakemake.wildcards['kind']
n = int(snakemake.wildcards.get('n', 0))
p = float(snakemake.wildcards.get('p', 0))
q = float(snakemake.wildcards.get('q', 0))

if kind == 'full':
    p = 1

rnd = np.random.default_rng(SEED)
G, n = graphs.make_graph(kind, n, p, q, rnd)

if kind != 'er':
    p = len(G.edges) * 2 / (n*n - n)

degs = list(dict(G.degree).values())

approx = sample_probs(G, p, samples=10, seed=SEED)
probs = {cycle: 2**(occurrence_probability_exact(G, [cycle])[0]) for cycle in tqdm(approx.keys())}
approx_old = approx_probabilities(G, probs.keys(), p)
if kind == 'sbm' or kind == 'fullbip':
    n1 = n//2
    same_cluster = {k : not (min(k) < n1 and max(k)) >= n1 for k in probs.keys()}
else:
    same_cluster = defaultdict(lambda: True)

comparison = np.array([(probs[k], approx[k], approx_old[k], same_cluster[k]) for k in probs.keys()])
df_result = pd.DataFrame(comparison, columns=['correct', 'approx', 'approx_old', 'same_cluster'])
df_result['len'] = [len(c) for c in probs.keys()]
df_result['kind'] = kind
df_result['n'] = n
df_result['p'] = p
df_result['q'] = q
df_result['dia'] = nx.diameter(G)
df_result['deg_std'] = np.std(degs)

df_result.to_csv(snakemake.output[0])
