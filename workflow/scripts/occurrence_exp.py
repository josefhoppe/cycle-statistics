import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import script_utils as su
import graphs
from random_ccs.sampling import estimate_cycle_count, NP_EDGE
import graph_tool.all as gt

snakemake = su.fix_smk()

kind = snakemake.wildcards['kind']
n = int(snakemake.wildcards.get('n',0))
p = float(snakemake.wildcards.get('p', 0))
q = float(snakemake.wildcards.get('q', 0))
samples = int(snakemake.wildcards['samples'])
run = int(snakemake.wildcards['run'])

if kind == 'full':
    p = 1

data_est = []
i = run
rnd = np.random.default_rng(su.RUN_SEEDS[i])
G, n = graphs.make_graph(kind, n, p, q, rnd, graph_tool=True)

# Iteratively remove nodes with degree 1 to compute effective node count
if isinstance(G, nx.Graph):
    n_eff = max([len(comp) for comp in nx.biconnected_components(G)])
else:
    # graph-tool graph
    n_eff = 0
    bc_edge, articulation_node, num = gt.label_biconnected_components(G)
    n_eff = 0
    # can take a long time, starting with the component that has the most edges and only using comps with m' > max(n') avoids most work
    work_list = sorted([(size, comp) for comp, size in enumerate(num)], reverse=True)
    for size, comp in work_list:
        if size > n_eff:
            edge_sel = G.new_edge_property('bool', False)
            edge_sel.set_values(bc_edge.get_array() == comp)
            G_reduced = gt.GraphView(G, efilt=edge_sel)
            n_comp = gt.extract_largest_component(G_reduced).num_vertices()
            n_eff = max(n_eff, n_comp)

estimation_p = (len(G.edges) if isinstance(G, nx.Graph) else G.num_edges()) * 2 / (n*n - n)

_, _, occurred = estimate_cycle_count(G, samples, estimation_p, rnd)

for l, count in enumerate(occurred):
    if count > 0:
        data_est.append({'l': l, 'occurred': count, 'run': i, 'kind': kind, 'gt_kind': graphs.netzschleuder_name(kind), 'n': n, 'n_eff': n_eff, 'p': p, 'q': q, 'samples': samples, 'est_p': estimation_p})

df_est = pd.DataFrame(data_est)
df_est.to_csv(snakemake.output[0])