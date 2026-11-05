"""
Graph generators shared by the experiment scripts.
"""

import networkx as nx
from scipy.spatial import Delaunay
import graph_tool.all as gt


def netzschleuder_name(name: str) -> str:
    """
    Converts a (file name friendly) graph name to the graph-tool / Netzschleuder name,
    i.e., '-' -> '_' and '*' -> '/' (e.g., 'sp-high-school*proximity' -> 'sp_high_school/proximity').
    """
    return name.replace('-','_').replace('*', '/')


def is_netzschleuder(name: str) -> bool:
    return netzschleuder_name(name) in gt.collection.ns


def load_netzschleuder(name: str) -> gt.Graph:
    """
    Loads a graph from the Netzschleuder collection as an undirected graph without parallel edges or self-loops,
    restricted to its largest connected component.
    """
    name = netzschleuder_name(name)
    G = gt.collection.ns[name]
    G.set_directed(False)
    gt.remove_parallel_edges(G)
    gt.remove_self_loops(G)
    print(f'{name} gt graph: n={G.num_vertices()}, m={G.num_edges()}')
    if gt.extract_largest_component(G, directed=False).num_vertices() != G.num_vertices():
        G = gt.extract_largest_component(G, directed=False)
        print(f'{name} largest cc: n={G.num_vertices()}, m={G.num_edges()}')
    return gt.Graph(G, prune=True)


def make_graph(kind: str, n: int = 0, p: float = 0, q: float = 0, rnd=None, graph_tool: bool = False) -> tuple[nx.Graph | gt.Graph, int]:
    """
    Generates a graph of the given kind.

    - `n`: number of nodes (side length for grids, ignored for real-world graphs)
    - `p`, `q`: edge probability (within / between blocks for 'sbm'), `m = n * p` for 'barabasialbert'
    - `rnd`: seed or `np.random.Generator` ('delaunay' requires a Generator)
    - `graph_tool`: if True, Netzschleuder graphs are returned as graph-tool graphs instead of networkx graphs

    Returns `(G, n)`, where `n` is the actual number of nodes.
    Nodes of grids have the attribute 'coords' with the original grid coordinates.
    """
    if kind == 'er':
        G = nx.gnp_random_graph(n, p, seed=rnd)
        while not nx.is_connected(G):
            print('WARN: not connected, regenerating')
            G = nx.gnp_random_graph(n, p, seed=rnd)
    elif kind == 'sbm':
        n1, n2 = n//2, n-n//2
        G = nx.stochastic_block_model([n1, n2], [[p,q], [q,p]], seed=rnd)
        while not nx.is_connected(G):
            G = nx.stochastic_block_model([n1, n2], [[p,q], [q,p]], seed=rnd)
    elif kind == 'full':
        G = nx.complete_graph(n)
    elif kind == 'fullbip':
        n1, n2 = n//2, n-n//2
        G = nx.complete_bipartite_graph(n1, n2)
    elif kind == 'grid':
        G = nx.grid_2d_graph(n, n)
        G = nx.convert_node_labels_to_integers(G, label_attribute='coords')
    elif kind == 'grid3d':
        G = nx.grid_graph((n, n, n))
        G = nx.convert_node_labels_to_integers(G, label_attribute='coords')
    elif kind == 'barabasialbert':
        m = int(n * p)
        G = nx.barabasi_albert_graph(n, m, rnd)
    elif kind == 'barbell':
        bell_size = (n-3) // 2
        bar_size = n - 2 * bell_size
        G = nx.barbell_graph(bell_size, bar_size)
        G = nx.convert_node_labels_to_integers(G) # just in case
    elif kind == 'delaunay':
        pts = rnd.uniform(0,1, (n,2))
        tri = Delaunay(pts)
        edges = list(set().union(*[[(simpl[0], simpl[1]), (simpl[1], simpl[2]), (simpl[0], simpl[2])] for simpl in tri.simplices]))
        G = nx.Graph(edges)
        G = nx.convert_node_labels_to_integers(G) # just in case
    elif kind == 'karate':
        G = nx.karate_club_graph()
        G = nx.convert_node_labels_to_integers(G)
    elif is_netzschleuder(kind):
        G = load_netzschleuder(kind)
        if graph_tool:
            return G, G.num_vertices()
        G = nx.from_numpy_array(gt.adjacency(G).todense())
    else:
        raise RuntimeError(f'unknown model: {kind}')
    return G, len(G.nodes)
