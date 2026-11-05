import networkx as nx
import scipy
import numpy as np
import matplotlib.pyplot as plt
import random_ccs as rcc
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from math import exp
from graphs import make_graph # workflow/scripts must be on the path

def get_spinglass(name: str, p=0.2) -> tuple[nx.Graph, scipy.sparse.sparray, dict]:
    rnd = np.random.default_rng(42)

    if name == "grid":
        G, _ = make_graph('grid3d', 4)
        p /= 4
    elif name == "grid2":
        G, _ = make_graph('grid3d', 4)
    elif name == "grid3":
        G, _ = make_graph('grid', 10)
        p = 0.08
    elif name =='er':
        G, _ = make_graph('er', 60, 0.1, rnd=42)
        p=0.1
    elif name =='er2':
        G, _ = make_graph('er', 60, .1, rnd=42)
        p = 0.3
    elif name =='er3':
        G, _ = make_graph('er', 100, .1, rnd=42)
        p = 0.05
    else:
        raise RuntimeError(f'unknown spinglass type {name}')

    grid_coords = nx.get_node_attributes(G, 'coords')
    if name in ('grid', 'grid2'):
        pos_dict = {i: (a + c/4, b + c/6) for i, (a,b,c) in grid_coords.items()}
    elif name == 'grid3':
        pos_dict = {i: (a, b) for i, (a,b) in grid_coords.items()}
    else:
        pos_dict = nx.spring_layout(G, seed=42)


    A = nx.adjacency_matrix(G).tocoo()
    data = A.data[A.coords[0] < A.coords[1]]
    J_coo_x = A.coords[0][A.coords[0] < A.coords[1]]
    J_coo_y = A.coords[1][A.coords[0] < A.coords[1]]
    J_data = rnd.choice(np.array([-1, 1]), size=len(data), p=[p, 1-p])
    full_data = np.concatenate((J_data, J_data))
    coords = (np.concatenate((J_coo_x, J_coo_y)), np.concatenate((J_coo_y, J_coo_x)))
    J = scipy.sparse.coo_array((full_data, coords)).tocsc()

    return (G, J, pos_dict)