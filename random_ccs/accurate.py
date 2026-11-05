import networkx as nx
import scipy
import numpy as np
from math import log2, log
from numpy.typing import NDArray
from .spanning_trees import get_induced_cycle
from typing import Generator
from sksparse.cholmod import cho_factor

def tree_incl_lapl(A: scipy.sparse.sparray, cycle: tuple[int]) -> scipy.sparse.csc_array:
    """
    Creates the Laplacian whose determinant characterizes spanning trees that include `cycle`.

    For this, all edges in the cycle are contracted (i.e., all nodes merged) into a multigraph.
    Returns the Laplacian of the new multigraph (with fewer nodes than in `A`).
    
    :param A: sparse adjacency matrix. Will be converted to coo format initially.
    :type A: scipy.sparse.sparray
    :param cycle: tuple of nodes in the cycle. 
    :type cycle: tuple[int]
    :return: Laplacian of the contracted Multigraph
    :rtype: scipy.sparse.csc_array
    """
    A = A.tocoo() # coo format is much faster for merging nodes
    data = A.data
    i = A.row
    j = A.col
    new_cols = list(range(A.shape[0])) # which cols will exist afterwards
    collapsed_node = cycle[0] # node to collapse into
    for node in cycle[1:]:
        # simply change coordinates of all entries to the collapsed node
        # duplicate entries will be summed up later
        i[i == node] = collapsed_node
        j[j == node] = collapsed_node
        new_cols.remove(node)
    # set diagonals to zero (only the entry for the collapsed node can be non-zero, but it will exist multiple times in general)
    data[i == j] = 0
    A = scipy.sparse.coo_array((data, (i,j)), shape=tuple(A.shape))
    degs = A.sum(axis=1)

    # construct laplacian coo matrix
    data = np.concatenate((-data, degs))
    i = np.concatenate((i, np.arange(len(degs))))
    j = np.concatenate((j, np.arange(len(degs))))
    L = scipy.sparse.coo_array((data, (i,j)), shape=tuple(A.shape))

    L = L.tocsc()
    L = L[new_cols, :][:, new_cols]
    return L

LN_2 = log(2)

def spanning_tree_count_sparse_log(L: scipy.sparse.sparray) -> float:
    """
    Given the laplacian L, calculates the number of spanning trees on the (multi-) graph represented by L via the determinant.
    
    :param L: Laplacian of a (multi-) graph
    :type L: scipy.sparse.sparray
    :return: log2(# of spanning trees)
    :rtype: float
    """
    factor = cho_factor(L[1:,1:])
    return factor.logdet() / LN_2

def occurrence_probability_exact(G: nx.Graph | scipy.sparse.sparray, cycles: list[tuple] | Generator[tuple, None, None] | None = None, tree: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None) -> list[float]:
    """
    Calculates the occurrence probability of all `cycles` (as log2(p_c)).
    
    :param G: Graph
    :type G: nx.Graph
    :param cycles: list of all cycles, each as tuple
    :type cycles: list[tuple[int]]
    :param tree: Tuple (parent, depth, us, vs), where parent and depth define the tree and u, v define the non-tree edges the cycles of which will be used for the calculation
    :return: list of [log2(p_c)], in the same order as `cycles`
    :rtype: list[float]
    """
    if type(G) is nx.Graph:
        A = nx.adjacency_matrix(G, weight=None).tocoo()
        L = nx.laplacian_matrix(G, weight=None).T
    else:
        A = G.tocoo()
        L = (scipy.sparse.diags(np.squeeze(np.asarray(G.sum(axis=1)))) - A).tocsc()
    total_number_trees = spanning_tree_count_sparse_log(L)
    if cycles is None:
        # yield cycles from tree
        if tree is None:
            raise AssertionError("need either cycles or tree as argument")
        parent, depth, us, vs = tree
        cycles = (get_induced_cycle((u, v), parent, depth) for u,v in zip(us,vs))
    res = []
    for cycle in cycles:
        L = tree_incl_lapl(A.copy(), cycle)
        try:
            log_tree_count = spanning_tree_count_sparse_log(L)
        except AssertionError as ex:
            raise AssertionError(f"Error encountered: '{str(ex)}' for cycle {cycle}")
        log_prob = log_tree_count + log2(len(cycle)) - total_number_trees
        res.append(log_prob)
    return res