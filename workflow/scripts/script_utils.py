"""
Helpers shared by the workflow scripts.
"""

# add code to import path
from pathlib import Path
import numpy as np
import scipy.sparse
from snakemake.script import Snakemake

import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

# base seed for single-seed experiments (offset by the optional `seed` wildcard)
BASE_SEED = 381920751
# one seed per run for experiments with a `run` wildcard
RUN_SEEDS = (754037574, 753916888, 878405526, 284910723, 490136116, 643595213, 359647163, 500837505, 411655577, 608270388, 387230776, 285543835, 629764367, 468466287, 981230837, 45531133, 16904131, 286347391, 567786555, 754310168, 257139682, 561448341, 343390849, 695695587, 431254817, 487509702, 469032745, 851658883, 486564388, 238093517, 664152317, 905027104, 771257888, 43201794, 639632436, 309853303, 705418535, 367973531, 240575967, 343427292, 172826508, 283056662, 377739114, 943539321, 965268193, 622578117, 673302424, 743645986, 67291836, 860230239, 773947807, 246583930, 1347798, 799989397, 523406397, 339606692, 898781469, 629118124, 15063413, 263629595, 348249844, 645627820, 105175836, 852879224, 212305831, 289942325, 588572490, 313684670, 861053471, 48173487, 30391100, 144769383, 998927226, 317625100, 262007253, 235800540, 766070342, 169406420, 456407560, 181331735, 261708019, 673706223, 970637432, 67859857, 603203796, 428181538, 561822346, 229580388, 815508389, 416158032, 919782092, 583784921, 445944949, 327458591, 412292816, 918189720, 910809961, 29503305, 423838683, 633065120)
# one seed per run for the runtime experiments
TIME_SEEDS = (757821932, 251989800, 395713524, 661430121, 629645171, 933919130, 893559118, 434866178, 286768719, 525508293)

def fix_smk() -> Snakemake:
    """
    Returns the `snakemake` object that Snakemake injects into the running script.
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return sys.modules['__main__'].snakemake

def calc_frust_depth(node, frust_depth, parent, edge_idx, edge_data):
    """
    Recursively computes the number of negative edges on the path from `node` to the root of the spanning tree given by `parent`.

    `frust_depth` (initialized to -1) is filled in place. It may have a second axis to handle multiple sign assignments at once;
    `edge_data` then has the same second axis.
    """
    if np.any(frust_depth[node] != -1):
        return
    if parent[node] == -1:
        frust_depth[node] = 0
        return
    calc_frust_depth(parent[node], frust_depth, parent, edge_idx, edge_data)
    frust_depth[node] = frust_depth[parent[node]] + edge_data[edge_idx[node, parent[node]]]

def edge_index(J):
    """
    Enumerates the edges (upper triangle) of the symmetric sparse matrix `J`.

    Returns `(edge_idx, J_data, coo_x, coo_y)`, where `edge_idx[u, v] = edge_idx[v, u]` is the index of edge (u, v)
    and `J_data`, `coo_x`, `coo_y` are the value and the endpoints of each edge.
    """
    J = J.tocoo()

    J_data = J.data[J.coords[0] < J.coords[1]]
    coo_x = J.coords[0][J.coords[0] < J.coords[1]]
    coo_y = J.coords[1][J.coords[0] < J.coords[1]]

    edge_enum = np.concatenate((np.arange(len(coo_x)), np.arange(len(coo_x))))
    edge_idx = scipy.sparse.coo_array((edge_enum, (np.concatenate((coo_x, coo_y)), np.concatenate((coo_y, coo_x))))).tocsc()
    return edge_idx, J_data, coo_x, coo_y

def calc_ratio(plus_minus, plus_plus):
    """
    Different from cycleindex package:
    - returns NaN when no result, instead of 0
    - returns fraction of balanced, not unbalanced cycles (to match other methods' implementation)
    - only works on plus_plus / plus_minus from a single subgraph, for sampling as described in the paper
    """
    plus_minus = np.array(plus_minus)
    plus_plus = np.array(plus_plus)
    pos = (plus_plus + plus_minus) / 2
    pos[np.isclose(pos, 0, atol=1e-4)] = 0
    plus_plus[np.isclose(plus_plus, 0, atol=1e-4)] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = pos / plus_plus
    if np.any(np.isinf(ratio)):
        print(plus_minus, plus_plus, pos, ratio)
    return ratio

def giscard_balance_estimate(A, lmax, samples=1000):
    """
    Calculates balance estimate from sampled subgraphs, according to the method described in the paper.

    Note: The implementation `cycleindex.balance_ratio` is different from the paper as it takes the ratio of the sum of subgraph counts, not the average of the ratios of individual subgraph counts.
    Also, it replaces non-occurrences with having one balanced cycle, which is not correct.
    """
    from cycleindex import batch_count_, cycle_count # only required for this baseline
    if lmax < A.shape[0]:
        counts = batch_count_(A, lmax, samples, counts=([],[]))
    else:
        pm, pp = cycle_count(A, lmax)
        counts = ([pm], [pp])
    balances = [calc_ratio(pm, pp) for pm, pp in zip(*counts)]
    return balances