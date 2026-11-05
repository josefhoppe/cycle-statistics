import numpy as np
from math import exp
from tqdm import tqdm

def simulate(J, h, T, steps, rnd=None):
    beta = 1/T
    if rnd is None:
        rnd = np.random.default_rng()
    for _ in tqdm(range(steps)):
        node = rnd.choice(J.shape[0])
        prob = 1 / (1 + exp(-2 * beta * (J @ h)[node]))
        h[node] = rnd.choice([1, -1], p=[prob, 1-prob])
    return h

def energy(J, h):
    return -0.5 * h @ (J @ h) # faster, compare `%timeit (h @ J) @ h` and `%timeit h @ (J @ h)`

def count_frust(J, h):
    J = J.tocoo()
    data = J.data.copy()
    data *= h[J.coords[0]] * h[J.coords[1]]
    return np.count_nonzero(data == -1) / 2