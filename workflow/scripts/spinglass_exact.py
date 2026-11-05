"""
Computes the ground state (optimal spin assignment) of a spin glass.
"""

import script_utils as su
from spinglass.generator import get_spinglass
from spinglass.utils import simulate, energy, count_frust
from spinglass.solver import spinglass_optimal_solution
import numpy as np

snakemake = su.fix_smk()

SEED = su.BASE_SEED
rnd = np.random.default_rng(SEED)

kind = snakemake.wildcards['kind']
G, J, *_ = get_spinglass(kind)

h = rnd.choice([-1,1], J.shape[0])
print(f'random: frust {count_frust(J, h)}; E {energy(J, h)}')

print('simulating:')
for T in [2,1,.5,.2,.1]:
    simulate(J, h, T, 10000)
    print(f'temp {T}: frust {count_frust(J, h)}; E {energy(J, h)}')

print('solving')
opt = spinglass_optimal_solution(J, h, update_freq=600) # one update every 10 minutes is enough

print(f'optimal: frust {count_frust(J, opt)}; E {energy(J, opt)}')

with open(snakemake.output[0], "w") as f:
    f.write(str([o.item() for o in opt]))
