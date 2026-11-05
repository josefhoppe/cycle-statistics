import numpy as np

from libc.time cimport clock_t, clock, CLOCKS_PER_SEC

cdef inline double from_binary(long[:] h):
    cdef double res = 0.0
    for bit in h:
        res = res * 2 + (bit == -1)
    return res

cdef inline long count_frust(long[:] data, long[:] coo_x, long[:] coo_y, long[:] h):
    cdef long frust = 0
    cdef long d, h1, h2
    cdef int i
    for i in range(len(data)):
        if (data[i] * h[coo_x[i]] * h[coo_y[i]]) == -1:
            frust += 1
    return frust // 2

cdef inline int count_node_frust(long[:] sorted_data, long[:] sorted_x, long[:] sorted_y, int[:] find_x, long[:] h, long node):
    cdef int frust = 0
    cdef int i = find_x[node]
    while i < len(sorted_x) and sorted_x[i] == node:
        if sorted_data[i] * h[node] * h[sorted_y[i]] == -1:
            frust += 1
        i += 1
    return frust

def spinglass_optimal_solution(J, guess_np, int update_freq=100):
    J = J.tocoo()
    cdef long[:] data = J.data
    cdef long[:] coo_x = J.coords[0]
    cdef long[:] coo_y = J.coords[1]

    sort_idx = np.argsort(J.coords[0])
    sorted_x_np = J.coords[0][sort_idx]
    sorted_y_np = J.coords[1][sort_idx]
    sorted_data_np = J.data[sort_idx]

    cdef long[:] sorted_x = sorted_x_np
    cdef long[:] sorted_y = sorted_y_np
    cdef long[:] sorted_data = sorted_data_np

    cdef int[:] find_x = np.zeros(J.shape[0], dtype=np.int32)

    last = -1
    cdef long i, j
    for i, x in enumerate(sorted_x):
        for j in range(last + 1, x + 1):
            find_x[j] = i
        last = x
    last = x
    print('dd\n')
    cdef long bound = count_frust(data, coo_x, coo_y, guess_np)
    print(f'guess: {bound}', flush=True)
    best_np = guess_np.copy()
    cdef long[:] best = best_np
    h_np = np.zeros_like(guess_np)
    node_frust_np = np.zeros_like(guess_np, dtype=np.int32)
    cdef int[:] node_frust = node_frust_np
    cdef int cur_frust = 0
    cdef int fr = 0
    cdef long[:] h = h_np
    h[0] = 1
    cdef int updates = -1
    i = 1

    cdef clock_t last_update = 0
    cdef double total_configs = 2**(len(guess_np)-1)
    cdef long CLOCKS_PER_UPDATE = CLOCKS_PER_SEC * update_freq

    while i > 0:
        if i == len(h):
            best[:] = h[:]
            bound = cur_frust
            print(f'found better: {bound}', flush=True)
            i -= 1
        elif h[i] == 0:
            h[i] = 1
            fr = count_node_frust(sorted_data, sorted_x, sorted_y, find_x, h, i)
            cur_frust += fr
            node_frust[i] = fr
            if cur_frust < bound:
                i += 1
            else:
                updates = updates + 1 % update_freq
                if (clock() - last_update) > CLOCKS_PER_UPDATE:
                    last_update = clock()
                    print(f"{from_binary(h):.4e} / {total_configs:.4e} ({i})", flush=True)
                # if updates == 0:
                #     progress.update(t, completed=to_binary(h))
        elif h[i] == 1:
            h[i] = -1
            cur_frust -= node_frust[i]
            fr = count_node_frust(sorted_data, sorted_x, sorted_y, find_x, h, i)
            cur_frust += fr
            node_frust[i] = fr
            if cur_frust < bound:
                i += 1
            else:
                updates = updates + 1 % update_freq
                if (clock() - last_update) >= CLOCKS_PER_UPDATE:
                    last_update = clock()
                    print(f"{from_binary(h):.4e} / {total_configs:.4e} ({i})", flush=True)
                # if updates == 0:
                #     progress.update(t, completed=to_binary(h))
        else: # h[i] == -1
            h[i] = 0
            cur_frust -= node_frust[i]
            i -= 1
    return best_np