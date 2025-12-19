%%cython -a

import numpy as np

def f_cython_step1(x):
    return x * (x - 1)

def integrate_f_cython_step1(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_cython_step1(a + i * dx)
    return s * dx

def apply_integrate_f_cython_step1(col_a, col_b, col_N):
    n = len(col_N)
    res = np.empty(n,dtype=np.float64)
    for i in range(n):
        res[i] = integrate_f_cython_step1(col_a[i], col_b[i], col_N[i])
    return res
