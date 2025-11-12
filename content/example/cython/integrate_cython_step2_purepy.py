%%cython -a

import cython
import numpy as np


def f_cython_step2(x: cython.double):
    return x ** 2 - x

def integrate_f_cython_step2(a: cython.double, b: cython.double, N: cython.long):   
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_cython_step2(a + i * dx)
    return s * dx

def apply_integrate_f_cython_step2(
    col_a: cython.double[:],
    col_b: cython.double[:],
    col_N: cython.long[:],
):  
    n = len(col_N)
    res = np.empty(n,dtype=np.float64)
    for i in range(n):
        res[i] = integrate_f_cython_step2(col_a[i], col_b[i], col_N[i])
    return res