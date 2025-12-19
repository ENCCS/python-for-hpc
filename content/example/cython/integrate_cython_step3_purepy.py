%%cython -a

import cython
import numpy as np


@cython.cfunc
def f_cython_step3(x: cython.double):
    return x ** 2 - x

@cython.cfunc
def integrate_f_cython_step3(a: float, b: float, N: int):   
    s = 0
    dx = (b - a) / N

    for i in range(N):
        s += f_cython_step3(a + i * dx)
    return s * dx

@cython.ccall
def apply_integrate_f_cython_step3(
    col_a: cython.double[:],
    col_b: cython.double[:],
    col_N: cython.long[:]
):  
    n = len(col_N)
    res = np.empty(n,dtype=np.float64)
    for i in range(n):
        res[i] = integrate_f_cython_step3(col_a[i], col_b[i], col_N[i])
    return res