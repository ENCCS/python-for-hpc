%%cython -a

import cython
import numpy as np

@cython.cfunc
def f_cython_step4(x: cython.double) -> cython.double:
    return x ** 2 - x

@cython.cfunc
def integrate_f_cython_step4(
    a: cython.double,
    b: cython.double,
    N: cython.long
) -> cython.double:   
    s: cython.double
    dx: cython.double
    i: cython.long

    s = 0
    dx = (b - a) / N
    
    for i in range(N):
        s += f_cython_step4(a + i * dx)
    return s * dx

@cython.ccall
def apply_integrate_f_cython_step4(
    col_a: cython.double[:],
    col_b: cython.double[:],
    col_N: cython.long[:]
) -> cython.double[:]:
    n: cython.int
    i: cython.int
    res: cython.double[:]
    
    n = len(col_N)
    res = np.empty(n,dtype=np.float64)
    for i in range(n):
        res[i] = integrate_f_cython_step4(col_a[i], col_b[i], col_N[i])
    return res