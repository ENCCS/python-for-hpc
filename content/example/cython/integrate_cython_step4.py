%%cython -a

import numpy as np


cdef double f_cython_step4(double x):
    return x ** 2 - x

cdef double integrate_f_cython_step4(double a, double b, long N):   
    cdef double s, dx
    cdef long i
    
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_cython_step4(a + i * dx)
    return s * dx

cpdef double[:] apply_integrate_f_cython_step4(
    double[:] col_a,
    double[:] col_b,
    long[:] col_N
):
    cdef long n,i
    cdef double[:] res
    
    n = len(col_N)
    res = np.empty(n,dtype=np.float64)
    for i in range(n):
        res[i] = integrate_f_cython_step4(col_a[i], col_b[i], col_N[i])
    return res
