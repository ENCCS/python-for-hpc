Cython
------

Cython is a superset of Python that additionally supports calling C functions and  declaring C types on variables and class attributes.
Under Cython, source code gets translated into optimized C/C++ code and compiled as Python extension modules. 

Developers can run the ``cython`` command-line utility to produce a ``.c`` file from a ``.py`` file which needs to be compiled with a C compiler to an ``.so`` library which can then be directly imported in a Python program.
There is, however, also an easy way to use Cython directly from Jupyter notebooks through the ``%%cython`` magic command. Herein, we restrict the discussion to the Jupyter-way.
A full overview of Cython capabilities refers to the `documentation <https://cython.readthedocs.io/en/latest/>`_.

.. demo:: Demo: Cython

   Consider a problem to integrate a function:

   .. math:: 
       \int^{b}_{a}(x^2-x)dx


Cython: Benchmarking (step 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use Cython, we need to import the Cython extension:

.. code-block:: ipython

   %load_ext cython

As a first cythonization step, we add the cython magic command (``%%cython -a``) on top of Jupyter code cell.
We start by a simply compiling the Python code using Cython without any changes. The code is shown below:

.. literalinclude:: example/cython/integrate_cython.py 

The yellow coloring in the output shows us the amount of pure Python code:

.. figure:: img/cython_annotate.png

Our task is to remove as much yellow as possible by *static typing*, *i.e.* explicitly declaring arguments, parameters, variables and functions.

We benchmark the Python code just using Cython, and it gives us about 10%-20% increase in performance. 

.. code-block:: ipython

   %timeit apply_integrate_f_cython(df['a'], df['b'], df['N'])
   # 141 ms ± 3.07 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


Cython: Adding data type annotation to input variables (step 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we can start adding data type annotation to the input variables as highlightbed in the code example/cython below:

.. literalinclude:: example/cython/integrate_cython_dtype0.py 
   :emphasize-lines: 6,9,16

.. code-block:: ipython

   # this will not work
   #%timeit apply_integrate_f_cython_dtype0(df['a'], df['b'], df['N'])
   
   # this command works (see the description below)
   %timeit apply_integrate_f_cython_dtype0(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
   # 64.1 ms ± 0.50 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

.. warning::

   You can not pass a Series directly since Cython definition is specific to an array. 
   Instead we should use ``Series.to_numpy()`` to get the underlying NumPy array which works nicely with Cython.

.. note:: 

   Cython uses the normal C syntax for types and provides all standard ones, including pointers.
   Here is a list of a few example/cythons:

   .. csv-table:: 
      :widths: auto
      :delim: ;

      NumPy dtype;  Cython type identifier; C type identifier
      import numpy as np; cimport numpy as cnp ;
      np.bool\_;      N/A ;             N/A
      np.int\_;       cnp.int_t;        long
      np.intc;       N/A ;             int       
      np.intp;       cnp.intp_t;       ssize_t
      np.int8;       cnp.int8_t;       signed char
      np.int16;      cnp.int16_t;      signed short
      np.int32;      cnp.int32_t;      signed int
      np.int64;      cnp.int64_t;      signed long long
      np.uint8;      cnp.uint8_t;      unsigned char
      np.uint16;     cnp.uint16_t;     unsigned short
      np.uint32;     cnp.uint32_t;     unsigned int
      np.uint64;     cnp.uint64_t;     unsigned long
      np.float\_;     cnp.float64_t;    double
      np.float32;    cnp.float32_t;    float
      np.float64;    cnp.float64_t;    double
      np.complex\_;   cnp.complex128_t; double complex
      np.complex64;  cnp.complex64_t;  float complex
      np.complex128; cnp.complex128_t; double complex

.. note::

   Differences between ``import`` (for Python) and ``cimport`` (for Cython) statements

   - ``import`` gives access to Python libraries, functions or attributes
   - ``cimport`` gives access to C libraries, functions or attributes 
   - it is common to use the following, and Cython will internally handle this ambiguity

   .. code-block:: ipython

      import numpy as np  # access to NumPy Python functions
      cimport numpy as np # access to NumPy C API


Cython: Adding data type annotation to functions (step 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next step, we further add type annotation to functions. There are three ways of declaring functions: 

- ``def`` - Python style:

   - Called by Python or Cython code, and both input/output are Python objects.
   - Declaring argument types and local types (thus return values) can allow Cython to generate optimized code which speeds up the execution.
   - Once types are declared, a ``TypeError`` will be raised if the function is passed with the wrong types.

- ``cdef`` - C style:

   - Called from Cython and C, but not from Python code.
   - Cython treats functions as pure C functions, which can take any type of arguments, including non-Python types, `e.g.`, pointers.
   - This usually gives the best performance. 
   - However, one should really take care of the functions declared by ``cdef`` as these functions are actually writing in C.

- ``cpdef`` - C/Python mixed style:

   - ``cpdef`` function combines both ``cdef`` and ``def``.
   - Cython will generate a ``cdef`` function for C types and a ``def`` function for Python types.
   - In terms of performance, ``cpdef`` functions may be as fast as those using ``cdef`` and might be as slow as ``def`` declared functions.  

.. literalinclude:: example/cython/integrate_cython_dtype1.py 
   :emphasize-lines: 6,9,16

.. code-block:: ipython

   %timeit apply_integrate_f_cython_dtype1(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
   # 54.9 ms ± 699 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


Cython: Adding data type annotation to local variables (step 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Last step, we can add type annotation to local variables within functions and the output.

.. literalinclude:: example/cython/integrate_cython_dtype2.py 
   :emphasize-lines: 6,9,10,11,16,20,21

.. code-block:: ipython

   %timeit apply_integrate_f_cython_dtype2(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
   # 13.8 ms ± 97.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Now it is ~ 10 times faster than the original Python implementation, and all we have done is to add type declarations on the Python code!
We indeed see much less Python interaction in the code from step 1 to step 4.

.. figure:: img/cython_annotate_2.png
