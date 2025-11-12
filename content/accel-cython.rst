Cython
------

Cython is a superset of Python that additionally supports calling C functions and  declaring C types on variables and class attributes.
It is also a versatile, general purpose compiler.
Since it is supports a superset of Python syntax, nearly all Python code, including 3rd party Python packages are also valid Cython code.
Under Cython, source code gets translated into optimized C/C++ code and compiled as Python extension modules. 

Developers can either:

- prototype and develop Python code in IPython/Jupyter using the ``%%cython`` magic command (**easy**), or
- run the ``cython`` command-line utility to produce a ``.c`` file from a ``.py`` or ``.pyx`` file,
  which in turn needs to be compiled with a C compiler to an ``.so`` library, which can then be directly imported in a Python program (**intermediate**), or
- use setuptools_ or meson_ with meson-python_ to automate the aforementioned build process (**advanced**).

.. _setuptools: https://setuptools.pypa.io/en/latest/userguide/ext_modules.html
.. _meson: https://mesonbuild.com/Cython.html
.. _meson-python: https://mesonbuild.com/meson-python/index.html

Herein, we restrict the discussion to the Jupyter-way of using the ``%%cython`` magic.
A full overview of Cython capabilities refers to the `documentation <https://cython.readthedocs.io/en/latest/>`_.

.. important::

   Due to a `known issue`_ with ``%%cython -a`` in ``jupyter-lab`` we have to use the ``jupyter-nbclassic`` interface
   for this episode.

.. _known issue: https://github.com/cython/cython/issues/7319

Python: Baseline (step 0)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. demo:: Demo: Cython

   Consider a problem to integrate a function:

   .. math:: 
       I = \int^{b}_{a}(x^2 - x)dx

   which can be numerically approximated as the following sum:

   .. math::
      I \approx \delta x \sum_{i=0}^{N-1} (x_i^2 - x_i)
   
   where :math:`a \le x_i \lt b`, and all :math:`x_i` are uniformly spaced apart by :math:`\delta x = (b - a) / N`.

   **Objective**: Repeatedly compute the approximate integral for 1000 different combinations of 
   :math:`a`, :math:`b` and :math:`N`.


Python code is provided below:

.. literalinclude:: example/integrate_python.py 

We generate a dataframe and apply the :meth:`apply_integrate_f` function on its columns, timing the execution:

.. code-block:: ipython

   import pandas as pd

   df = pd.DataFrame(
       {
           "a": np.random.randn(1000),
           "b": np.random.randn(1000),
           "N": np.random.randint(low=100, high=1000, size=1000)
       }
   )          

   %timeit apply_integrate_f(df['a'], df['b'], df['N'])
   # 108 ms ± 5.93 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


Cython: Benchmarking (step 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use Cython, we need to import the Cython extension:

.. code-block:: ipython

   %load_ext cython

As a first cythonization step, we add the cython magic command (``%%cython -a``) on top of Jupyter code cell.
We start by a simply compiling the Python code using Cython without any changes. The code is shown below:

.. literalinclude:: example/cython/integrate_cython_step1.py 


.. figure:: img/cython_annotate.png
   :width: 80%
   :align: left
   :alt: The Cython code above is displayed where various lines of the code are highlighted with yellow background colour of varying intensity.

   Annotated Cython code obtained by running the code above.
   The yellow coloring in the output shows us the amount of pure Python code.

Our task is to remove as much yellow as possible by *static typing*, *i.e.* explicitly declaring arguments, parameters, variables and functions.

We benchmark the Python code just using Cython, and it may give either similar or a slight increase in performance.

.. code-block:: ipython

   %timeit apply_integrate_f_cython_step1(df['a'], df['b'], df['N'])
   # 98.7 ms ± 578 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)


Cython: Adding data type annotation to input variables (step 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we can start adding data type annotation to the input variables as highlightbed in the code example/cython below:

.. tabs::
    .. group-tab:: Pure Python
        .. literalinclude:: example/cython/integrate_cython_step2_purepy.py 
           :emphasize-lines: 7,10,18-20

    .. group-tab:: Cython
        .. literalinclude:: example/cython/integrate_cython_step2.py 
           :emphasize-lines: 6,9,17-19

.. code-block:: ipython

   # this will not work
   #%timeit apply_integrate_f_cython_step2(df['a'], df['b'], df['N'])
   
   # this command works (see the description below)
   %timeit apply_integrate_f_cython_step2(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
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
      np.bool\_;     N/A ;             N/A
      np.int\_;      cnp.int_t;        long
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
      np.float\_;    cnp.float64_t;    double
      np.float32;    cnp.float32_t;    float
      np.float64;    cnp.float64_t;    double
      np.complex\_;  cnp.complex128_t; double complex
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

.. tabs::
    .. group-tab:: Pure Python
        .. literalinclude:: example/cython/integrate_cython_step3_purepy.py 
           :emphasize-lines: 7,11,20

    .. group-tab:: Cython
        .. literalinclude:: example/cython/integrate_cython_step3.py 
           :emphasize-lines: 6,9,16

.. code-block:: ipython

   %timeit apply_integrate_f_cython_step3(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
   # 54.9 ms ± 699 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


Cython: Adding data type annotation to local variables (step 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Last step, we can add type annotation to local variables within functions and the output.

.. tabs::
    .. group-tab:: Pure Python
        .. literalinclude:: example/cython/integrate_cython_step4_purepy.py 
           :emphasize-lines: 7,10,18-20

    .. group-tab:: Cython
        .. literalinclude:: example/cython/integrate_cython_step4.py 
           :emphasize-lines: 6,9,16

.. literalinclude:: example/cython/integrate_cython_step4.py 
   :emphasize-lines: 6,9,10,11,16,20,21

.. code-block:: ipython

   %timeit apply_integrate_f_cython_step4(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
   # 13.8 ms ± 97.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Now it is ~ 10 times faster than the original Python implementation, and all we have done is to add type declarations on the Python code!
We indeed see much less Python interaction in the code from step 1 to step 4.

.. figure:: img/cython_annotate_2.png

.. seealso::

   In order to make Cython code reusable often some packaging is necessary. The compilation to binary extension can either happen during the packaging itself, or
   during installation of a Python package. To learn more about how to package such extensions, read the following guides:

   - *pyOpenSci Python packaging guide*'s page on `build tools <https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-build-tools.html>`__
   - *Python packaging user guide*'s page on `packaging binary extensions <https://packaging.python.org/en/latest/guides/packaging-binary-extensions/>`__