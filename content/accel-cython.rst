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
   # 101 ms ± 736 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)


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
   # 102 ms ± 2.06 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


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
   # 34.3 ms ± 537 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)

.. warning::

   You can not pass a Series directly since Cython definition is specific to an array. 
   Instead we should use ``Series.to_numpy()`` to get the underlying NumPy array which works nicely with Cython.

.. note:: 

   Cython uses the normal C syntax for types and provides all standard ones, including pointers.
   Here is a list of some primitive C data types (refer to Cython's documentation on :cython:ref:`types`):

   .. csv-table:: 
      :widths: auto
      :delim: ;

      Cython type identifier; Pure Python dtype;  
      ``char``;               ``cython.char``
      ``int``;                ``cython.int``
      ``unsigned int``;       ``cython.uint``
      ``long``;               ``cython.long``
      ``float``;              ``cython.float``
      ``double``;             ``cython.double``     
      ``double complex``;     ``cython.doublecomplex``
      ``size_t``;             ``cython.size_t``

   Using these data types, we can also annotate arrays (see :cython:ref:`memoryviews`):

   - 1D ``np.float64`` array would be equivalent to ``cython.double[:]``,
   - 2D ``np.float64`` array would be equivalent to ``cython.double[:, :]`` and so on...

.. important::

   to quote the :cython:ref:`Cython documentation <language-basics>`,

      **Typing is not a necessity**

      Providing static typing to parameters and variables is convenience to speed up your code, but it is not a necessity. Optimize where and when needed. In fact, 
      typing can slow down your code in the case where the typing does not allow optimizations but where Cython still needs to check that the type of some object matches the declared type.


Cython: Adding data type annotation to functions (step 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next step, we further add type annotation to functions. There are three ways of declaring functions: 

- ``def`` - Python style:

  - Called by Python or Cython code, and both input/output are Python objects.
  - Declaring argument types and local types (thus return values) can allow Cython to generate optimized code which speeds up the execution.
  - Once types are declared, a ``TypeError`` will be raised if the function is passed with the wrong types.

- ``@cython.cfunc`` or ``cdef`` - C style:

  - :cython:ref:`cdef <cdef>` functions are called from Cython and C, but not from Python code.
  - Cython treats functions as pure C functions, which can take any type of arguments, including non-Python types, `e.g.`, pointers.
  - This usually gives the *best performance*.
  - However, one should really take care of the functions declared by ``cdef`` as these functions are actually writing in C.

- ``@cython.ccall`` or ``cpdef`` - C/Python mixed style:

  - :cython:ref:`cpdef <cpdef>` function combines both ``cdef`` and ``def``.
  - Cython will generate a ``cdef`` function for C types and a ``def`` function for Python types.
  - In terms of performance, ``cpdef`` functions may be *as fast as* those using ``cdef`` and might be as slow as ``def`` declared functions.  

.. tabs::
 .. group-tab:: Pure Python
     .. literalinclude:: example/cython/integrate_cython_step3_purepy.py 
        :emphasize-lines: 7,11,20

 .. group-tab:: Cython
     .. literalinclude:: example/cython/integrate_cython_step3.py 
        :emphasize-lines: 6,9,16

.. code-block:: ipython

   %timeit apply_integrate_f_cython_step3(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
   # 29.2 ms ± 152 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)


Cython: Adding data type annotation to local variables and return (step 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Last step, we can add type annotation to local variables within functions and the return value.

.. tabs::
 .. group-tab:: Pure Python
     .. literalinclude:: example/cython/integrate_cython_step4_purepy.py 
        :emphasize-lines: 7,15-18,32-35

 .. group-tab:: Cython
     .. literalinclude:: example/cython/integrate_cython_step4.py 
        :emphasize-lines: 6,9-11,19,24-25

.. code-block:: ipython

   %timeit apply_integrate_f_cython_step4(df['a'].to_numpy(), df['b'].to_numpy(), df['N'].to_numpy())
   # 471 μs ± 7.38 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


Now it is ~200 times faster than the baseline Python implementation, and all we have done is to add type declarations on the Python code!

.. figure:: img/cython_annotate_2.png
   :width: 80%
   :align: left
   
   We indeed see much less Python interaction in the code from step 1 to step 4.

Other useful features
^^^^^^^^^^^^^^^^^^^^^

There are some useful (and possibly advanced) features which are not covered in this episode. Some of
these features are called :cython:ref:`magic attributes <magic_attributes>`. Here are a few:

- ``cython.cimports`` package for importing and calling C libraries such as :cython:ref:`libc.math`.

.. note::

   Differences between ``import`` (for Python) and ``cimport`` (for Cython) statements

   - ``import`` gives access to Python libraries, functions or attributes
   - ``cimport`` gives access to C libraries, functions or attributes 

   In case of Numpy it is common to use the following, and Cython will internally handle this ambiguity.

   .. tabs::
       .. group-tab:: Pure Python
          .. code-block:: python

             from cython.cimports.libc.stdlib import malloc, free  # Allocate and free memory
             from cython.cimports.libc import math  # For math functions like sin, cos etc.
             from cython.cimports import numpy as np # access to NumPy C API

       .. group-tab:: Cython
          .. code-block:: cython

             from libc.stdlib cimport malloc, free
             from libc.libc cimport math
             cimport numpy as np


- ``cython.nogil``, which can act both as a decorator or context-manager, to manage the GIL (Global Interpreter Lock).
  See :cython:ref:`cython_and_gil`.

- ``@cython.boundscheck(False)`` and ``@cython.wraparound(False)`` decorators to tune indexing of Numpy array.
  See :cython:ref:`numpy_tutorial`.

- ``@cython.cclass`` to declare :cython:ref:`extension-types` which behave similar to Python classes.

In addition to the above Cython can also,

- :cython:ref:`augment with .pxd files <augmenting_pxd>` where the Python code is kept as it is and the ``.pxd`` file
  describes the type annotation. In this form ``.pxd`` is very similar in function to a C/C++ header file
  or ``.pyi`` Python type annnotation file,

- create parallel code using :cython:ref:`parallel-block` and ``prange`` iterator for element-wise parallel operation or reductions
  based on OpenMP threads (see :cython:ref:`parallel-tutorial`).


.. demo::
   
   Here is a code which showcases most of the features above, except the ``@cython.cclass`` feature and the use of ``.pxd`` files.

   .. tabs::
       .. group-tab:: Pure Python
          .. code-block:: python
             :emphasize-lines: 2-3,5-6,12-13

             import cython
             from cython.parallel import parallel, prange
             from cython.cimports.libc.math import sqrt

             @cython.boundscheck(False)
             @cython.wraparound(False)
             def normalize(x: cython.double[:]):
                """Normalize a 1D array by dividing all its elements using its root-mean-square (RMS) value."""
                i: cython.Py_ssize_t
                total: cython.double = 0
                norm: cython.double
                with cython.nogil, parallel():
                   for i in prange(x.shape[0]):
                         total += x[i]*x[i]
                   norm = sqrt(total)
                   for i in prange(x.shape[0]):
                         x[i] /= norm

       .. group-tab:: Cython
          .. code-block:: cython
             :emphasize-lines: 2-3,5-6,12-13

             cimport cython
             from cython.parallel cimport parallel, prange
             from libc.math cimport sqrt

             @cython.boundscheck(False)
             @cython.wraparound(False)
             def normalize(double[:] x):
                 """Normalize a 1D array by dividing all its elements using its root-mean-square (RMS) value."""
                 cdef Py_ssize_t i
                 cdef double total = 0
                 cdef double norm
                 with nogil, parallel():
                     for i in prange(x.shape[0]):
                         total += x[i]*x[i]
                     norm = sqrt(total)
                     for i in prange(x.shape[0]):
                         x[i] /= norm

       .. group-tab:: Numpy
          .. code-block:: python

             def normalize_numpy(x):
                 total = np.dot(x, x)
                 norm = total ** 0.5

                 x[:] /= norm

       .. group-tab:: Naive Python implementation
          .. code-block:: python

             from math import sqrt

             def normalize_naive(x):
                 total = 0
                 for i in range(x.shape[0]):
                     total += x[i] * x[i]

                 norm = sqrt(total)
                 for i in range(x.shape[0]):
                     x[i] /= norm

   .. note::

      If you compare performance of the the Cython code versus the Numpy code, you might observe that it is either on-par, or slightly worse than Numpy.
      This is because Numpy vectorized operations also makes use of OpenMP parallelism and is heavily optimized. Nevertheless, it is orders of magnitude
      better than a naive implementation.

Conclusions
^^^^^^^^^^^

.. keypoints::

   - Cython is a versatile, general purpose compiler for Python code
   - Cython is a great way to write high-performance code in Python where algorithms are not available in scientific libraries like Numpy and Scipy and
     require custom implementation

.. seealso::

   In order to make Cython code reusable often some packaging is necessary. The compilation to binary extension can either happen during the packaging itself, or
   during installation of a Python package. To learn more about how to package such extensions, read the following guides:

   - *pyOpenSci Python packaging guide*'s page on `build tools <https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-build-tools.html>`__
   - *Python packaging user guide*'s page on `packaging binary extensions <https://packaging.python.org/en/latest/guides/packaging-binary-extensions/>`__

