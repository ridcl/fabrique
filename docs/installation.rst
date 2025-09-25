
============
Installation
============

*fabrique* runs on JAX, so you need to install its CPU, GPU or TPU version first.
See `official instructions <https://docs.jax.dev/en/latest/installation.html>`_ for details.


After that, you can install latest released version of *fabrique* using ``pip``:

.. code-block:: bash

   pip install fabrique


Alternatively, you can add development version of *fabrique* as a git submodule and have direct access to code:

.. code-block:: bash

   cd /path/to/your/project

   # add submodule
   mkdir lib
   git submodule add git@github.com:ridlc/fabrique.git lib/fabrique

   # set up PYTHONPATH to include fabrique as a package
   export PYTHONPATH=${PYTHONPATH}:lib/fabrique/src

