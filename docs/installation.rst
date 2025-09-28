
============
Installation
============

*fabrique* runs on JAX, so you’ll need to install its CPU, GPU, or TPU version first.
See the `official instructions <https://docs.jax.dev/en/latest/installation.html>`_ for details.

After that, you can install the latest released version of *fabrique* using ``pip``::

.. code-block:: bash

   pip install fabrique


Alternatively, you can add the development version of *fabrique* as a Git submodule to have direct access to the source code::

.. code-block:: bash

   cd /path/to/your/project

   # add submodule
   mkdir lib
   git submodule add git@github.com:ridlc/fabrique.git lib/fabrique

   # set up PYTHONPATH to include fabrique as a package
   export PYTHONPATH=${PYTHONPATH}:lib/fabrique/src

