.. Conditional Density Estimation documentation master file, created by
   sphinx-quickstart on Sun Mar  4 16:33:46 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Conditional Density Estimation Documentation
==========================================================

The ``pytorch-migration`` branch is now the default PyTorch-focused version of this package;
everything in this docs tree reflects the torch-native estimators, the new conda environment script,
and the fact that ``experiments/`` is ignored by default. Legacy TensorFlow helpers and tests live in
the dedicated ``legacy-tf`` branch and won't be merged into the PyTorch release.

Installation & Workflow Notes
###########################################################

* Run ``bash scripts/setup_pytorch_env.sh`` to create the supported Python 3.11/3.10 conda
  environment with CPU PyTorch, pinned NumPy/SciPy, and the pre-installed dependencies.
* Activate the ``cde-pytorch`` env and install the repo in editable mode using
  ``pip install --break-system-packages -e .`` before running the demos/tests from this branch.
* Keep your experiment artifacts under ``experiments/`` so the new ``.gitignore`` continues to ignore them.

Table of contents
###########################################################

.. toctree::
  :maxdepth: 3
  :glob:

  density_estimator/density_estimator
  density_simulation/density_simulation



Indices and tables
#############################################################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
