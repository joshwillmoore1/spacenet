.. _documentation:
Documentation
=============

Package structure overview...

.. currentmodule:: netpcf

Utility functions
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   utils.generate_spatial_network
   utils.plot_spatial_network 


Pair correlation functions
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pcf.cross_pair_correlation_function
   pcf.weighted_pair_correlation_function
   pcf.cross_weighted_pair_correlation_function


Spatial network partitioning
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   partition.compact_volume_partition


Example datasets
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:


   datasets.load_dataset


Correlation helper functions
----------------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   pcf.helpers.spatial_bootstrap
   pcf.helpers.polynomial_kernel
   pcf.helpers.integrated_poly_finite_kernel
   pcf.helpers.is_connected_filter
   pcf.helpers.compute_contributions
   pcf.helpers.compute_weighted_contributions
   pcf.helpers.compute_contributions_parallel
   pcf.helpers.compute_weighted_contributions_parallel
   pcf.helpers.batched_dijkstra

