.. _documentation:
Documentation
=============

somestuff

.. currentmodule:: netpcf

Computing pair correlation functions
------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cross_pair_correlation_function
   weighted_pair_correlation_function
   cross_weighted_pair_correlation_function


.. currentmodule:: netpcf

Useful functions
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   utils.generate_spatial_network
   utils.plot_spatial_network 


Example datasets
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   datasets.load_dataset


Backend helper functions
------------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   helpers.leiden_compact_volume_partition
   helpers.spatial_bootstrap
   helpers.polynomial_kernel
   helpers.integrated_poly_finite_kernel
   helpers.is_connected_filter
   helpers.compute_contributions
   helpers.compute_weighted_contributions
   helpers.compute_contributions_parallel
   helpers.compute_weighted_contributions_parallel
   helpers.batched_dijkstra

