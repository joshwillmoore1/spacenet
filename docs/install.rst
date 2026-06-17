.. _installation:

Getting Started
===============

This guide will help you install **SpaceNet** and run a minimal example.

Installation
------------

SpaceNet is available on PyPI and can be installed using ``pip`` or with modern Python package managers such as ``uv`` and ``pixi``. Choose the method that best suits your environment:

.. tab-set::

    .. tab-item:: pip
        :sync: pip

        .. code-block:: bash

            pip install spacenet

    .. tab-item:: uv
        :sync: uv

        .. code-block:: bash

            uv add spacenet

    .. tab-item:: pixi
        :sync: pixi

        .. code-block:: bash

            pixi add spacenet

Minimal example
---------------

Below is a minimal example demonstrating how to import SpaceNet and begin working with spatial networks.

.. code-block:: python

   import spacenet as sn

   # Load the spiral dataset and extract the 'x' and 'y' columns as points
   spiral_data = sn.datasets.load_dataset('spiral')
   points = spiral_data[['x', 'y']].to_numpy()

   # generate a spatial network
   G = sn.utils.generate_spatial_network(points,max_edge_distance=50)

   # plot the spatial network
   sn.utils.plot_spatial_network(G)



The output of this snippet will be a plot of the spatial network generated from the spiral dataset. 
The nodes of the network correspond to the points in the dataset, and edges are drawn between nodes that are within a specified distance of each other.

Next steps
----------

.. grid:: 1 1 1 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :img-top: _static/images/examples.svg
        :text-align: center

        **Using SpaceNet**
        ^^^

        Examples of how to use SpaceNet for different applications can be found in the examples section. 
        The examples are a great way to get started with SpaceNet and to learn how to use it for your own projects.

        +++

        .. button-ref:: examples
            :color: secondary
            :click-parent:

            To the examples

    .. grid-item-card::
        :img-top: _static/images/docs.svg
        :text-align: center

        **API reference**
        ^^^

        Explore the full API reference for SpaceNet, including detailed documentation for all functions, classes, and modules. 
        This is the go-to resource for understanding the capabilities of SpaceNet and how to use them effectively.

        +++

        .. button-ref:: documentation
            :color: secondary
            :click-parent:

            To the documentation
