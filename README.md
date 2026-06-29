# SpaceNet

[![Python package](https://github.com/joshwillmoore1/spacenet/actions/workflows/python-package.yml/badge.svg)](https://github.com/joshwillmoore1/spacenet/actions/workflows/python-package.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)]()

<img width="3229" height="375" alt="docs_banner" src="https://github.com/user-attachments/assets/38c2e128-6a9d-48ce-aa5f-2b02b61fae10" />

**SpaceNet** is an open-source Python package for the spatial analysis of **spatially embedded networks**.

The package provides tools for quantifying the geometry, topology, and spatial structure and statistical properties of spatial networks. SpaceNet is designed to support both applied analyses and the development of new methodology in spatial network analysis.


## Installation

Install the latest release from PyPI:

```bash
pip install spacenet
```

Or install the development version directly from GitHub:

```bash
git clone https://github.com/joshwillmoore1/spacenet.git
cd spacenet
pip install -e .
```

---

## Quick Example

```python
import spacenet as sn

# Load the spiral dataset and extract the 'x' and 'y' columns as points
spiral_data = sn.datasets.load_dataset('spiral')
points = spiral_data[['x', 'y']].to_numpy()

# generate a spatial network
G = sn.utils.spatial_network_from_points(points,max_edge_distance=50)

# plot the spatial network
sn.utils.plot_spatial_network(G)
```

See the documentation for additional examples and tutorials at www.spacenet-python.com/latest/.

---

## Citation

If you use SpaceNet in your research, please cite the accompanying publication.

```text
Citation information will be added upon publication.
```

---

## Contributing

Contributions are welcome!

If you encounter a bug, have a feature request, or would like to contribute code or documentation, please open an issue or submit a pull request.

---

## License

SpaceNet is released under the BSD 3-Clause License. See the `LICENSE` file for details.
