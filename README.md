# Metric Optimization in Penner Coordinates

<strong>Ryan Capouellez<sup>1</sup>, Denis Zorin<sup>1</sup></strong>

<small><sup>1</sup>New York University</small>

An implementation of [Metric Optimization in Penner Coordinates](https://dl.acm.org/doi/10.1145/3618394).

![Parameterization with interpolation](media/teaser.jpg)

### Overview

This method generates an approximately isometric parameterization of an input `obj` mesh with parametric cone angle constraints. Retriangulation is often necessary to satisfy these constraints, so the initial mesh is intrinsically refined to produce an output mesh with a compatible parameterization.

## Installation

To install this project on a Unix-based system, use the following standard CMake build procedure:

```bash
git clone https://github.com/geometryprocessing/penner-optimization.git
cd penner-optimization
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 4
```

## Usage

The core parameterization method is `bin/optimize_metric`. This executable takes the following arguments:

|flag | description|
| --- | --- |
|`mesh` | Mesh filepath|
|`cones` | Target cone filepath|
|`energy` | Energy to optimize|
|`direction` | Direction to use for descent|
|`--num_iter` | Maximum number of iterations|
|`--output` | Output directory|
|`--show_parameterization` | Open polyscope viewer for the output|

Supported parameter values for `energy` and `direction` are listed by `bin/optimize_metric --help`.

The input mesh must be a manifold surface with a single connected component. The cone file must be a list of newline separated target vertex cone angles satisfying the discrete Gauss-Bonnet condition. Methods to generate such a cone prescription will be provided soon. The output is a refined mesh with a parameterization and a file of metric coordinate values.

We also provide the executable `bin/optimize_shear` for generating parameterizations using explicit shear coordinate optimization (see paper for details). The executable arguments and output are the same, but the allowed directions are different. This method is in practice much slower to converge than the metric optimization method, but it does have standard formal convergence guarantees.

## Figure Reproduction

Scripts to generate the figures of "Metric Optimization in Penner Coordinates" are included in `figures`.

![Some example figures](media/examples.jpg)

The models (with cone angles) and cameras used in [Metric Optimization in Penner Coordinates](https://dl.acm.org/doi/10.1145/3618394) necessary for these scripts can be downloaded [here](https://cims.nyu.edu/gcl/papers/2021-Conformal.zip);  `MPZ_closed`, `MPZ_open`, `MPZ_cut`, and `cameras` must be copied to `data/closed-Myles`, `data/open-Myles`, `data/cut-Myles`, and `data/cameras` respectively.

A Conda environment must be activated (before compiling the code) with
```
conda env create -f environment.yml
conda activate penner-optimization
```
The figure bash scripts can then be run independently or in batch with
```
bash fig-all.sh
```

Note that most bash scripts generate an output directory with a JSON file specifying parameters for the parameterization and rendering pipeline python script `scripts/_pipeline.py`. Such JSON files can also be used for general batch parameterization and analysis.

### Library

Many parametrization and mapping-related problems in geometry processing can be viewed as metric optimization problems, i.e., computing a metric minimizing a functional and satisfying a set of constraints, such as flatness.

Penner coordinates are global coordinates on the space of metrics on meshes with a fixed vertex set and topology, but varying connectivity, making it homeomorphic to the Euclidean space of dimension equal to the number of edges in the mesh, without any additional constraints imposed.

Crucially for practical applications, there is an efficient algorithm to convert these abstract Penner coordinates to a more familiar representation of a metric: a mesh with standard Euclidean logarithmic edge lengths. Moreover, this mesh only differs from the input mesh by a finite sequence of edge flips, and the resulting edge lengths are analytic functions of the Penner coordinates. We can thus formulate many constraints, such as vertex cone angles, on the mesh with Euclidean edge lengths as differentiable functions of Penner coordinates.

To engender future work in this exciting direction, we provide a library `PennerOptimizationLib` containing:

1. A representation of a mesh with Penner coordinates and cone angle constraints that supports:
   1.  the computation of the corresponding mesh with Euclidean logarithmic edge lengths and the Jacobian of the logarithmic edge lengths with respect to Penner coordinates
   2.  conformal projection to the cone angle constraints
2.  Various energy functionals and constrained optimization methods for meshes with Penner coordinates
3.  Layout and refinement methods to generate a parameterization for a mesh from arbitrary Penner coordinates, i.e., the parameterization is an embedding in the uv plane of the metric defined by the Penner coordinate.

## Citation

```
@article{capouellez:2023:penner,
author = {Capouellez, Ryan and Zorin, Denis},
title = {Metric Optimization in Penner Coordinates},
year = {2023},
issue_date = {December 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {42},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/3618394},
doi = {10.1145/3618394},
journal = {ACM Trans. Graph.},
month = {dec},
articleno = {234},
numpages = {19},
keywords = {cone metrics, conformal mapping, discrete metrics, intrinsic triangulation, parametrization, penner coordinates}
}
```
