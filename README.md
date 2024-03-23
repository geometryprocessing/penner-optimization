# Metric Optimization in Penner Coordinates

<strong>Ryan Capouellez<sup>1</sup>, Denis Zorin<sup>1</sup></strong>

<small><sup>1</sup>New York University</small>

An implementation of [Metric Optimization in Penner Coordinates](https://dl.acm.org/doi/10.1145/3618394).

![Contour pipeline](media/teaser.jpg)

### Overview

This method generates an approximately isometric parameterization of an input `obj` mesh with parametric cone angle constraints. Retriangulation is often necessary to satisfy these constraints, so the initial mesh is intrinsically refined to produce an output mesh with a compatible parameterization.

## Installation

To install this project on a Unix-based system, use the following standard CMake build procedure:

```bash
git clone https://github.com/rjc8237/penner-optimization.git
cd penner-optimization
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 4
```

## Usage

TODO Make binary folder
The core parameterization method is `bin/optimize_metric`. This executable takes the following arguments:

|flag | description|
| --- | --- |
|`mesh` | Mesh filepath|
|`cones` | Target cone filepath|
|`energy` | Energy to optimize|
|`direction` | Direction to use for descent|
|`--num_iter` | Maximum number of iterations|
|`--output` | Output directory|

Supported parameter values for `energy` and `direction` are listed by `bin/optimize_metric --help`.

The input mesh must be a manifold surface with a single connected component. The cone file must be a list of newline separated target vertex cone angles satisfying the discrete Gauss-Bonnet condition. Such a cone prescription can be generated by TODO. The output is a refined mesh with a parameterization and a file of metric coordinate values.

We also provide the executable `bin/optimize_shear` for generating parameterizations using explicit shear coordinate optimization (see paper for details). The executable arguments and output are the same, but the allowed directions are different.

## Figure Reproduction

Scripts to generate the figures of "Metric Optimization in Penner Coordinates" are included in `figures`.

![Some example figures](media/examples.jpg)

TODO
The models (with parameterizations) and cameras used in [Algebraic Smooth Occluding Contours](http://ryanjcapouellez.com/papers/algebraic_smooth_occluding_contours.html) necessary for these scripts can be downloaded [here](http://ryanjcapouellez.com/papers/algebraic-contours-data.zip); they must be copied to `data/meshes` and `data/cameras` respectively.

The figure bash scripts can be run independently or in batch with the command
```
bash fig-all.sh
```

Note that most bash scripts generate an output directory with a JSON file specifying parameters for the parameterization and rendering pipeline python script `scripts/pipeline.py`. Such JSON files can also be used for general batch parameterization and analysis.

### Library

Many parametrization and mapping-related problems in geometry processing can be viewed as metric optimization problems, i.e., computing a metric minimizing a functional and satisfying a set of constraints, such as flatness.

Penner coordinates are global coordinates on the space of metrics on meshes with a fixed vertex set and topology, but varying connectivity, making it homeomorphic to the Euclidean space of dimension equal to the number of edges in the mesh, without any additional constraints imposed.

Crucially for practical applications, a new mesh with standard Euclidean edge lengths corresponding to a metric determined by arbitrary Penner coordinates can be computed by an efficient algorithm. Moreover, the new mesh only differs from the original by a finite sequence of edge flips. Since the resulting edge lengths are analytic functions of the Penner coordinates, standard first-order optimization methods can be applied.

To engender future work in this exciting direction, we provide a library `PennerOptimizationLib` containing:

1. A representation of a mesh with Penner coordinates and cone angle constraints that supports:
   1.  the computation of the corresponding mesh with Euclidean log edge lengths and the Jacobian of the log edge lengths with respect to Penner coordinates
   2.  conformal projection to the cone angle constraints
2.  Various energy functionals and constrained optimization methods for meshes with Penner coordinates
3.  Layout and refinement methods to generate a parameterization for a mesh with arbitrary Penner coordinates such that the parameterization has the intrinsic metric prescribed by the coordinates.

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
