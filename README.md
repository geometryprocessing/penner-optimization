# Penner Coordinate Methods for Robust Parametrization


An implementation of the Penner Coordinate methods for seamless parametrization, with optional feature alignment constraints and metric optimization, described in the following papers:
- [Metric Optimization in Penner Coordinates](https://dl.acm.org/doi/10.1145/3618394).
- [Seamless Parametrization in Penner Coordinates](https://dl.acm.org/doi/10.1145/3658202).
- [Feature-Aligned Parametrization in Penner Coordinates](https://dl.acm.org/doi/10.1145/3731216).


### Overview

This library provides support for generating approximately isometric parameterization of an input triangle mesh with various constraints, including cone angle, seamless, and feature-alignment constraints.

Retriangulation is often necessary to satisfy the constraints supported by this library, and in fact this retriangulation is key to the robustness of Penner coordinate methods. When necessary, the initial mesh is intrinsically refined to produce an output mesh with a compatible parameterization.

The core library provides support for vertex cone angle constraints. The following modules support for additional constraints:
- `Holonomy`: Support for seamless constraints for arbitrary holonomy signatures (see [Seamless Parametrization in Penner Coordinates](https://dl.acm.org/doi/10.1145/3658202))
- `Feature`: Support for feature alignment constraints (see [Feature-Aligned Parametrization in Penner Coordinates](https://dl.acm.org/doi/10.1145/3731216))

The holonomy signature for seamless constraints are generally inferred from a guiding cross-field. A field can be loaded from file, or generated using the following module:
- `Field`: Support for smooth, curvature-aligned cross-field generation

Penner coordinate methods are primarily useful for robustly generating an initial feasible parameterization satisfying constraints. To optimize this parametrization while preserving constraints, use the following modules:
- `Optimization`: Intrinsic metric optimization in terms of Penner coordinates (See [Metric Optimization in Penner Coordinates](https://dl.acm.org/doi/10.1145/3618394))
- `SymmetricDirichlet`: $uv$ coordinate optimization for symmetric Dirichlet with field alignment

Warning: Most of the library is released under a permissive MPL2 license, but `SymmetricDirichlet` and `Field` depend less permissive copyleft licenses. To disable these dependencies, build with the option `-DUSE-COPYLEFT=OFF`. 

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

### Library Structure

- Metric: Underlying methods for differentiable intrinsic metrics, including (a) area and angle values, and (b) length, Penner, shear, and conformal coordinates.
	- Area: Methods to compute the area of triangles from intrinsic lengths
	- Cone Metric: Core representation of a differentiable intrinsic metric on an underlying halfedge mesh
	- Constraint: Methods to compute triangle inequality and differentiable cone angle constraints.
	- Flip Matrix Generator: Data structure to iteratively compute the change of coordinate matrix for intrinsic flips
	- Globals: Global parameters
	- Projection: Methods to project a metric conformally to cone constraints, and to project metric coordinate vectors to the tangent space of the constraint submanifold.
	- Reparametrization: Methods for changing edge and interior barycentric coordinates as determined by hyperbolic edge translations.
	- Shear: Methods to compute the shear coordinates of a metric from Penner coordinates and shear coordinate bases, linearly independent from the conformal scaling space.
- Parametrization: Methods to generate a UV coordinate map from an intrinsic metric
	- Interpolation: Data structures to compute pointwise maps between two intrinsic metrics.
	- Layout: Methods to layout a uv parameterization determined by intrinsic lengths
	- Parametrize: Methods to build a parameterization for a halfedge mesh with the original embedding geometry and a target intrinsic metric
	- Refinement: Methods to refine a triangulation with an accompanying overlay layout sufficiently to ensure the parametrization does not have inverted elements. 
	- Translation: Method to compute hyperbolic translations determining a continuous map between two intrinsic metrics
	- Triangulation: Methods for triangulating self overlapping polygons in the plane
- Holonomy: Library for parametrizing surfaces with full seamless constraints with arbitrary holonomy signatures
	- Core: Basic utilities for tracking dual loops on a surface
		- Boundary Basis: Basis loops for boundary alignment constraints
		- Common: Assorted utilities
		- Dual Lengths: Compute lengths of dual edges on a surface
		- Dual Loop: Data structures for representing dual loops on a surface with efficient queries of edge and path intersections
		- Dual Segment: Minimal representation of dual paths on a surface as list of atomic face crossings
		- Homology Basis: Construct homology basis loops for a surface.
		- Quality: Compute triangle quality measures on a mesh 
		- Viewer: Assorted viewers
	- Holonomy: 
		- Cones: Compute cones, as well as related queries and modification heuristics
		- Constraints: Holonomy angle constraints, including cone and dual loop constraints.
		- Holonomy: Compute holonomy of dual loops on a surface with metric
		- Marked Penner Cone Metric: Differentiable cone metric with full holonomy signature constraints
		- Newton: Modified least squares Newton method for holonomy angle constraints
		- Rotation Form: Method to compute the rotation of a field across edges from a cross field 
	- Similarity: Methods to compute a parametrization satisfying arbitrary holonomy constraints with scale jumps across seams
  	- Conformal: Method to find a conformally equivalent similarity metric satisfying holonomy constraints
  	- Constraint: Similarity structure constraints for holonomy signatures, including vertex angle constraints, dual loop holonomy constraints, and closed form constraints
  	- Layout: Method to parametrize a similarity metric
  	- Similarity: Representation for a mesh with a differentiable intrinsic metric and additional conformal scaling by an integrated harmonic one form, which is sufficient for arbitrary holonomy constraints
- Feature: Library for parametrizing surfaces with both seamless and feature alignment constraints
	- Core
		- Boundary Path: Data structure for tracking the line of symmetry between two boundary vertices
		- Common: Assorted Utilities
		- Component Mesh: Methods and data structures for splitting a disconnected mesh into separate components, as well as corresponding reindexing maps.
		- IO: Methods to write/read edges to/from file.
		- Quads: Methods to compute relevant quad mesh statistics
		- Union Meshes: Methods to combine collections of meshes into a single mesh with multiple components.
		- VF Corners: Methods for representing mesh edges using opposite corners.
		- Viewer: Assorted viewers
	- Dirichlet: Methods for enforcing hard and optimizing soft feature alignment constraints
		- Angle Constraint Relaxer: Method to compute a relaxed angle constraint system that maintains total vertex angles while relaxing feature edge alignment.
		- Cone Perturber: Data structure to modify sector angle constraints for feature alignment
		- Constraints: Methods to compute feature aligned seamless constraints and associated Jacobians
		- Dirichlet Penner Cone Metric: Extension of the differentiable cone metric using Penner coordinates to include full feature aligned seamless constraints
		- Optimization: Two-phase optimization method for alignment of relaxed angle constraint
	- Feature: Methods to build a salient feature graph on a mesh.
		- Error: Methods to compute feature alignment error
		- Features: Method to find feature edges on a surface and hard feature constraint subsets
		- Gluing: Methods to get cut mesh edge and vertex correspondences
	- Surgery: Methods to cut, refine, and stitch meshes and their parametrizations
		- Cut Mesh Layout: Methods to parameterize the uv components of a cut mesh
		- Cut Metric Generator: Data structure to generate fields and differentiable metrics for a mesh cut along feature lines.
		- Refinement: Data structure with limited support to refine the faces and edges of a mesh for feature alignment
		- Stitching: Method to stitch a mesh cut along feature lines and parametrized, possibly with edge refinement, into a single parametrized mesh.
- Optimization
	- Metric Optimization: Methods for optimizing distortion energies for an intrinsic metric with constraints
		- Energies: Differentiable per-face energy functions with gradients for metric optimization
		- Energy Functor: Class for differentiable distortion measures supporting evaluation, gradients, and Hessians
		- Energy Weights: Methods to scale per-element values by element weights
		- Explicit Optimization: Methods to optimize a metric with angle constraints using an explicit basis complimentary to the conformal scaling space.
		- Implicit Optimization:  Method to optimize distortion with constraints using a walk-on-manifold approach in a full space of metric coordinates using projection
		- Nonlinear Optimization: Methods to perform advanced nonlinear optimization, including conjugate gradient and L-BFGS-B
	- Util: Utility for tests and validation
- Field
	- Cross Field: Methods for generating cross fields as represented by four rotationally symmetric tangent vectors.
	-  Facet Field: Methods for generating cross fields using legacy facet field file representations.
	- Field: Methods to generate field directions and cones from an nrosy field.
	- Forms: Methods for creating and manipulating one forms on a surface, represented as anti-symmetric halfedge values.
	- Frame Field: TODO: move assorted functions here to appropriate locations
	- Intrinsic Field: Methods to generate a smooth cross field on a halfedge mesh using only the intrinsic metric, with curvature alignment for extrinsic geoemtry
- Util:
	- Boundary: Methods to find and navigate the boundary elements of a halfedge mesh
	- Common: Type definitions and basic utility functions
	- Embedding: Generate maps between edges and halfedges, as well as edges and unique representative edges in doubled symmetric meshes
	- IO: Methods to read and write vectors, matrices, and meshes to and from file
	- Linear Algebra: Basic linear algebra routines, such as dot products, cross products, and linear solves
	- Map: Methods for composing, reindexing, permuting, and combining maps
	- Spanning Tree: Methods to build spanning trees and cotrees, either dual or primal, on a halfedge mesh
	- Union Find: Union Find data structure
- Vector: Methods to query, convert, and fill vectors of data
- VF Mesh: Methods to process a VF mesh representation