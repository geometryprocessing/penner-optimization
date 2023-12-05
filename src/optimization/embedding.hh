#pragma once

#include "common.hh"

namespace CurvatureMetric {

// Methods to expand symmetric functions defined on unique representative edges
// embedded in a symmetric mesh to per edge and per halfedge functions on the
// full symmetric mesh. Conventions:
//     E denotes an edge in the embedded mesh
//     V denotes a vertex in the embedded mesh
//     e denotes an edge in the double mesh
//     h denotes a halfedge in the double mesh
//     v denotes a vertex in the double mesh

/// Data structure to store maps from full doubled mesh halfedge indices to edge indices
/// and to embedded mesh edge indices. Also includes lists of free and fixed vertices and
/// edge indices for variable mesh functions.
struct ReductionMaps
{
  // Map between original and doubled mesh
  std::vector<int> proj;
  std::vector<int> embed;

  // Map between edges and halfedges of the doubled mesh
  std::vector<int> he2e;
  std::vector<int> e2he;
  MatrixX projection;

  // List of fixed and free vertices for the constraint
  std::vector<int> fixed_v;
  std::vector<int> free_v;

  // List of fixed and free edge variables
  std::vector<int> fixed_e;
  std::vector<int> free_e;

  // List of boundary and interior edges
  std::vector<int> bd_e;
  std::vector<int> int_e;

  // Element counts
  size_t num_reduced_edges;
  size_t num_edges;
  size_t num_halfedges;

  ReductionMaps(const Mesh<Scalar>& m, bool fix_bd_lengths = false);
};

/// Build projection he2e mapping halfedge indices for m to a list of edge
/// indices.
///
/// The edges are labeled based on the order of the first occurrence of
/// one of their corresponding halfedges (identified by m.opp) in the list of
/// halfedge indices. Also builds a map e2he from edges to the lower index
/// halfedge.
/// Note: These edge maps remain valid after flips or, more generally,
/// as long as the opp array does not change.
///
/// @param[in] m: mesh to build edge map for. Note that only m.opp is used.
/// @param[out] he2e: map from halfedge indices to edge indices
/// @param[out] e2he: map from edge indices to halfedge indices
void
build_edge_maps(const Mesh<Scalar>& m,
                std::vector<int>& he2e,
                std::vector<int>& e2he);

/// Build projection from edges in a doubled mesh to the edges that intersect
/// the original mesh i.e. edges that are not type 2.
///
/// Also build an embedding of edges that intersect the original mesh to the double
/// mesh. If there are no perpendicular edges that cross the line of symmetry (as is 
/// the case for a doubled mesh before any flips are done), then this defines a
/// retraction to the original surface and an embedding of the original surface
/// in the double mesh in the topological sense.
///
/// @param[in] m: mesh to build projection and embedding for
/// @param[in] he2e: map from halfedge indices to edge indices
/// @param[in] e2he: map from edge indices to halfedge indices
/// @param[out] proj: map from edges in the double mesh to edges that intersect
/// the original mesh.
/// @param[out] embed: map from edges that intersect the original mesh to the
/// double mesh
void
build_refl_proj(const Mesh<Scalar>& m,
                const std::vector<int>& he2e,
                const std::vector<int>& e2he,
                std::vector<int>& proj,
                std::vector<int>& embed);

/// Build projection from halfedges in a doubled mesh to the halfedges that
/// intersect the original mesh i.e. edges that are not type 2.
///
/// Also build an embedding of halfedges that intersect the original mesh to the
/// double mesh.
///
/// Note that this map is different from that induced by the reflection structure
/// on edges as edges on the boundary of the embedded mesh have (1, 2) halfedge pairs.
///
/// @param[in] m: mesh to build projection and embedding for
/// @param[in] he2e: map from halfedge indices to edge indices
/// @param[in] e2he: map from edge indices to halfedge indices
/// @param[out] he_proj: map from halfedges in the double mesh to halfedges that
/// intersect the original mesh.
/// @param[out] he_embed: map from halfedges that intersect the original mesh to
/// the double mesh
void
build_refl_he_proj(const Mesh<Scalar>& m,
                   const std::vector<int>& he2e,
                   const std::vector<int>& e2he,
                   std::vector<int>& he_proj,
                   std::vector<int>& he_embed);

MatrixX
build_edge_matrix(
  const std::vector<int>& he2e,
  const std::vector<int>& e2he
);

/// Create matrix representing the projection of the double mesh onto the
/// embedding.
///
/// @param[in] proj: map from edges in the double mesh to edges that intersect
/// the original mesh.
/// @param[in] embed: map from edges that intersect the original mesh to the
/// double mesh
/// @param[out] projection: matrix representing the projection
void
build_refl_matrix(
  const std::vector<int>& proj,
  const std::vector<int>& embed,
  MatrixX& projection);

/// Return true iff the edge is in the original embedded mesh.
///
/// @param[in] proj: map from edges in the double mesh to edges that intersect
/// the original mesh.
/// @param[in] embed: map from edges that intersect the original mesh to the
/// double mesh
/// @param[in] e: edge to check
/// @return true iff e is in the embedded mesh
bool
is_embedded_edge(
  const std::vector<int>& proj,
  const std::vector<int>& embed,
  int e
);

/// Utility function to reduce a symmetric function defined on the full symmetric
/// mesh to a function defined on the embedding.
///
/// If the function is not symmetric, the value of the embedded mesh edge is used.
///
/// @param[in] m: underlying symmetric mesh for edge function
/// @param[in] symmetric_function: symmetric function on edges of symmetric mesh
/// @param[out] reduced_function: function on embedded edges
void
reduce_symmetric_function(const std::vector<int>& embed,
                          const VectorX& symmetric_function,
                          VectorX& reduced_function);

/// Utility function to expand a function defined on the embedding defined by
/// the reflection projection and embedding to a symmetric function on the full
/// symmetric mesh that agrees with f on the embedded mesh.
///
/// @param[in] m: underlying symmetric mesh for edge function
/// @param[in] reduced_function: function on embedded edges
/// @param[out] symmetric_function: symmetric function on edges of symmetric
/// mesh
void
expand_reduced_function(const std::vector<int>& proj,
                        const VectorX& reduced_function,
                        VectorX& symmetric_function);

/// Utility function to restrict a function defined on halfedges to a function
/// defined on edges using the identification from build_edge_maps.
///
/// If the function is not well defined on halfedge pairs, the lower index halfedge
/// value is used.
///
/// @param[in] e2he: map from edge indices to halfedge indices
/// @param[in] f_he: function on halfedges of mesh
/// @param[out] f_e: function on edges of mesh
void
restrict_he_func(
  const std::vector<int>& e2he,
  const VectorX& f_he,
  VectorX& f_e);

/// Utility function to expand a function defined on edges to a function defined
/// on halfedges using the identification from build_edge_maps.
///
/// @param[in] he2e: map from halfedge indices to edge indices
/// @param[in] f_e: function on edges of mesh
/// @param[out] f_he: function on halfedges of mesh
void
expand_edge_func(
  const std::vector<int>& he2e,
  const VectorX& f_e,
  VectorX& f_he);

/// Ensure mesh m is valid
///
/// @param[in] m: underlying mesh 
bool
is_valid_halfedge(const Mesh<Scalar>& m);

/// Ensure mesh m is a valid double mesh
///
/// @param[in] m: underlying mesh 
bool
is_valid_symmetry(const Mesh<Scalar>& m);

}
