/*********************************************************************************
*  This file is part of reference implementation of SIGGRAPH Asia 2023 Paper     *
*  `Metric Optimization in Penner Coordinates`           *
*  v1.0                                                                          *
*                                                                                *
*  The MIT License                                                               *
*                                                                                *
*  Permission is hereby granted, free of charge, to any person obtaining a       *
*  copy of this software and associated documentation files (the "Software"),    *
*  to deal in the Software without restriction, including without limitation     *
*  the rights to use, copy, modify, merge, publish, distribute, sublicense,      *
*  and/or sell copies of the Software, and to permit persons to whom the         *
*  Software is furnished to do so, subject to the following conditions:          *
*                                                                                *
*  The above copyright notice and this permission notice shall be included in    *
*  all copies or substantial portions of the Software.                           *
*                                                                                *
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
*  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE  *
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING       *
*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS  *
*  IN THE SOFTWARE.                                                              *
*                                                                                *
*  Author(s):                                                                    *
*  Ryan Capouellez, Denis Zorin,                                                 *
*  Courant Institute of Mathematical Sciences, New York University, USA          *
*                                          *                                     *
*********************************************************************************/
#pragma once

#include "util/common.h"

namespace Penner {

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
    ReductionMaps(const Mesh<Scalar>& m, bool fix_bd_lengths = false);

    // Map between original and doubled mesh
    std::vector<int> proj;
    std::vector<int> embed;

    // Map between edges and halfedges of the doubled mesh
    std::vector<int> he2e;
    std::vector<int> e2he;
    MatrixX projection;

    // List of fixed and free vertices for the constraint
    // TODO: This should be part of the mesh, not the reduction maps
    std::vector<int> fixed_v;
    std::vector<int> free_v;

    // List of fixed and free edge variables
    // TODO: This should be part of the mesh, not the reduction maps
    std::vector<int> fixed_e;
    std::vector<int> free_e;

    // List of boundary and interior edges
    std::vector<int> bd_e;
    std::vector<int> int_e;

    // Element counts
    size_t num_reduced_edges;
    size_t num_edges;
    size_t num_halfedges;
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
/// @param[in] m: mesh to build edge map for
/// @param[out] he2e: map from halfedge indices to edge indices
/// @param[out] e2he: map from edge indices to halfedge indices
void build_edge_maps(const Mesh<Scalar>& m, std::vector<int>& he2e, std::vector<int>& e2he);
void build_edge_maps(const std::vector<int>& opp, std::vector<int>& he2e, std::vector<int>& e2he);


/// Build projection he2e mapping halfedge indices for a halfedge mesh to a list of edge
/// indices.
///
/// @param[in] opp: map of opposite halfedges
/// @return map from halfedge indices to edge indices
std::vector<int> build_edge_map(const std::vector<int>& opp);

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
template <typename Scalar>
void build_refl_proj(
    const Mesh<Scalar>& m,
    const std::vector<int>& he2e,
    const std::vector<int>& e2he,
    std::vector<int>& proj,
    std::vector<int>& embed)
{
    // Resize arrays
    proj.resize(e2he.size());
    embed.clear();
    embed.reserve(e2he.size());

    // Build injective map from edges that are not type 2 to double mesh
    int num_edges = e2he.size();
    for (int e = 0; e < num_edges; ++e) {
        int h0 = m.h0(e2he[e]);
        int h1 = m.h1(e2he[e]);
        if ((m.type[h0] != 2) || (m.type[h1] != 2)) {
            embed.push_back(e);
        }
    }

    // Construct map from double mesh to the embedded mesh
    // Map reflection of edges in the image of the embedding to the original edge
    int num_embedded_edges = embed.size();
    for (int E = 0; E < num_embedded_edges; ++E) {
        int e = embed[E];
        int Re = he2e[m.R[e2he[e]]];
        proj[Re] = E;
    }

    // Map embedded edge to itself. Note that if E is identified with e = embed[E]
    // then this implies proj[E] = E and the map is a projection.
    for (int E = 0; E < num_embedded_edges; ++E) {
        int e = embed[E];
        proj[e] = E;
    }
}

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
void build_refl_he_proj(
    const Mesh<Scalar>& m,
    const std::vector<int>& he2e,
    const std::vector<int>& e2he,
    std::vector<int>& he_proj,
    std::vector<int>& he_embed);

/// Build matrix mapping edge indices to the pair of corresponding halfedge indices
///
/// @param[in] he2e: map from halfedge indices to edge indices
/// @param[in] e2he: map from edge indices to halfedge indices
/// @return matrix mapping edges to halfedges
MatrixX build_edge_matrix(const std::vector<int>& he2e, const std::vector<int>& e2he);

/// Create matrix representing the projection of the double mesh onto the
/// embedding.
///
/// @param[in] proj: map from edges in the double mesh to edges that intersect
/// the original mesh.
/// @param[in] embed: map from edges that intersect the original mesh to the
/// double mesh
/// @return: matrix representing the projection
MatrixX build_refl_matrix(const std::vector<int>& proj, const std::vector<int>& embed);

/// Return true iff the edge is in the original embedded mesh.
///
/// @param[in] proj: map from edges in the double mesh to edges that intersect
/// the original mesh.
/// @param[in] embed: map from edges that intersect the original mesh to the
/// double mesh
/// @param[in] e: edge to check
/// @return true iff e is in the embedded mesh
bool is_embedded_edge(const std::vector<int>& proj, const std::vector<int>& embed, int e);

/// Utility function to reduce a symmetric function defined on the full symmetric
/// mesh to a function defined on the embedding.
///
/// If the function is not symmetric, the value of the embedded mesh edge is used.
///
/// @param[in] m: underlying symmetric mesh for edge function
/// @param[in] symmetric_function: symmetric function on edges of symmetric mesh
/// @param[out] reduced_function: function on embedded edges
void reduce_symmetric_function(
    const std::vector<int>& embed,
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
void expand_reduced_function(
    const std::vector<int>& proj,
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
void restrict_he_func(const std::vector<int>& e2he, const VectorX& f_he, VectorX& f_e);

/// Utility function to expand a function defined on edges to a function defined
/// on halfedges using the identification from build_edge_maps.
///
/// @param[in] he2e: map from halfedge indices to edge indices
/// @param[in] f_e: function on edges of mesh
/// @param[out] f_he: function on halfedges of mesh
void expand_edge_func(const std::vector<int>& he2e, const VectorX& f_e, VectorX& f_he);

/// Ensure mesh m is valid
///
/// TODO: Use for cone metric class
///
/// @param[in] m: underlying mesh
bool is_valid_halfedge(const Mesh<Scalar>& m);

/// Ensure mesh m is a valid double mesh
///
/// @param[in] m: underlying mesh
bool is_valid_symmetry(const Mesh<Scalar>& m);

} // namespace Penner
