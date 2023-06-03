#pragma once

#include "common.hh"
#include "conformal_ideal_delaunay/OverlayMesh.hh"

namespace CurvatureMetric {
/// Given logarithmic edge lengths for two adjacent triangles with ccw edges
/// e,a,b and e,c,d, compute the logarithmic edge length of e after flipping
/// edge it to form triangles with edges e,b,c and e,d,a.
///
/// @param[in] lld: log edge length of the center edge
/// @param[in] lla: first log edge length of the first triangle
/// @param[in] llb: second log edge length of the first triangle
/// @param[in] llc: first log edge length of the second triangle
/// @param[in] lld: second log edge length of the second triangle
/// @return log edge length of e after flip
Scalar
log_length_regular(Scalar lle, Scalar lla, Scalar llb, Scalar llc, Scalar lld);

// FIXME Might not be index
/// Compute the Delaunay index of an edge of length ld in a triangle with
/// other edge lengths la and lb.
///
/// @param[in] ld: length of the edge of the triangle to compute the index for
/// @param[in] la: length of the next edge of the triangle
/// @param[in] lb: length of the previous edge of the triangle
/// @return Delaunay index
Scalar
Delaunay_ind_T(Scalar ld, Scalar la, Scalar lb);

// FIXME Might not be index
/// Compute the Delaunay index of halfedge h in the mesh m with metric
/// coordinates per halfedge given by he_metric_coords.
///
/// @param[in] m: mesh to compute Delaunay index for
/// @param[in] he_metric_coords: metric coordinates for m
/// @param[in] h: halfedge to compute index for
/// @return Delaunay index for h
Scalar
metric_halfedge_Delaunay_ind(const Mesh<Scalar>& m,
                             const VectorX& he_metric_coords,
                             int h);

/// Compute the Delaunay index of the edge e with representative halfedge h in
/// the mesh m with metric coordinates given by lambdas_he. The
/// Delaunay index is the sum of both halfedges corresponding to the edge.
///
/// @param[in] m: mesh to compute Delaunay index for
/// @param[in] he_metric_coords: metric coordinates for m
/// @param[in] h: halfedge representing the edge e to compute index for
/// @return Delaunay index for e
Scalar
metric_edge_Delaunay_ind(const Mesh<Scalar>& m,
                         const VectorX& he_metric_coords,
                         int h);

//// Compute the Delaunay index of all edges in the mesh m with metric
/// coordinates given by he_metric_coords.
///
/// @param[in] m: mesh to compute Delaunay indices for
/// @param[in] he_metric_coords: metric coordinates for m
/// @param[out] inds: edge Delaunay indices per halfedge
void
compute_metric_Delaunay_inds(const Mesh<Scalar>& m,
                             const VectorX& he_metric_coords,
                             VectorX& inds);

/// Flip the edge with representative halfedge h in mesh m with metric
/// coordinates he_metric_coords.
///
/// @param[in, out] m: mesh to flip the edge in
/// @param[in, out] he_metric_coords: metric coordinatees for m
/// @param[in] h: halfedge representing the edge e to flip
/// @return: true if and only if the edge is not adjacent to two distinct
/// triangles
bool
flip_metric_ccw(Mesh<Scalar>& m, VectorX& he_metric_coords, int h);

/// Multiply the current Jacobian of the make_delaunay map (represented as a
/// list of lists J_del_lol) by the matrix corresponding to the flip of the
/// single edge with representative h in the mesh m with metric coordinates
/// he_metric_coords.
///
/// @param[in] m: mesh to flip the edge in
/// @param[in] he_metric_coords: metric coordinates for m
/// @param[in] h: halfedge representing the edge e to flip
/// @param[in] he2e: map from halfedges to edges of m
/// @param[in, out] J_del_lol: list of lists matrix for the make_delaunay map to
/// update
/// @return: true if and only if the edge is not adjacent to two distinct
/// triangles
void
update_jacobian_del(const Mesh<Scalar>& m,
                    const VectorX& he_metric_coords,
                    int h,
                    const std::vector<int>& he2e,
                    std::vector<std::map<int, Scalar>>& J_del_lol);

/// Flip the edge with representative halfedge h in mesh m with log lengths
/// lambdas_he and update the Jacobian of the corresponding map from log edge
/// lengths to log edge lengths if need_jacobian is true. Ensure the flip is
/// symmetric if m has a symmetry map (and update the Jacobian for all flips
/// necessary to maintain symmetry), and add potentially non-Delaunay edges
/// after the flip to q.
///
/// @param[in] m: (possibly symmetric) mesh to flip the edge in
/// @param[in] he_metric_coords: metric coordinates for m
/// @param[in] h: halfedge representing the edge e to flip
/// @param[in] tag: technical utility tag used for recursive calls and
/// maintaining symmetry
/// @param[in, out] q: set of potentially non-Delauany edge representative
/// halfedges
/// @param[in, out] flip_seq: sequence of edges flipped
/// @param[in] he2e: map from halfedges to edges of m
/// @param[in, out] J_del_lol: Jacobian of the make_delauany map
/// @param[in] need_jacobian: update Jacobian if true
/// @return true if and only if an edge is flipped
bool
edge_flip(Mesh<Scalar>& m,
          VectorX& he_metric_coords,
          int h,
          int tag,
          std::set<int>& q,
          std::vector<int>& flip_seq,
          const std::vector<int>& he2e,
          std::vector<std::map<int, Scalar>>& J_del_lol,
          bool need_jacobian = false);

/// Make the mesh m with log edge lengths lambdas_full Delaunay using intrinsic
/// flips. If need_jacobian is true, return the Jacobian of the map from
/// lambdas_full to lambdas_full_del, i.e. the map from log edge lengths of the
/// mesh to log edge lengths of the Delaunay mesh. This function performs flips
/// in place.
///
/// @param[in, out] m_del: mesh to make delaunay
/// @param[in] metric_coords: initial metric coordinates
/// @param[out] metric_coords_del: final Delaunay (now log edge length) metric
/// coordinates
/// @param[out] J_del: Jacobian of the make_delauany map
/// @param[out] flip_seq: sequence of halfedges flipped to make the mesh
/// Delaunay
/// @param[in] need_jacobian: create Jacobian if true
void
make_delaunay_with_jacobian_in_place(Mesh<Scalar>& m_del,
                                     const VectorX& metric_coords,
                                     VectorX& metric_coords_del,
                                     MatrixX& J_del,
                                     std::vector<int>& flip_seq,
                                     bool need_jacobian = true);

/// Make the mesh m with metric coordiantes metric_coords Delaunay using
/// intrinsic flips. If need_jacobian is true, return the Jacobian of the map
/// from metric_coords to metric_coords_del, i.e. the map from metric
/// coordinates of the mesh to log edge length metric coordinates of the
/// Delaunay mesh. This function creates a copy of the mesh and metric_coords.
///
/// @param[in] m: (possibly symmetric) mesh to make Delaunay
/// @param[in] metric_coords: metric coordinates for m
/// @param[out] m_del: Delaunay mesh after flips
/// @param[out] lambdas_full_del: log edge lengths for m_del
/// @param[out] J_del: Jacobian of the make_delauany map
/// @param[out] flip_seq: sequence of halfedges flipped to make the mesh
/// Delaunay
/// @param[in] need_jacobian: create Jacobian if true
void
make_delaunay_with_jacobian(const Mesh<Scalar>& m,
                            const VectorX& metric_coords,
                            Mesh<Scalar>& m_del,
                            VectorX& metric_coords_del,
                            MatrixX& J_del,
                            std::vector<int>& flip_seq,
                            bool need_jacobian = true);

/// Follow the flip sequence in flip_seq, where nonnegative indices correspond
/// to Ptolemy flips and negative indices correspond to Euclidean flips of
/// (-h-1), to generate a flipped mesh m_flip with log lengths lambdas_full_flip
/// from m with log lengths lambdas_full. All flips are performed as Ptolemy
/// flips here.
///
/// @param[in] m: (possibly symmetric) mesh to flip
/// @param[in] metric_coords: metric coordinates for m
/// @param[in] flip_seq: sequence of halfedges to flip
/// @param[out] m_flip: mesh after flips
/// @param[out] metric_coords_flip: metric coordinates for m_flip
void
flip_edges(const Mesh<Scalar>& m,
           const VectorX& metric_coords,
           const std::vector<int>& flip_seq,
           Mesh<Scalar>& m_flip,
           VectorX& metric_coords_flip);

#ifdef PYBIND
std::tuple<Mesh<Scalar>,
           VectorX,
           Eigen::SparseMatrix<Scalar, Eigen::RowMajor>,
           std::vector<int>>
make_delaunay_with_jacobian_pybind(const Mesh<Scalar>& C,
                                   const VectorX& lambdas,
                                   bool need_jacobian = true);

std::tuple<OverlayMesh<Scalar>,
           VectorX,
           Eigen::SparseMatrix<Scalar, Eigen::RowMajor>,
           std::vector<int>>
make_delaunay_with_jacobian_overlay(const OverlayMesh<Scalar>& C,
                                    const VectorX& lambdas,
                                    bool need_jacobian = true);

std::tuple<Mesh<Scalar>, VectorX>
flip_edges_pybind(const Mesh<Scalar>& m,
                  const VectorX& metric_coords,
                  const std::vector<int>& flip_seq);
#endif

}
