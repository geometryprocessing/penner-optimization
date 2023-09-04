#pragma once

#include "common.hh"

namespace CurvatureMetric {

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

#endif

}
