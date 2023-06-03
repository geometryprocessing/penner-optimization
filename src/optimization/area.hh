#pragma once

#include "common.hh"

namespace CurvatureMetric {

/// Compute the squared area of the triangle with edge lengths li, lj, lk using
/// the numerically stable version of Heron's formula.
///
/// Heron's formula gives the formula s(s-a)(s-b)(s-c), where s is is the
/// semi-perimeter of the triangle. If the edge lengths do not satisfy the
/// triangle inequality this is no longer the squared area, but the formula is
/// still polynomial in the lengths.
///
/// @param[in] li: length of the first side of a triangle
/// @param[in] lj: length of the second side of a triangle
/// @param[in] lk: length of the third side of a triangle
/// @return: formal squared area of the triangle
Scalar
area_squared(Scalar li, Scalar lj, Scalar lk);

/// Compute the derivative of the squared area of the triangle with given edge
/// lengths using the numerically stable version of Heron's formula with respect
/// to the first length.
///
/// Note that if the edge lengths do not satisfy the triangle inequality this is
/// no longer the squared area, but the formula is still polynomial in the
/// lengths.
///
/// @param[in] variable_length: length of the triangle to compute the derivative
/// with respect to
/// @param[in] lj: length of the second side of a triangle
/// @param[in] lk: length of the third side of a triangle
/// @return: the derivative of the squared area of the triangle
Scalar
area_squared_derivative(Scalar variable_length, Scalar lj, Scalar lk);

/// Compute the squared area of the triangle containing each halfedge for the
/// mesh m with log edge lengths log_length_coords.
///
/// @param[in] m: (possibly symmetric) Delaunay mesh
/// @param[in] log_length_coords: log edge lengths for m
/// @param[out] he2areasq: map from halfedges to the square of the area of the
/// face containing it
void
areas_squared_from_log_lengths(const Mesh<Scalar>& m,
                               const VectorX& log_length_coords,
                               VectorX& he2areasq);

/// Compute the area of the triangle containing each halfedge for the
/// mesh m with log edge lengths log_length_coords.
///
/// @param[in] m: (possibly symmetric) Delaunay mesh
/// @param[in] log_length_coords: log edge lengths for m
/// @param[out] he2area: map from halfedges to the area of the face containing it
void
areas_from_log_lengths(const Mesh<Scalar>& m,
                       const VectorX& log_length_coords,
                       VectorX& he2area);

/// Compute the derivatives of the squared area of the triangle containing each
/// halfedge for the mesh m with log edge lengths log_length_coords with respect
/// to the log edge lengths.
///
/// @param[in] m: (possibly symmetric) Delaunay mesh
/// @param[in] log_length_coords: log edge lengths for m
/// @param[out] he2areasqderiv: map from halfedges to the derivative of the
/// square of the area of the face containing it
void
area_squared_derivatives_from_log_lengths(const Mesh<Scalar>& m,
                                          const VectorX& log_length_coords,
                                          VectorX& he2areasqderiv);

#ifdef PYBIND
VectorX // he2areasq
areas_squared_from_log_lengths_pybind(const Mesh<Scalar>& m,
                                      const VectorX& log_length_coords);
#endif

}
