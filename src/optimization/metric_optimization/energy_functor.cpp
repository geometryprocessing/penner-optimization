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
#include "optimization/metric_optimization/energy_functor.h"

#include <igl/cotmatrix_entries.h>
#include <igl/doublearea.h>
#include <igl/edge_lengths.h>
#include "optimization/core/area.h"
#include "optimization/core/constraint.h"
#include "optimization/metric_optimization/energies.h"
#include "optimization/metric_optimization/energy_weights.h"
#include "optimization/core/projection.h"

/// FIXME Do cleaning pass

namespace Penner {
namespace Optimization {

/// Compute the Jacobian matrix of the change of coordinates from log edge
/// lengths to regular edge lengths.
///
/// @param[in] log_length_coords: log lengths for the  original mesh in m
/// @param[out] J_l: Jacobian of the change of coordinates
void length_jacobian(const VectorX& log_length_coords, MatrixX& J_l)
{
    VectorX J_l_vec;
    J_l_vec.setZero(log_length_coords.size());
    for (int e = 0; e < log_length_coords.size(); ++e) {
        J_l_vec[e] = 2.0 / exp(log_length_coords[e]);
    }

    // The Jacobian is a diagonal matrix
    J_l = J_l_vec.asDiagonal();
}

/// Create matrix mapping edge indices to opposite face corners in a mesh
///
/// @param[in] m: (possibly symmetric) mesh
/// @return 3|F|x|H| matrix representing the reindexing.
MatrixX generate_edge_to_face_he_matrix(const Mesh<Scalar>& m)
{
    // Create reindexing matrix
    int num_halfedges = m.n.size();
    int num_faces = m.h.size();
    std::vector<T> tripletList;
    tripletList.reserve(num_halfedges);
    for (int f = 0; f < num_faces; ++f) {
        // Get face halfedges
        int hij = m.h[f];
        int hjk = m.n[hij];
        int hki = m.n[hjk];

        tripletList.push_back(T(3 * f + 0, hij, 1.0));
        tripletList.push_back(T(3 * f + 1, hjk, 1.0));
        tripletList.push_back(T(3 * f + 2, hki, 1.0));
    }
    MatrixX R;
    R.resize(3 * num_faces, num_halfedges);
    R.reserve(tripletList.size());
    R.setFromTriplets(tripletList.begin(), tripletList.end());

    return R;
}

/// Compute the per vertex function given by the maximum of the per halfedge
/// function g on the mesh m among incoming halfedges. Assumes that g is
/// nonnegative.
///
/// @param[in] m: (possibly symmetric) mesh
/// @param[in] g: per halfedge function
/// @return: per vertex function
VectorX halfedge_function_to_vertices(const Mesh<Scalar>& m, const VectorX& g)
{
    // Initializes vector G to zero vector of length |V|
    VectorX G;
    G.setZero(m.n_ind_vertices());

    // Iterate over halfedges to compute the maximum
    int num_halfedges = m.n.size();
    for (int h = 0; h < num_halfedges; ++h) {
        G[m.v_rep[m.to[h]]] = max(G[m.v_rep[m.to[h]]], g[h]);
    }

    return G;
}

LogLengthEnergy::LogLengthEnergy(const DifferentiableConeMetric& target_cone_metric, int order)
    : m_metric_target(target_cone_metric.get_reduced_metric_coordinates())
    , m_order(order)
{}

Scalar LogLengthEnergy::energy(const VectorX& metric_coords) const
{
    assert(metric_coords.size() == m_metric_target.size());
    if (m_order == 2) {
        VectorX difference = metric_coords - m_metric_target;
        return 0.5 * difference.squaredNorm();
    } else {
        int num_edges = metric_coords.size();
        VectorX difference = metric_coords - m_metric_target;
        Scalar energy = 0.0;
        for (int E = 0; E < num_edges; ++E) {
            // Iterate to compute the pth power of the edge difference absolute value
            Scalar term = 1.0;
            Scalar edge_difference = difference[E];
            for (int i = 0; i < m_order; ++i) {
                term *= abs(edge_difference);
            }

            // Add the term (weighted by 1/p)
            energy += term / static_cast<Scalar>(m_order);
        }
        return energy;
    }
}

VectorX LogLengthEnergy::gradient(const VectorX& metric_coords) const
{
    assert(metric_coords.size() == m_metric_target.size());
    if (m_order == 2) {
        VectorX difference = metric_coords - m_metric_target;
        return difference;
    } else {
        int num_edges = metric_coords.size();
        VectorX difference = metric_coords - m_metric_target;
        VectorX p_norm_gradient(num_edges);
        for (int E = 0; E < num_edges; ++E) {
            // Iterate to compute the gradient of the edge difference as the signed
            // p-1th power of the absolute value
            Scalar edge_difference = difference[E];
            p_norm_gradient[E] = edge_difference;
            for (int i = 2; i < m_order; ++i) {
                p_norm_gradient[E] *= abs(edge_difference);
            }
        }
        return p_norm_gradient;
    }
}

MatrixX LogLengthEnergy::hessian(const VectorX& metric_coords) const
{
    if (m_order != 2) {
        throw std::runtime_error("No hessian defined");
    }

    return id_matrix(metric_coords.size());
}

MatrixX LogLengthEnergy::hessian_inverse(const VectorX& metric_coords) const
{
    if (m_order != 2) {
        throw std::runtime_error("No hessian defined");
    }

    return id_matrix(metric_coords.size());
}

QuadraticSymmetricDirichletEnergy::QuadraticSymmetricDirichletEnergy(
    const DifferentiableConeMetric& target_cone_metric,
    const DiscreteMetric& discrete_metric)
    : m_metric_target(target_cone_metric.get_reduced_metric_coordinates())
{
    // Compute quadratic energy matrix
    MatrixX T = discrete_metric.get_transition_jacobian();
    MatrixX R = generate_edge_to_face_he_matrix(discrete_metric) * T;
    MatrixX M = surface_hencky_strain_energy(discrete_metric);

    // Compute quadratic energy matrix_inverse
    MatrixX inverse_M = surface_hencky_strain_energy_inverse(discrete_metric);

    // Conjugate halfedge energy with edge to halfedge maps
    m_quadratic_energy_matrix = R.transpose() * (M * R);
    m_quadratic_energy_matrix_inverse = R.transpose() * (inverse_M * R);
}

Scalar QuadraticSymmetricDirichletEnergy::energy(const VectorX& metric_coords) const
{
    assert(metric_coords.size() == m_metric_target.size());
    VectorX difference = metric_coords - m_metric_target;
    return 0.5 * (difference.dot((m_quadratic_energy_matrix * difference)));
}

VectorX QuadraticSymmetricDirichletEnergy::gradient(const VectorX& metric_coords) const
{
    assert(metric_coords.size() == m_metric_target.size());
    VectorX difference = metric_coords - m_metric_target;
    return m_quadratic_energy_matrix * difference;
}

MatrixX QuadraticSymmetricDirichletEnergy::hessian(const VectorX&) const
{
    return m_quadratic_energy_matrix;
}

MatrixX QuadraticSymmetricDirichletEnergy::hessian_inverse(const VectorX&) const
{
    return m_quadratic_energy_matrix_inverse;
}

LogScaleEnergy::LogScaleEnergy(const DifferentiableConeMetric& target_cone_metric)
    : m_target_cone_metric(target_cone_metric.clone_cone_metric())
    , m_expansion_matrix(target_cone_metric.get_expansion_matrix())
{}

Scalar LogScaleEnergy::energy(const VectorX& metric_coords) const
{
    // Can convert halfedge gradient g to reduced gradient with T^t g = (df dT)^t
    VectorX u = best_fit_conformal(*m_target_cone_metric, m_expansion_matrix * metric_coords);
    return 0.5 * (u.dot(u));
}

VectorX LogScaleEnergy::gradient(const VectorX& metric_coords) const
{
    return scale_distortion_direction(*m_target_cone_metric, metric_coords);
}

SymmetricDirichletEnergy::SymmetricDirichletEnergy(
    const DifferentiableConeMetric& target_cone_metric,
    const DiscreteMetric& discrete_metric)
    : m_target_cone_metric(target_cone_metric.clone_cone_metric())
    , m_face_area_weights(compute_face_area_weights(discrete_metric))
{}

Scalar SymmetricDirichletEnergy::energy(const VectorX& metric_coords) const
{
    assert(metric_coords.size() == m_target_cone_metric->n_halfedges());
    // Get per face symmetric dirichlet energy
    VectorX f2energy;
    MatrixX J_f2energy;
    bool need_jacobian = false;
    symmetric_dirichlet_energy(
        *m_target_cone_metric,
        metric_coords,
        f2energy,
        J_f2energy,
        need_jacobian);

    // Compute face integrated symmetric dirichlet energy
    return f2energy.dot(m_face_area_weights);
}

VectorX SymmetricDirichletEnergy::gradient(const VectorX& metric_coords) const
{
    assert(metric_coords.size() == m_target_cone_metric->n_halfedges());
    // Get per face symmetric dirichlet energy
    VectorX f2energy;
    MatrixX J_f2energy;
    bool need_jacobian = true;
    symmetric_dirichlet_energy(
        *m_target_cone_metric,
        metric_coords,
        f2energy,
        J_f2energy,
        need_jacobian);

    // Compute face integrated symmetric dirichlet energy Jacobian
    return (m_face_area_weights.transpose() * J_f2energy);
}

RegularizedQuadraticEnergy::RegularizedQuadraticEnergy(
    const DifferentiableConeMetric& target_cone_metric,
    const DiscreteMetric& discrete_metric,
    double weight)
    : m_metric_target(target_cone_metric.get_reduced_metric_coordinates())
{
    // Compute quadratic energy matrix
    MatrixX R = generate_edge_to_face_he_matrix(discrete_metric);
    MatrixX M = surface_hencky_strain_energy(discrete_metric);

    // Conjugate halfedge energy with edge to halfedge maps
    MatrixX Q = R.transpose() * M * R;
    MatrixX I = id_matrix(m_metric_target.size());

    // Compute mesh area
    VectorX face_area_weights = compute_face_area_weights(discrete_metric);
    Scalar mesh_area = face_area_weights.sum();

    // Compute regularized matrix
    m_quadratic_energy_matrix = Q + (weight * mesh_area) * I;
}

Scalar RegularizedQuadraticEnergy::energy(const VectorX& metric_coords) const
{
    assert(metric_coords.size() == m_metric_target.size());
    VectorX difference = metric_coords - m_metric_target;
    return 0.5 * (difference.dot((m_quadratic_energy_matrix * difference)));
}

VectorX RegularizedQuadraticEnergy::gradient(const VectorX& metric_coords) const
{
    assert(metric_coords.size() == m_metric_target.size());
    VectorX difference = metric_coords - m_metric_target;
    return m_quadratic_energy_matrix * difference;
}

#ifdef PYBIND

MatrixX length_jacobian_pybind(const VectorX& lambdas_full)
{
    MatrixX J_l;
    length_jacobian(lambdas_full, J_l);

    return J_l;
}

#endif

} // namespace Optimization
} // namespace Penner