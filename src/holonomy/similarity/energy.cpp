#include "holonomy/similarity/energy.h"

#include "holonomy/core/forms.h"

namespace Penner {
namespace Holonomy {

JumpEnergy::JumpEnergy(const Mesh<Scalar>& m)
    : m_opp(m.opp)
{}

Scalar JumpEnergy::energy(const VectorX& metric_coords) const
{
    int num_halfedges = m_opp.size();
    assert(metric_coords.size() == num_halfedges);
    Scalar energy = 0;
    for (int h = 0; h < num_halfedges; ++h) {
        Scalar jump = metric_coords[h] - metric_coords[m_opp[h]];
        energy += jump * jump;
    }

    return 0.25 * energy;
}

VectorX JumpEnergy::gradient(const VectorX& metric_coords) const
{
    int num_halfedges = m_opp.size();
    assert(metric_coords.size() == num_halfedges);
    VectorX gradient(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        Scalar jump = metric_coords[h] - metric_coords[m_opp[h]];
        gradient[h] = jump;
    }

    return gradient;
}

MatrixX JumpEnergy::hessian(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

MatrixX JumpEnergy::hessian_inverse(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

CoordinateEnergy::CoordinateEnergy(
    const DifferentiableConeMetric& target_cone_metric,
    std::vector<int> coordinate_indices)
    : m_metric_target(target_cone_metric.get_reduced_metric_coordinates())
    , m_coordinate_indices(coordinate_indices)
{}

Scalar CoordinateEnergy::energy(const VectorX& metric_coords) const
{
    Scalar energy = 0;
    for (const auto i : m_coordinate_indices) {
        Scalar coord_diff = metric_coords[i] - m_metric_target[i];
        energy += coord_diff * coord_diff;
    }

    return 0.5 * energy;
}

VectorX CoordinateEnergy::gradient(const VectorX& metric_coords) const
{
    int num_coordinates = metric_coords.size();
    VectorX gradient;
    gradient.setZero(num_coordinates);
    for (const auto i : m_coordinate_indices) {
        Scalar coord_diff = metric_coords[i] - m_metric_target[i];
        gradient[i] = coord_diff;
    }

    return gradient;
}

MatrixX CoordinateEnergy::hessian(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

MatrixX CoordinateEnergy::hessian_inverse(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

IntegratedEnergy::IntegratedEnergy(const SimilarityPennerConeMetric& target_similarity_metric)
    : m_target_similarity_metric(target_similarity_metric)
{
    // Integrate the scaling form on the target metric
    std::vector<bool> cut_h, is_cut_h;
    MatrixX one_form_matrix = build_dual_loop_basis_one_form_matrix(
        target_similarity_metric,
        target_similarity_metric.get_homology_basis_loops());
    MatrixX integral_matrix =
        build_one_form_integral_matrix(target_similarity_metric, cut_h, is_cut_h);
    MatrixX integrated_scaling_matrix =
        build_integrated_one_form_scaling_matrix(target_similarity_metric);

    // Get metric expansion matrix
    MatrixX identification, projection;
    std::vector<int> he2e, e2he, proj, embed;
    build_edge_maps(target_similarity_metric, he2e, e2he);
    build_refl_proj(target_similarity_metric, he2e, e2he, proj, embed);
    identification = build_edge_matrix(he2e, e2he);
    projection = build_refl_matrix(proj, embed);

    // Build energy matrices
    m_scaling_matrix = integrated_scaling_matrix * (integral_matrix * one_form_matrix);
    m_expansion_matrix = identification * projection;
    m_metric_target = m_target_similarity_metric.get_metric_coordinates();
    Axx = m_expansion_matrix.transpose() * m_expansion_matrix;
    Axy = m_expansion_matrix.transpose() * m_scaling_matrix;
    Ayx = m_scaling_matrix.transpose() * m_expansion_matrix;
    Ayy = m_scaling_matrix.transpose() * m_scaling_matrix;
    bx = -m_expansion_matrix.transpose() * m_metric_target;
    by = -m_scaling_matrix.transpose() * m_metric_target;
}

Scalar IntegratedEnergy::energy(const VectorX& metric_coords) const
{
    // Separate metric and one-form coordinates
    VectorX reduced_length_coords, harmonic_form_coords;
    m_target_similarity_metric.separate_coordinates(
        metric_coords,
        reduced_length_coords,
        harmonic_form_coords);

    // Compute the integrated metric coordinates
    VectorX integrated_metric_coords =
        m_expansion_matrix * reduced_length_coords + m_scaling_matrix * harmonic_form_coords;
    VectorX difference = integrated_metric_coords - m_metric_target;

    return 0.5 * difference.squaredNorm();
}

VectorX IntegratedEnergy::gradient(const VectorX& metric_coords) const
{
    // Separate metric and one-form coordinates and relabel to x and y
    VectorX reduced_length_coords, harmonic_form_coords;
    m_target_similarity_metric.separate_coordinates(
        metric_coords,
        reduced_length_coords,
        harmonic_form_coords);
    const VectorX& x = reduced_length_coords;
    const VectorX& y = harmonic_form_coords;

    // Compute the gradient in two parts
    VectorX gradient(metric_coords.size());

    // Compute the metric part of the gradient from precomputed matrices
    int num_length_coordinates = reduced_length_coords.size();
    gradient.head(num_length_coordinates) = Axx * x + Axy * y + bx;

    // Compute the one-form part of the gradient from precomputed matrices
    int num_form_coordinates = harmonic_form_coords.size();
    gradient.tail(num_form_coordinates) = Ayx * x + Ayy * y + by;

    return gradient;
}

MatrixX IntegratedEnergy::hessian(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

MatrixX IntegratedEnergy::hessian_inverse(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

// Utility power 2 function
Scalar pow2(Scalar x)
{
    return x * x;
}

TriangleQualityEnergy::TriangleQualityEnergy(const MarkedPennerConeMetric& target_marked_metric)
    : m_target_marked_metric(target_marked_metric)
{ }

Scalar TriangleQualityEnergy::energy(const VectorX& metric_coords) const
{
    if (metric_coords.size() == 0) return 0;
    const auto& n = m_target_marked_metric.n;

    // Build edge maps
    std::vector<int> he2e, e2he, proj, embed;
    build_edge_maps(m_target_marked_metric, he2e, e2he);
    build_refl_proj(m_target_marked_metric, he2e, e2he, proj, embed);

    // Compute embedded edge lengths 
    int num_halfedges = he2e.size();
    VectorX l(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        l[h] = exp(metric_coords[proj[he2e[h]]] / 2.0);
    }

    int num_faces = m_target_marked_metric.n_faces();
    Scalar energy = 0;
    for (int f = 0; f < num_faces; ++f) {
        // Get face halfedges
        int hij = m_target_marked_metric.h[f];
        int hjk = n[hij];
        int hki = n[hjk];

        // Compute ratio of inradius to outradius for face
        Scalar numer = 2*l[hij]*l[hjk]*l[hki];
        Scalar denom = ((-l[hij] + l[hjk] + l[hki])*(l[hij] - l[hjk] + l[hki])*(l[hij] + l[hjk] - l[hki]));
        energy += numer / denom;
    }

    return energy;
}

VectorX TriangleQualityEnergy::gradient(const VectorX& metric_coords) const
{
    const auto& n = m_target_marked_metric.n;

    // Build edge maps
    std::vector<int> he2e, e2he, proj, embed;
    build_edge_maps(m_target_marked_metric, he2e, e2he);
    build_refl_proj(m_target_marked_metric, he2e, e2he, proj, embed);

    // Compute embedded edge lengths 
    int num_halfedges = he2e.size();
    VectorX l(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        l[h] = exp(metric_coords[proj[he2e[h]]] / 2.0);
    }
    
    // Compute gradient by edge iteration
    VectorX gradient;
    int num_edges = m_target_marked_metric.n_edges();
    gradient.setZero(metric_coords.size());
    for (int e = 0; e < num_edges; ++e) {
        int h = m_target_marked_metric.e2he[e];

        // Gradient has components for both halfedges of the edge
        for (int hij : {h, m_target_marked_metric.opp[h]}) {
            // Get other halfedges in face
            int hjk = n[hij];
            int hki = n[hjk];

            // Gradient computed by symbolic computation
            Scalar numer = 2 * l[hjk] * l[hki] *
                           (-l[hij] * (-l[hij] + l[hjk] + l[hki]) * (l[hij] - l[hjk] + l[hki]) -
                            l[hij] * (-l[hij] + l[hjk] + l[hki]) * (l[hij] + l[hjk] - l[hki]) +
                            l[hij] * (l[hij] - l[hjk] + l[hki]) * (l[hij] + l[hjk] - l[hki]) +
                            (-l[hij] + l[hjk] + l[hki]) * (l[hij] - l[hjk] + l[hki]) *
                                (l[hij] + l[hjk] - l[hki]));
            Scalar denom =
                (pow2(-l[hij] + l[hjk] + l[hki]) * pow2(l[hij] - l[hjk] + l[hki]) *
                 pow2(l[hij] + l[hjk] - l[hki]));

            gradient[e] += numer / denom;
        }
    }

    return gradient;
}

MatrixX TriangleQualityEnergy::hessian(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

MatrixX TriangleQualityEnergy::hessian_inverse(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

LogTriangleQualityEnergy::LogTriangleQualityEnergy(const MarkedPennerConeMetric& target_marked_metric)
    : m_target_marked_metric(target_marked_metric)
{ }

Scalar LogTriangleQualityEnergy::energy(const VectorX& metric_coords) const
{
    const auto& n = m_target_marked_metric.n;

    // Build edge maps
    std::vector<int> he2e, e2he, proj, embed;
    build_edge_maps(m_target_marked_metric, he2e, e2he);
    build_refl_proj(m_target_marked_metric, he2e, e2he, proj, embed);

    // Get sum of face energies
    int num_faces = m_target_marked_metric.n_faces();
    Scalar energy = 0;
    for (int f = 0; f < num_faces; ++f) {
        // Get halfedges of face
        int hij = m_target_marked_metric.h[f];
        int hjk = n[hij];
        int hki = n[hjk];

        // Get log penner coordinates of face
        Scalar llij = metric_coords[proj[he2e[hij]]];
        Scalar lljk = metric_coords[proj[he2e[hjk]]];
        Scalar llki = metric_coords[proj[he2e[hki]]];

        // Compute sum of squared coordinate differences
        energy += pow2((-2 * llij) + lljk + llki);
        energy += pow2(llij - (2 * lljk) + llki);
        energy += pow2(llij + lljk - (2 * llki));
    }

    return 0.5 * energy;
}

VectorX LogTriangleQualityEnergy::gradient(const VectorX& metric_coords) const
{
    const auto& n = m_target_marked_metric.n;

    // Build edge maps
    std::vector<int> he2e, e2he, proj, embed;
    build_edge_maps(m_target_marked_metric, he2e, e2he);
    build_refl_proj(m_target_marked_metric, he2e, e2he, proj, embed);
    
    // Compute gradient by edge iteration
    VectorX gradient;
    int num_edges = m_target_marked_metric.n_edges();
    gradient.setZero(metric_coords.size());
    for (int e = 0; e < num_edges; ++e) {
        int h = m_target_marked_metric.e2he[e];

        // Gradient has components for both halfedges of the edge
        for (int hij : {h, m_target_marked_metric.opp[h]}) {
            // Get other halfedges in face
            int hjk = n[hij];
            int hki = n[hjk];

            // Get log edge lengths of face
            Scalar llij = metric_coords[proj[he2e[hij]]];
            Scalar lljk = metric_coords[proj[he2e[hjk]]];
            Scalar llki = metric_coords[proj[he2e[hki]]];

            // Compute gradients for each term containing the edge coordinate
            gradient[e] += -2 * ((-2 * llij) + lljk + llki);
            gradient[e] += (llij - (2 * lljk) + llki);
            gradient[e] += (llij + lljk - (2 * llki));
        }
    }

    return gradient;
}

MatrixX LogTriangleQualityEnergy::hessian(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

MatrixX LogTriangleQualityEnergy::hessian_inverse(const VectorX& metric_coords) const
{
    throw std::runtime_error("No hessian defined");
    return id_matrix(metric_coords.size());
}

} // namespace Holonomy
} // namespace Penner