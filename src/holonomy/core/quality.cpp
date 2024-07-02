#include "holonomy/core/quality.h"

namespace PennerHolonomy {

Scalar compute_triangle_quality(Scalar lij, Scalar ljk, Scalar lki)
{
    Scalar numer = 2 * lij * ljk * lki;
    Scalar denom = ((-lij + ljk + lki) * (lij - ljk + lki) * (lij + ljk - lki));
    return (!float_equal(denom, 0)) ? numer / denom : 1e10;
}

// Helper function to compute face quality for a triangle mesh face
Scalar compute_face_quality(
    const DifferentiableConeMetric& cone_metric,
    const VectorX& metric_coords,
    int f)
{
    int hij = cone_metric.h[f];
    int hjk = cone_metric.n[hij];
    int hki = cone_metric.n[hjk];

    // Get edge log lengths and average
    Scalar llij = metric_coords[cone_metric.he2e[hij]];
    Scalar lljk = metric_coords[cone_metric.he2e[hjk]];
    Scalar llki = metric_coords[cone_metric.he2e[hki]];
    Scalar llijk = (llij + lljk + llki) / 3.0;

    // Compute lengths scaled by average (triangle quality is scale invariant)
    Scalar lij = exp((llij - llijk) / 2.0);
    Scalar ljk = exp((lljk - llijk) / 2.0);
    Scalar lki = exp((llki - llijk) / 2.0);

    return compute_triangle_quality(lij, ljk, lki);
}

VectorX compute_mesh_quality(const DifferentiableConeMetric& cone_metric)
{
    // Get metric coordinates
    VectorX metric_coords = cone_metric.get_metric_coordinates();

    // Compute per face quality
    int num_faces = cone_metric.n_faces();
    VectorX mesh_quality(num_faces);
    for (int f = 0; f < num_faces; ++f) {
        mesh_quality[f] = compute_face_quality(cone_metric, metric_coords, f);
    }

    return mesh_quality;
}

Scalar compute_min_angle(const DifferentiableConeMetric& cone_metric)
{
    // Get angles
    VectorX angles, cotangents;
    cone_metric.get_corner_angles(angles, cotangents);

    // Compute per face quality
    return angles.minCoeff();
}

} // namespace PennerHolonomy
