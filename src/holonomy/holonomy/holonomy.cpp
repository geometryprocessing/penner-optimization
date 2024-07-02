#include "holonomy/holonomy/holonomy.h"

#include "holonomy/core/dual_loop.h"

#include "optimization/core/area.h"

#include <stdexcept>

namespace PennerHolonomy {

Scalar compute_dual_segment_holonomy(
    const Mesh<Scalar>& m,
    const VectorX& he2angle,
    const DualSegment& dual_segment)
{
    assert(is_valid_dual_segment(m, dual_segment));
    int h_start = dual_segment[0];
    int h_end = dual_segment[1];

    // Return negative angle if the segment subtends an angle to the right
    if (h_end == m.n[h_start]) {
        int h_opp = m.n[h_end]; // halfedge opposite subtended angle
        return -he2angle[h_opp];
    }
    // Return positive angle if the segment subtends an angle to the left
    else if (h_start == m.n[h_end]) {
        int h_opp = m.n[h_start]; // halfedge opposite subtended angle
        return he2angle[h_opp];
    }
    // Trivial dual segment
    else if (h_start == h_end) {
        throw std::runtime_error("Cannot compute holonomy for a trivial dual segment");
        return 0.0;
    }
    // Segment in non-triangular face
    else {
        throw std::runtime_error("Cannot compute holonomy for a nontriangular face");
        return 0.0;
    }
}

Scalar compute_dual_loop_holonomy(
    const Mesh<Scalar>& m,
    const VectorX& he2angle,
    const DualLoop& dual_loop)
{
    // Sum up holonomy for all segments of the loop
    Scalar holonomy = 0.0;
    for (const auto& dual_segment : dual_loop) {
        holonomy += compute_dual_segment_holonomy(m, he2angle, dual_segment);
    }

    return holonomy;
}

Scalar compute_dual_segment_rotation(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form,
    const DualSegment& dual_segment,
    bool remove_boundary_rotation=false)
{
    int h_start = dual_segment[0];

    // Skip boundary edges
    if ((remove_boundary_rotation) && (m.type[h_start] != 0) && (m.opp[m.R[h_start]] == h_start))
    {
        return 0.;
    }

    // Return angle of first edge
    return rotation_form(h_start);
}

Scalar compute_dual_loop_rotation(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form,
    const DualLoop& dual_loop)
{
    // Sum up rotation for all segments of the loop
    Scalar rotation = 0.0;
    for (const auto& dual_segment : dual_loop) {
        rotation += compute_dual_segment_rotation(m, rotation_form, dual_segment);
    }

    return rotation;
}

} // namespace PennerHolonomy