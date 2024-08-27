#include "holonomy/core/dual_segment.h"
#include <algorithm>
#include <stdexcept>

#ifdef ENABLE_VISUALIZATION
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Holonomy {

bool is_valid_dual_segment(const Mesh<Scalar>& m, const DualSegment& dual_segment)
{
    // Just check segment halfedges belong to the same face
    return (m.f[dual_segment[0]] == m.f[dual_segment[1]]);
}

bool is_valid_dual_path(const Mesh<Scalar>& m, const std::vector<DualSegment>& dual_path)
{
    int path_length = dual_path.size();
    for (int i = 0; i < path_length; ++i) {
        // Check segment i is valid
        if (!is_valid_dual_segment(m, dual_path[i])) {
            spdlog::error("Invalid dual segment ({}, {})", dual_path[i][0], dual_path[i][1]);
            return false;
        }

        // Check sequential dual segments are adjacent in the mesh
        if ((i + 1 < path_length) && (dual_path[i][1] != m.opp[dual_path[i + 1][0]])) {
            spdlog::error("Dual segments {} and {} are not adjacent", i, i + 1);
            return false;
        }
    }
    return true;
}

bool is_valid_dual_loop(const Mesh<Scalar>& m, const std::vector<DualSegment>& dual_loop)
{
    // Check if the loop is a valid path
    if (!is_valid_dual_path(m, dual_loop)) {
        return false;
    }

    // Loop must be at least size 1
    if (dual_loop.empty()) {
        return false;
    }

    // Check last dual segment is adjacent to the first
    if (dual_loop.back()[1] != m.opp[dual_loop.front()[0]]) {
        spdlog::error("Initial and final dual segments are not adjacent");
        return false;
    }

    return true;
}

DualSegment reverse_dual_segment(const DualSegment& dual_segment)
{
    return DualSegment({dual_segment[1], dual_segment[0]});
}

std::vector<DualSegment> reverse_dual_path(const std::vector<DualSegment>& dual_path)
{
    int num_segments = dual_path.size();
    std::vector<DualSegment> reversed_dual_path(num_segments);
    for (int i = 0; i < num_segments; ++i) {
        reversed_dual_path[num_segments - 1 - i] = reverse_dual_segment(dual_path[i]);
    }

    return reversed_dual_path;
}

int compute_dual_segment_face(const Mesh<Scalar>& m, const DualSegment& dual_segment)
{
    assert(is_valid_dual_segment(m, dual_segment));
    int h = dual_segment[0];
    return m.f[h];
}

std::vector<int> build_face_sequence_from_dual_path(
    const Mesh<Scalar>& m,
    const std::vector<DualSegment>& dual_path)
{
    // Resize face path to the size of the dual path
    int num_segments = dual_path.size();
    std::vector<int> dual_path_faces(num_segments);

    // Map dual segments to faces
    for (int i = 0; i < num_segments; ++i) {
        dual_path_faces[i] = compute_dual_segment_face(m, dual_path[i]);
    }

    return dual_path_faces;
}

std::vector<DualSegment> build_dual_path_from_face_sequence(
    const Mesh<Scalar>& m,
    const std::vector<int>& dual_loop_faces)
{
    // Resize dual loop to the size of the face loop
    int num_segments = dual_loop_faces.size();
    std::vector<DualSegment> dual_loop(num_segments);

    // Get initial halfedge for the first face
    int h = m.h[dual_loop_faces[0]];
    for (int i = 0; i < num_segments; ++i) {
        // Find the edge adjacent to the next face in the sequence
        int j = (i + 1) % num_segments; // next periodic index
        int next_face = dual_loop_faces[j];
        int h_start = h;
        while (m.f[m.opp[h]] != next_face) {
            h = m.n[h];

            // Catch full face circulation without finding the desired next face
            if (h == h_start) {
                throw std::runtime_error("Face dual loop is not connected");
            }
        }

        // Set dual loop halfedge indices for the current edge
        dual_loop[i][1] = h;
        dual_loop[j][0] = m.opp[h];

        // Increment traversal halfedge to the next face
        h = m.n[m.opp[h]];
    }

    assert(is_valid_dual_loop(m, dual_loop));
    return dual_loop;
}

void update_dual_loop_under_ccw_flip(
    const Mesh<Scalar>& m,
    int halfedge_index,
    std::vector<DualSegment>& dual_loop)
{
    assert(is_valid_dual_loop(m, dual_loop));

    // Get halfedges in the flipped quad
    int hij = halfedge_index;
    int hjk = m.n[hij];
    int hki = m.n[hjk];

    int hji = m.opp[hij];
    int hil = m.n[hji];
    int hlj = m.n[hil];

    //    Before
    //      vk
    // hki /   \ hjk
    //    / hij \ .
    //   vi --- vj
    //    \ hji / .
    // hil \   / hlj
    //      vl
    //
    //     After
    //      vk
    // hki / | \ hjk
    //    /  |  \ .
    //   vi  |  vj
    //    \  |  / .
    // hil \ | / hlj
    //      vl

    // Adjust portions of the dual loop that intersect the quad
    int num_segments = dual_loop.size();
    std::vector<DualSegment> flipped_dual_loop(0);
    flipped_dual_loop.reserve(num_segments + 10);
    for (int n = 0; n < num_segments; ++n) {
        DualSegment curr_segment = dual_loop[n];
        DualSegment next_segment = dual_loop[(n + 1) % num_segments];

        // Should never process from center of quad; skip if occurs in first iteration
        if ((curr_segment[0] == hij) || (curr_segment[0] == hji)) {
            spdlog::trace("Skipping initial segment in flipped quad");
            continue;
        }

        // There are three possible cases (up to rotational symmetry)
        //   1. The dual loop enters and leaves the quad from the same triangle
        //   2. The dual loop crosses the diagonal and exits the opposite side of the quad
        //   3. The dual loop crosses the diagonal and exits the same side of the quad
        // For each possible initial edge, we handle the above three cases in this order

        // Dual loop edge enters quad from hil
        if (curr_segment[0] == hil) {
            spdlog::trace("Loop enters at hil");
            if (curr_segment[1] == hlj) {
                flipped_dual_loop.push_back({curr_segment[0], hij});
                flipped_dual_loop.push_back({hji, curr_segment[1]});
                spdlog::trace("Loop exits same triangle");
            } else if (next_segment[0] == hij && next_segment[1] == hjk) {
                flipped_dual_loop.push_back({curr_segment[0], hij});
                flipped_dual_loop.push_back({hji, next_segment[1]});
                n += 1; // increment since two segments processed
                spdlog::trace("Loop exits opposite side");
            } else if (next_segment[0] == hij && next_segment[1] == hki) {
                flipped_dual_loop.push_back({curr_segment[0], next_segment[1]});
                n += 1; // increment since two segments processed
                spdlog::trace("Loop exits same side");
            }
        }

        // Dual loop edge enters quad from hlj
        else if (curr_segment[0] == hlj) {
            spdlog::trace("Loop enters at hlj");
            if (curr_segment[1] == hil) {
                flipped_dual_loop.push_back({curr_segment[0], hji});
                flipped_dual_loop.push_back({hij, curr_segment[1]});
                spdlog::trace("Loop exits same triangle");
            } else if (next_segment[0] == hij && next_segment[1] == hki) {
                flipped_dual_loop.push_back({curr_segment[0], hji});
                flipped_dual_loop.push_back({hij, next_segment[1]});
                n += 1; // increment since two segments processed
                spdlog::trace("Loop exits opposite side");
            } else if (next_segment[0] == hij && next_segment[1] == hjk) {
                flipped_dual_loop.push_back({curr_segment[0], next_segment[1]});
                n += 1; // increment since two segments processed
                spdlog::trace("Loop exits same side");
            }
        }

        // Dual loop edge enters quad from hjk
        else if (curr_segment[0] == hjk) {
            spdlog::trace("Loop enters at hjk");
            if (curr_segment[1] == hki) {
                flipped_dual_loop.push_back({curr_segment[0], hji});
                flipped_dual_loop.push_back({hij, curr_segment[1]});
                spdlog::trace("Loop exits same triangle");
            } else if (next_segment[0] == hji && next_segment[1] == hil) {
                flipped_dual_loop.push_back({curr_segment[0], hji});
                flipped_dual_loop.push_back({hij, next_segment[1]});
                n += 1; // increment since two segments processed
                spdlog::trace("Loop exits opposite side");
            } else if (next_segment[0] == hji && next_segment[1] == hlj) {
                flipped_dual_loop.push_back({curr_segment[0], next_segment[1]});
                n += 1; // increment since two segments processed
                spdlog::trace("Loop exits same side");
            }
        }

        // Dual loop edge enters quad from hki
        else if (curr_segment[0] == hki) {
            spdlog::trace("Loop enters at hki");
            if (curr_segment[1] == hjk) {
                flipped_dual_loop.push_back({curr_segment[0], hij});
                flipped_dual_loop.push_back({hji, curr_segment[1]});
                spdlog::trace("Loop exits same triangle");
            } else if (next_segment[0] == hji && next_segment[1] == hlj) {
                flipped_dual_loop.push_back({curr_segment[0], hij});
                flipped_dual_loop.push_back({hji, next_segment[1]});
                n += 1; // increment since two segments processed
                spdlog::trace("Loop exits opposite side");
            } else if (next_segment[0] == hji && next_segment[1] == hil) {
                flipped_dual_loop.push_back({curr_segment[0], next_segment[1]});
                n += 1; // increment since two segments processed
                spdlog::trace("Loop exits same side");
            }
        }

        // Just copy dual segment
        else {
            flipped_dual_loop.push_back(curr_segment);
        }
    }

    // Copy back to original vector
    dual_loop = flipped_dual_loop;
}

void update_dual_loop_under_ccw_flip_sequence(
    const Mesh<Scalar>& m,
    const std::vector<int>& flip_seq,
    std::vector<DualSegment>& dual_loop)
{
    Mesh<Scalar> m_flip = m;
    for (const auto& h_flip : flip_seq) {
        update_dual_loop_under_ccw_flip(m_flip, h_flip, dual_loop);
        m_flip.flip_ccw(h_flip);
    }
}

void view_dual_path(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Mesh<Scalar>& m,
    const std::vector<DualSegment>& dual_path)
{
    // Build corresponding face sequence
    std::vector<int> dual_path_faces = build_face_sequence_from_dual_path(m, dual_path);
    Eigen::VectorXd is_dual_path_face;
    is_dual_path_face.setZero(m.n_faces());
    for (const auto& dual_path_face : dual_path_faces) {
        is_dual_path_face(dual_path_face) = 1.0;
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh("path_mesh", V, F)
        ->addFaceScalarQuantity("dual_path", is_dual_path_face);
    polyscope::show();
#else
    int n_v = V.rows();
    int n_f = F.rows();
    spdlog::error("Cannot visualize dual path for mesh with {} vertices and {} faces", n_v, n_f);
#endif
}

} // namespace Holonomy
} // namespace Penner