#include "holonomy/core/intrinsic_field.h"

#include <gmm/gmm.h>
#include <CoMISo/Solver/ConstrainedSolver.hh>
#include <CoMISo/Solver/GMM_Tools.hh>
#include <CoMISo/Solver/MISolver.hh>

#include <igl/boundary_facets.h>

#include "holonomy/core/field.h"
#include "util/spanning_tree.h"
#include "holonomy/core/viewer.h"
#include "holonomy/core/forms.h"

#include "optimization/core/constraint.h"
#include "util/vector.h"

#include <stdexcept>

#ifdef ENABLE_VISUALIZATION
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Holonomy {

// compute the cone angles at vertices
VectorX compute_cone_angles(const Mesh<Scalar>& m, const VectorX& alpha)
{
    // Sum up angles around vertices
    VectorX t(m.n_ind_vertices());
    t.setZero();
    for (int h = 0; h < m.n_halfedges(); h++) {
        t[m.v_rep[m.to[m.n[h]]]] += alpha[h];
    }
    return t;
}

// build a bfs forest with faces adjacent to the boundaries as the roots
std::vector<int> build_double_dual_bfs_forest(const Mesh<Scalar>& m)
{
    int num_faces = m.n_faces();
    int num_halfedges = m.n_halfedges();
    std::vector<bool> is_processed_face(num_faces, false);
    std::vector<int> halfedge_from_face(num_faces, -1);

    // Split mesh along boundary by marking all doubled faces as processed
    for (int h = 0; h < num_halfedges; ++h) {
        if ((m.type[h] == 2) || (m.type[m.opp[h]] == 2)) {
            is_processed_face[m.f[h]] = true;
        }
    }

    // Initialize queue with boundary faces to process
    std::deque<int> faces_to_process;
    for (int h = 0; h < num_halfedges; ++h) {
        if ((m.type[h] == 2) || (m.type[m.opp[h]] == 2)) {
            int fi = m.f[h];
            is_processed_face[fi] = true;
            faces_to_process.push_back(fi);
        }
    }

    // Perform Prim or Dijkstra algorithm
    while (!faces_to_process.empty()) {
        // Get the next face to process
        int fi = faces_to_process.front();
        faces_to_process.pop_front();

        // Iterate over the face circulator via halfedges
        int h_start = m.h[fi];
        int hij = h_start;
        do {
            // Get the face in the one ring at the tip of the halfedge
            int fj = m.f[m.opp[hij]];

            // Check if the edge to the tip face is the best seen so far
            if (!is_processed_face[fj]) {
                is_processed_face[fj] = true;
                halfedge_from_face[fj] = hij;
                faces_to_process.push_back(fj);
            }

            // Progress to the next halfedge in the face circulator
            hij = m.n[hij];
        } while (hij != h_start);
    }

    return halfedge_from_face;
}

// build a dfs forest with faces adjacent to the boundaries as the roots
std::vector<int> build_double_dual_dfs_forest(const Mesh<Scalar>& m)
{
    int num_faces = m.n_faces();
    int num_halfedges = m.n_halfedges();
    std::vector<bool> is_processed_face(num_faces, false);
    std::vector<int> halfedge_from_face(num_faces, -1);

    // Split mesh along boundary by marking all doubled faces as processed
    for (int h = 0; h < num_halfedges; ++h) {
        if (m.type[h] == 2) {
            is_processed_face[m.f[h]] = true;
        }
    }

    // Initialize queue with face to process
    std::deque<int> faces_to_process;
    for (int fi = 0; fi < num_faces; ++fi) {
        if (m.type[m.h[fi]] == 1)
        {
            faces_to_process.push_back(fi);
            break;
        }
    }

    // Perform Prim or Dijkstra algorithm
    while (!faces_to_process.empty()) {
        // Get the next face to process
        int fi = faces_to_process.back();
        faces_to_process.pop_back();
        if (is_processed_face[fi]) continue;
        is_processed_face[fi] = true;

        // Iterate over the face circulator via halfedges
        int h_start = m.h[fi];
        int hij = h_start;
        do {
            // Get the face in the one ring at the tip of the halfedge
            int fj = m.f[m.opp[hij]];
            faces_to_process.push_back(fj);

            // Check if the edge to the tip face is the best seen so far
            if (!is_processed_face[fj]) {
                halfedge_from_face[fj] = hij;
            }

            // Progress to the next halfedge in the face circulator
            hij = m.n[hij];
        } while (hij != h_start);
    }

    return halfedge_from_face;
}


// compute the signed angle from the given halfedge h to the reference halfedge in the same face
Scalar IntrinsicNRosyField::compute_angle_to_reference(const Mesh<Scalar>& m, const VectorX& he2angle, int h) const
{
    // Get reference edges for the adjacent face
    int f = m.f[h];
    int h_ref = face_reference_halfedge[f];

    // Determine local orientation of h and h_ref
    int hij = h;
    int hjk = m.n[hij];
    int hki = m.n[hjk];

    // Reference halfedge is input halfedge
    if (h_ref == hij) {
        return 0.0;
    }
    // Reference halfedge is ccw from input halfedge
    else if (h_ref == hjk) {
        return (M_PI - he2angle[hki]);
    }
    // Reference halfedge is cw from input halfedge
    else if (h_ref == hki) {
        return (he2angle[hjk] - M_PI);
    }
    // Face is not triangular
    else {
        throw std::runtime_error("Cannot compute field for mesh with nontriangular face");
        return 0.0;
    }
}

// compute the signed angle from frame hij to hji
// TODO think through this
Scalar IntrinsicNRosyField::compute_angle_between_frames(const Mesh<Scalar>& m, const VectorX& he2angle, int h) const
{
    // Get angles from the edge to the reference halfedges for the adjacent faces
    int hij = h;
    int hji = m.opp[hij];
    Scalar kappa0 = compute_angle_to_reference(m, he2angle, hij);
    Scalar kappa1 = compute_angle_to_reference(m, he2angle, hji);

    // Compute angle between frames in range [-pi, pi]
    return (pos_fmod(2 * M_PI + kappa0 - kappa1, 2 * M_PI) - M_PI);
}

void IntrinsicNRosyField::initialize_local_frames(const Mesh<Scalar>& m)
{
    // For each face, select a reference halfedge
    face_reference_halfedge = m.h;

    // Set initial face angles to 0
    int num_faces = m.n_faces();
    theta.setZero(num_faces);

    // Mark faces as free except one
    is_face_fixed = std::vector<bool>(num_faces, false);
    is_face_fixed[0] = true;

    // Compute corner angles
    Optimization::corner_angles(m, he2angle, he2cot);

    // Compute the angle between reference halfedges across faces
    int num_halfedges = m.n_halfedges();
    kappa.setZero(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // Only process each edge once
        int hji = m.opp[hij];
        if (hij < hji) continue;

        // compute oriented angles between frames across edge eij
        kappa[hij] = compute_angle_between_frames(m, he2angle, hij);
        kappa[hji] = -kappa[hij];
    }
}

void IntrinsicNRosyField::initialize_double_local_frames(const Mesh<Scalar>& m)
{
    // For each face, select a reference halfedge, ensuring consistency of doubled reference
    int num_halfedges = m.n_halfedges();
    int num_faces = m.n_faces();
    face_reference_halfedge.resize(num_faces);
    theta.setZero(num_faces);
    for (int f = 0; f < num_faces; ++f) {
        int h = m.h[f];

        // Just use face to halfedge map for faces in original mesh
        if (m.type[h] != 2) {
            face_reference_halfedge[f] = h;
            theta[f] = 0.;
        }
        // Use reflection conjugated halfedge for doubled mesh
        else {
            int Rf = m.f[m.R[h]]; // get reflection of the face
            face_reference_halfedge[f] = m.R[m.h[Rf]];
            theta[f] = M_PI;
        }
    }

    // Ensure all reference halfedges on boundary are aligned
    for (int h = 0; h < num_halfedges; ++h) {
        if ((m.opp[m.R[h]] == h) && (m.type[h] == 1)) {
            int f = m.f[h];
            int Rf = m.f[m.R[h]]; // get reflection of the face
            face_reference_halfedge[f] = h;
            face_reference_halfedge[Rf] = m.R[h];
        }
    }

    // Mark double and faces on the boundary as fixed
    is_face_fixed = std::vector<bool>(num_faces, false);
    for (int f = 0; f < num_faces; ++f) {
        if (m.type[m.h[f]] == 2) {
            is_face_fixed[f] = true;
        }
    }
    for (int h = 0; h < num_halfedges; ++h) {
        if ((m.opp[m.R[h]] == h) && (m.type[h] == 1)) {
            is_face_fixed[m.f[h]] = true;
        }
    }

    // Compute corner angles
    Optimization::corner_angles(m, he2angle, he2cot);

    // Compute the angle between reference halfedges across faces
    kappa.setZero(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // Only process each edge once
        int hji = m.opp[hij];
        if (hij < hji) continue;
        if ((m.type[hij] == 2) && (m.type[hji] == 2)) continue;

        // compute oriented angles between frames across edge eij and reflected edge
        kappa[hij] = compute_angle_between_frames(m, he2angle, hij);
        kappa[hji] = -kappa[hij];
        kappa[m.R[hij]] = -kappa[hij];
        kappa[m.opp[m.R[hij]]] = kappa[hij];
    }
}

void IntrinsicNRosyField::initialize_period_jump(const Mesh<Scalar>& m)
{
    // Get edge maps
    build_edge_maps(m, he2e, e2he);

    // Build dual spanning tree
    int num_halfedges = m.n_halfedges();
    DualTree dual_tree(m, std::vector<Scalar>(num_halfedges, 0.0));

    // Initialize period jumps of value pi/2 to 0 and mark dual tree edges as fixed
    period_jump.setZero(num_halfedges);
    period_value = VectorX::Constant(num_halfedges, M_PI / 2.);
    is_period_jump_fixed = std::vector<bool>(num_halfedges, false);
    for (int h = 0; h < num_halfedges; ++h) {
        if (dual_tree.is_edge_in_tree(he2e[h])) {
            is_period_jump_fixed[h] = true;
        }
    }

    // TODO: Handle multiple fixed faces and boundary constraints
}

void IntrinsicNRosyField::initialize_double_period_jump(const Mesh<Scalar>& m)
{
    // Get edge maps
    build_edge_maps(m, he2e, e2he);

    // Initialize period jumps of value pi/2 to 0
    int num_halfedges = m.n_halfedges();
    period_value = VectorX::Constant(num_halfedges, M_PI / 2.);

    // Change period value for boundary edges to pi
    for (int h = 0; h < num_halfedges; ++h) {
        if (m.opp[m.R[h]] == h) {
            period_value[h] = M_PI; 
        }
    }

    // Mark edges in dual spanning tree and double as fixed
    // TODO: Mark first boundary period jump as fixed?
    constrain_bd = false;
    is_period_jump_fixed = std::vector<bool>(num_halfedges, false);
    if (constrain_bd)
    {
        std::vector<int> halfedges_from_face;
        halfedges_from_face = build_double_dual_bfs_forest(m);
        for (int h : halfedges_from_face) {
            if (h < 0) continue;
            for (int h_rel : { h, m.opp[h], m.R[h], m.opp[m.R[h]] }) {
                is_period_jump_fixed[h_rel] = true;
            }
        }
    } else {
        std::vector<int> halfedges_from_face;
        halfedges_from_face = build_double_dual_bfs_forest(m);
        for (int h : halfedges_from_face) {
            if (h < 0) continue;
            for (int h_rel : { h, m.opp[h], m.R[h], m.opp[m.R[h]] }) {
                is_period_jump_fixed[h_rel] = true;
            }
        }
        for (int h = 0; h < num_halfedges; ++h) {
            if ((m.opp[m.R[h]] == h) && (m.type[h] == 1)) {
                is_period_jump_fixed[h] = true;
                is_period_jump_fixed[m.opp[h]] = true;
            }
        }

    //    DualTree dual_tree(m, std::vector<Scalar>(num_halfedges, 0.0));
    //    for (int h = 0; h < num_halfedges; ++h) {
    //        if ((m.type[h] == 2) || (m.type[m.opp[h]] == 2)) continue;

    //        if (dual_tree.is_edge_in_tree(he2e[h])) {
    //            is_period_jump_fixed[h] = true;
    //        }
    //    }
    //    for (int h = 0; h < num_halfedges; ++h) {
    //        if (m.opp[m.R[h]] == h) {
    //            is_period_jump_fixed[h] = true;
    //            is_period_jump_fixed[m.opp[h]] = true;
    //            break;
    //        }
    //    }
    }

    // set the period jump (necessary for jumps between fixed faces)
    period_jump.setZero(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        int hji = m.opp[hij];
        if (hij < hji) continue; // only process each edge once
        if ((m.type[hij] == 2) && (m.type[m.opp[hij]] == 2)) continue;
        
        int fi = m.f[hij];
        int fj = m.f[hij];
        period_jump[hij] = (int)(round((theta[fi] - theta[fj] - kappa[hij])/period_value[hij]));
        period_jump[hji] = -period_jump[hij];
        period_jump[m.R[hij]] = -period_jump[hij];
        period_jump[m.opp[m.R[hij]]] = period_jump[hij];
    }

    // TODO: Handle multiple fixed faces and boundary constraints
}

std::tuple<std::vector<int>, std::vector<int>> build_variable_index(
    const std::vector<bool>& is_fixed)
{
    int num_var = is_fixed.size();

    // Allocate maps from variables to all values to variable and the left inverse
    std::vector<int> all2var(num_var, -1);
    std::vector<int> var2all;
    var2all.reserve(num_var);

    // Build maps
    for (int i = 0; i < num_var; ++i) {
        if (!is_fixed[i]) {
            all2var[i] = var2all.size();
            var2all.push_back(i);
        }
    }

    return std::make_tuple(var2all, all2var);
}

void IntrinsicNRosyField::initialize_mixed_integer_system(const Mesh<Scalar>& m)
{
    int num_faces = m.n_faces();
    int num_halfedges = m.n_halfedges();

    // Count and tag the variables
    face_var_id = std::vector<int>(num_faces, -1);
    halfedge_var_id = std::vector<int>(num_halfedges, -1);
    int count = 0;
    for (int fi = 0; fi < num_faces; ++fi) {
        if (!is_face_fixed[fi]) {
            face_var_id[fi] = count;
            count++;
        }
    }
    int num_face_var = count;
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (hij < m.opp[hij]) continue; // only process each edge once
        if (!is_period_jump_fixed[hij]) {
            halfedge_var_id[hij] = count;
            count++;
        }
    }
    int num_var = count;
    int num_edge_var = num_var - num_face_var;

    b.setZero(num_var);
    std::vector<Eigen::Triplet<Scalar>> T;
    T.reserve(3 * 4 * num_edge_var);

    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (hij < m.opp[hij]) continue; // only process each edge once
        int hji = m.opp[hij];
        int f0 = m.f[hij];
        int f1 = m.f[hji];
        int f0_var = face_var_id[f0];
        int f1_var = face_var_id[f1];
        int e_var = halfedge_var_id[hij];
        int row;

        // partial with respect to f0
        if (!is_face_fixed[f0]) {
            row = f0_var;
            T.emplace_back(row, f0_var, 2);

            if (is_face_fixed[f1]) {
                b(row) += 2 * theta[f1];
            } else {
                T.emplace_back(row, f1_var, -2);
            }

            if (is_period_jump_fixed[hij]) {
                b(row) += -2. * period_value[hij] * period_jump[hij];
            } else {
                T.emplace_back(row, e_var, 2. * period_value[hij]);
            }

            b(row) += -2 * kappa[hij];
        }
        // partial with respect to f1
        if (!is_face_fixed[f1]) {
            row = f1_var;
            T.emplace_back(row, f1_var, 2);

            if (is_face_fixed[f0]) {
                b(row) += 2 * theta[f0];
            } else {
                T.emplace_back(row, f0_var, -2);
            }

            if (is_period_jump_fixed[hij]) {
                b(row) += 2. * period_value[hij] * period_jump[hij];
            } else {
                T.emplace_back(row, e_var, -2. * period_value[hij]);
            }

            b(row) += 2 * kappa[hij];
        }
        // partial with respect to eij
        if (!is_period_jump_fixed[hij]) {
            row = e_var;
            T.emplace_back(row, e_var, 2. * period_value[hij] * period_value[hij]);

            if (is_face_fixed[f0]) {
                b(row) += -2. * period_value[hij] * theta[f0];
            } else {
                T.emplace_back(row, f0_var, 2. * period_value[hij]);
            }

            if (is_face_fixed[f1]) {
                b(row) += 2. * period_value[hij] * theta[f1];
            } else {
                T.emplace_back(row, f1_var, -2. * period_value[hij]);
            }

            b(row) += -2. * period_value[hij] * kappa[hij];
        }
    }

    A.resize(num_var, num_var);
    A.setFromTriplets(T.begin(), T.end());
    spdlog::debug("Cross field system vector has norm {}", b.norm());
    spdlog::debug("Cross field system matrix has norm {}", A.norm());

    // TODO: Soft constraints
}

void IntrinsicNRosyField::initialize_double_mixed_integer_system(const Mesh<Scalar>& m)
{
    int num_faces = m.n_faces();
    int num_halfedges = m.n_halfedges();

    // Count and tag the variables
    face_var_id = std::vector<int>(num_faces, -1);
    halfedge_var_id = std::vector<int>(num_halfedges, -1);
    int count = 0;
    for (int fi = 0; fi < num_faces; ++fi) {
        if ((m.type[m.h[fi]] == 1) && (!is_face_fixed[fi])) {
            face_var_id[fi] = count;
            count++;
        }
    }
    int num_face_var = count;
    for (int hij = 0; hij < num_halfedges; ++hij) {
        int hji = m.opp[hij];
        if (is_period_jump_fixed[hij]) continue;
        if ((m.type[hij] == 2) && (m.type[hji] == 2))
            continue; // don't process interior reflection
        if (hij < hji) continue; // unique index check

        halfedge_var_id[hij] = count;
        count++;
    }
    int num_var = count;
    int num_edge_var = num_var - num_face_var;

    b.setZero(num_var);
    std::vector<Eigen::Triplet<Scalar>> T;
    T.reserve(3 * 4 * num_edge_var);

    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (hij < m.opp[hij]) continue; // only process each edge once
        if ((m.type[hij] == 2) && (m.type[m.opp[hij]] == 2)) continue;
        int hji = m.opp[hij];
        int f0 = m.f[hij];
        int f1 = m.f[hji];
        int f0_var = face_var_id[f0];
        int f1_var = face_var_id[f1];
        int e_var = halfedge_var_id[hij];
        int row;

        bool use_boundary_energy = true;
        bool is_boundary = (m.type[hij] == 2) || (m.type[m.opp[hij]] == 2);
        if ((use_boundary_energy) && (is_boundary)) {
            if (!is_face_fixed[f0]) {
                row = f0_var;
                T.emplace_back(f0_var, f0_var, 4.);

                if (is_period_jump_fixed[hij]) {
                    b(row) += -2. * period_value[hij] * period_jump[hij];
                } else {
                    T.emplace_back(row, e_var, 2. * period_value[hij]);
                }
            }

            if (!is_face_fixed[f1]) {
                row = f1_var;
                T.emplace_back(f1_var, f1_var, 4.);

                if (is_period_jump_fixed[hij]) {
                    b(row) += 2. * period_value[hij] * period_jump[hij];
                } else {
                    T.emplace_back(row, e_var, -2. * period_value[hij]);
                }
            }

            if (!is_period_jump_fixed[hij]) {
                row = e_var;
                T.emplace_back(row, e_var, 2. * period_value[hij] * period_value[hij]);

                if (!is_face_fixed[f0]) {
                    T.emplace_back(row, f0_var, 4. * period_value[hij]);
                }

                if (!is_face_fixed[f1]) {
                    T.emplace_back(row, f1_var, -4. * period_value[hij]);
                }
            }
            continue;
        }

        // partial with respect to f0
        if (!is_face_fixed[f0]) {
            row = f0_var;
            T.emplace_back(row, f0_var, 4.);

            if (is_face_fixed[f1]) {
                b(row) += 4. * theta[f1];
            } else {
                b(row) += 4. * theta[f1];
                T.emplace_back(row, f1_var, -4.);
            }

            if (is_period_jump_fixed[hij]) {
                b(row) += -4. * period_value[hij] * period_jump[hij];
            } else {
                T.emplace_back(row, e_var, 4. * period_value[hij]);
            }

            b(row) += -4. * kappa[hij];
        }
        // partial with respect to f1
        if (!is_face_fixed[f1]) {
            row = f1_var;
            T.emplace_back(row, f1_var, 4);

            if (is_face_fixed[f0]) {
                b(row) += 4. * theta[f0];
            } else {
                b(row) += 4. * theta[f0];
                T.emplace_back(row, f0_var, -4.);
            }

            if (is_period_jump_fixed[hij]) {
                b(row) += 4. * period_value[hij] * period_jump[hij];
            } else {
                T.emplace_back(row, e_var, -4. * period_value[hij]);
            }

            b(row) += 4. * kappa[hij];
        }
        // partial with respect to eij
        if (!is_period_jump_fixed[hij]) {
            row = e_var;
            T.emplace_back(row, e_var, 4. * period_value[hij] * period_value[hij]);

            if (is_face_fixed[f0]) {
                b(row) += -4. * period_value[hij] * theta[f0];
            } else {
                b(row) += -4. * period_value[hij] * theta[f0];
                T.emplace_back(row, f0_var, 4. * period_value[hij]);
            }

            if (is_face_fixed[f1]) {
                b(row) += 4. * period_value[hij] * theta[f1];
            } else {
                b(row) += 4. * period_value[hij] * theta[f1];
                T.emplace_back(row, f1_var, -4. * period_value[hij]);
            }

            b(row) += -4. * period_value[hij] * kappa[hij];
        }
    }

    A.resize(num_var, num_var);
    A.setFromTriplets(T.begin(), T.end());

    // TODO: Soft constraints
    if (!constrain_bd) return;

    // List boundary halfedges in the original mesh
    std::vector<int> bd_halfedges = {};
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if ((m.type[hij] == 1) && (m.type[m.opp[hij]] == 2)) {
            bd_halfedges.push_back(hij);
        }
    }

    // Build constraint
    int num_vertices = m.n_vertices();
    int num_bd_vertices = bd_halfedges.size();
    int num_int_vertices = num_vertices - num_bd_vertices;
    int euler_char = m.n_vertices() - m.n_edges() + m.n_faces();
    Scalar target_curvature = 2. * M_PI * euler_char;
    VectorX cone_angles = compute_cone_angles(m, he2angle);
    VectorX target_indices(num_bd_vertices);
    bool use_corners = true;
    Scalar boundary_curvature = 0.;
    for (int i = 0; i < num_bd_vertices; ++i) {
        int hij = bd_halfedges[i];
        int vi = m.v_rep[m.to[m.opp[hij]]];

        // Initialize constant to 2 pi
        target_indices(i) = 2. * M_PI;
        if (use_corners)
        {
            target_indices(i) = M_PI * max(round(cone_angles[vi]/ M_PI), 1.);
            spdlog::trace("Setting cone constraint {} to {}", vi, target_indices(i));
        }
        boundary_curvature += ((2. * M_PI) - target_indices(i));
    }
    //C.setZero(num_bd_vertices, num_var + 1);

   for (int i = 0; i < num_bd_vertices; ++i) {
        int h_var = bd_halfedges[i];
        h_var = m.opp[m.n[m.n[h_var]]];

        // TODO Generalize to case where bd not fixed
        while (m.type[h_var] != 2) {
            if (!is_period_jump_fixed[h_var])
            {
                int row;
                int sign;
                if (h_var < m.opp[h_var])
                {
                    row = halfedge_var_id[m.opp[h_var]];
                    sign = -100000.;
                }
                else {
                    row = halfedge_var_id[h_var];
                    sign = 100000.;
                }
                int hij = bd_halfedges[i];

                if (!is_period_jump_fixed[hij]) {
                    if (hij < m.opp[hij])
                    {
                        int e_var = halfedge_var_id[m.opp[hij]];
                        T.emplace_back(row, e_var, -sign * period_value[hij]);
                    } else {
                        int e_var = halfedge_var_id[hij];
                        T.emplace_back(row, e_var, sign * period_value[hij]);
                    }
                } else {
                    b(row) -= sign * period_value[hij] * period_jump[hij];
                }
                b(row) += sign * kappa[hij];
                b(row) -= sign * 2. * he2angle[m.n[hij]];

                hij = m.opp[m.n[m.n[hij]]];

                // Circulate ccw until other boundary edge reached
                while (m.type[hij] != 2) {
                    int hji = m.opp[hij];

                    // add period jump constraint term for given edge
                    if (!is_period_jump_fixed[hij]) {
                        if (hij < hji)
                        {
                            int e_var = halfedge_var_id[hji];
                            T.emplace_back(row, e_var, -sign * 2. * period_value[hij]);
                        } else {
                            int e_var = halfedge_var_id[hij];
                            T.emplace_back(row, e_var, sign * 2. * period_value[hij]);
                        }
                    } else {
                        b(row) -= sign * 2. * period_value[hij] * period_jump[hij];
                    }

                    // add original index term for given edge
                    b(row) += sign * 2. * kappa[hij];
                    b(row) -= sign * 2. * he2angle[m.n[hij]];

                    // circulate halfedge 
                    hij = m.opp[m.n[m.n[hij]]];
                }
                
                // Add constraints for last boundary edge
                if (!is_period_jump_fixed[hij]) {
                    if (hij < m.opp[hij])
                    {
                        int e_var = halfedge_var_id[m.opp[hij]];
                        T.emplace_back(row, e_var, -sign * period_value[hij]);
                    } else {
                        int e_var = halfedge_var_id[hij];
                        T.emplace_back(row, e_var, sign * period_value[hij]);
                    }
                } else {
                    b(row) -= sign * period_value[hij] * period_jump[hij];
                }
                b(row) += sign * kappa[hij];
                b(row) += sign * target_indices(i);
            }

            h_var = m.opp[m.n[m.n[h_var]]];
        }
   }

    spdlog::info("Adding constraints");


    bool correct_curvature = true;
    if (correct_curvature)
    {
        Scalar curvature_defect = target_curvature - boundary_curvature;
        spdlog::info("Correcting curvature defect {} for {} interior vertices", curvature_defect, num_int_vertices);
        while (curvature_defect < (-M_PI * num_int_vertices))
        //while (curvature_defect < 0)
        {
            // TODO Be more clever
            Scalar min_defect = 3. * M_PI;
            int min_index = -1;
            for (int i = 0; i < num_bd_vertices; ++i) {
                if (min_defect > C(i, num_var))
                {
                    min_defect = C(i, num_var);
                    min_index = i;
                }
            }


            C(min_index, num_var) +=  M_PI;
            curvature_defect +=  M_PI;
        }
        while (curvature_defect > (M_PI * num_int_vertices))
        //while (curvature_defect > 0)
        {
            // TODO Be more clever
            Scalar max_defect = -1.;
            int max_index = -1;
            for (int i = 0; i < num_bd_vertices; ++i) {
                if (max_defect < C(i, num_var))
                {
                    max_defect = C(i, num_var);
                    max_index = i;
                }
            }

            C(max_index, num_var) -=  M_PI;
            curvature_defect -=  M_PI;
            
        }
        spdlog::info("Curvature defect {}", curvature_defect);
    }

    for (int i = 0; i < num_bd_vertices; ++i) {
        int hij = bd_halfedges[i];

        // Add constraint term for first face
        // TODO Check if theta necessary
        //int f0 = m.f[hij];
        //int f0_var = face_var_id[f0];
        //if (!is_face_fixed[f0]) {
        //    C(i, f0_var) = 1;
        //} else {
        //    C(i, num_var + 1) -= theta[f0];
        //}
        if (!is_period_jump_fixed[hij]) {
            if (hij < m.opp[hij])
            {
                int e_var = halfedge_var_id[m.opp[hij]];
                C(i, e_var) -= period_value[hij];
            } else {
                int e_var = halfedge_var_id[hij];
                C(i, e_var) += period_value[hij];
            }
        } else {
            C(i, num_var) -= period_value[hij] * period_jump[hij];
        }
        C(i, num_var) += kappa[hij];
        C(i, num_var) -= 2. * he2angle[m.n[hij]];

        hij = m.opp[m.n[m.n[hij]]];

        // Circulate ccw until other boundary edge reached
        while (m.type[hij] != 2) {
            int hji = m.opp[hij];

            // add period jump constraint term for given edge
            if (!is_period_jump_fixed[hij]) {
                if (hij < hji)
                {
                    int e_var = halfedge_var_id[hji];
                    C(i, e_var) -= 2. * period_value[hij];
                } else {
                    int e_var = halfedge_var_id[hij];
                    C(i, e_var) += 2. * period_value[hij];
                }
            } else {
                C(i, num_var) -= 2. * period_value[hij] * period_jump[hij];
            }

            // add original index term for given edge
            C(i, num_var) += 2. * kappa[hij];
            C(i, num_var) -= 2. * he2angle[m.n[hij]];

            // circulate halfedge 
            hij = m.opp[m.n[m.n[hij]]];
        }
        
        // Add constraints for last boundary edge
        if (!is_period_jump_fixed[hij]) {
            if (hij < m.opp[hij])
            {
                int e_var = halfedge_var_id[m.opp[hij]];
                C(i, e_var) -= period_value[hij];
            } else {
                int e_var = halfedge_var_id[hij];
                C(i, e_var) += period_value[hij];
            }
        } else {
            C(i, num_var) -= period_value[hij] * period_jump[hij];
        }
        C(i, num_var) += kappa[hij];

        spdlog::trace("Constraint {} is {}", i, C(i, num_var));

    }

    // Scale C by pi/2 so it's integer multipled
    C /= (M_PI / 2.);

    // TODO: Soft constraints
}

void IntrinsicNRosyField::solve(const Mesh<Scalar>& m)
{
    int n = A.rows();
    int c = C.rows();

    gmm::col_matrix<gmm::wsvector<double>> gmm_A(n, n);
    std::vector<double> gmm_b(n);
    gmm::row_matrix<gmm::wsvector<double>> gmm_C(c, n+1);
    std::vector<int> var_edges;
    std::vector<double> x(n);

    // Copy A
    for (int k = 0; k < A.outerSize(); ++k) {
        for (MatrixX::InnerIterator it(A, k); it; ++it) {
            gmm_A(it.row(), it.col()) += (double)(it.value());
        }
    }

    // Copy b
    for (int i = 0; i < n; ++i) {
        gmm_b[i] = (double)(b[i]);
    }

    // Set variables to round
    var_edges.clear();
    for (int var_id : halfedge_var_id) {
        if (var_id != -1) {
            var_edges.push_back(var_id);
        }
    }

    // Empty constraints
    for (int i = 0; i < C.rows(); ++i) {
        for (int j = 0; j < C.cols(); ++j) {
            gmm_C(i, j) += (double)(C(i, j));
        }
    }

    // Solve system
    COMISO::ConstrainedSolver cs;
    cs.solve(gmm_C, gmm_A, x, gmm_b, var_edges, 0.0, false, true);

    // Copy the face angles
    int num_faces = m.n_faces();
    for (int fi = 0; fi < num_faces; ++fi) {
        if (face_var_id[fi] != -1) {
            theta[fi] += x[face_var_id[fi]];
            theta[m.f[m.R[m.h[fi]]]] -= x[face_var_id[fi]];
        }
    }

    // Copy the period jumps (and add sign)
    int num_halfedges = m.n_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (halfedge_var_id[hij] != -1) {
            //if ((m.type[hij] == 2) || (m.type[m.opp[hij]] == 2)) continue;
            int hji = m.opp[hij];
            period_jump[hij] = (int)std::round(x[halfedge_var_id[hij]]);
            period_jump[hji] = -period_jump[hij];

            if (m.type[hij] > 0)
            {
                period_jump[m.R[hij]] = -period_jump[hij];
                period_jump[m.opp[m.R[hij]]] = period_jump[hij];
            }
        }
    }
}

std::vector<Scalar> generate_cones_from_rotation_form_FIXME(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form)
{
    // Compute the corner angles
    VectorX he2angle, he2cot;
    Optimization::corner_angles(m, he2angle, he2cot);

    // Compute cones from the rotation form as holonomy - rotation around each vertex
    // Per-halfedge iteration is used for faster computation
    int num_vertices = m.n_ind_vertices();
    std::vector<Scalar> Th_hat(num_vertices, 0.);
    for (int h = 0; h < m.n_halfedges(); h++) {
        // Add angle to vertex opposite the halfedge
        Th_hat[m.v_rep[m.to[m.n[h]]]] += he2angle[h];

        // Add rotation to the vertex at the tip of the halfedge
        // NOTE: By signing convention, this is the negative of the rotation ccw around
        // the vertex
        Th_hat[m.v_rep[m.to[h]]] += rotation_form[h];
    }

    return Th_hat;
}

VectorX IntrinsicNRosyField::compute_rotation_form(const Mesh<Scalar>& m)
{
    int num_halfedges = m.n_halfedges();
    VectorX rotation_form(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (hij < m.opp[hij]) continue;
        int hji = m.opp[hij];
        int f0 = m.f[hij];
        int f1 = m.f[hji];
        rotation_form[hij] =
            theta[f0] - theta[f1] + kappa[hij] + period_value[hij] * period_jump[hij];
        rotation_form[hji] = -rotation_form[hij];
    }

    std::vector<Scalar> Th_hat = generate_cones_from_rotation_form_FIXME(m, rotation_form);

    int double_genus = 2 - (m.n_vertices() - m.n_edges() + m.n_faces());
    Scalar targetsum = M_PI * (2 * m.n_vertices() - 2 * (2 - double_genus));
    Scalar th_hat_sum = 0.0;
    for(auto t: Th_hat)
    {
      th_hat_sum += t;
    }
    spdlog::info("Gauss-Bonnet error before cone removal: {}", th_hat_sum - targetsum);

    int num_vertices = Th_hat.size();
    std::queue<int> cones = {};
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        if (Th_hat[vi] < min_angle - 1e-6)
        {
            cones.push(vi);
        }
    }
    while (!cones.empty())
    {
        int vi = cones.front();
        cones.pop();
        if (Th_hat[vi] > min_angle + 1e-6) continue;
        spdlog::info("Fixing {} cone at {} in rotation form", Th_hat[vi], vi);

        // find largest cone angle near the cone vertex
        int h_opt = -1;
        Scalar max_angle = -1.;
        int h_start = m.out[vi];
        int hij = h_start;
        int vj;
        do {
            vj = m.v_rep[m.to[hij]];

            // skip boundary edges
            if ((m.R[m.opp[hij]] != hij) && (Th_hat[vj] > max_angle))
            {
                h_opt = hij;
                max_angle = Th_hat[vj];
                spdlog::info("Max angle {} at {}", max_angle, vj);
            }
            
            hij = m.opp[m.n[m.n[hij]]];
        } while (hij != h_start);

        // push curvature to adjacent cone if candidate found
        if (h_opt != -1)
        {
            vj = m.v_rep[m.to[h_opt]];
            spdlog::info("Decreasing cone {} at {}", Th_hat[vj], vj);

            // modify the rotation form and resulting cone angles
            rotation_form[h_opt] -= (M_PI / 2.);
            rotation_form[m.opp[h_opt]] += M_PI / 2.;
            if (m.type[h_opt] != 0) {
                rotation_form[m.R[h_opt]] += M_PI / 2.;
                rotation_form[m.opp[m.R[h_opt]]] -= M_PI / 2.;
            }
            Th_hat[vi] += M_PI / 2.;
            Th_hat[vj] -= M_PI / 2.;

            // check if candidate vertex is now a cone
            if (Th_hat[vj] < min_angle - 1e-6) 
            {
                cones.push(vj);
            }
        }

        // check if vertex vi is still a cone
        if (Th_hat[vi] < min_angle - 1e-6) 
        {
            cones.push(vi);
        }
    }

    th_hat_sum = 0.0;
    for(auto t: Th_hat)
    {
      th_hat_sum += t;
    }
    spdlog::info("Gauss-Bonnet error after cone removal: {}", th_hat_sum - targetsum);

    assert(is_valid_one_form(m, rotation_form));
    return rotation_form;
}

VectorX IntrinsicNRosyField::run(const Mesh<Scalar>& m)
{
    // Initialize mixed integer system
    if (m.type[0] == 0) {
        initialize_local_frames(m);
        initialize_period_jump(m);
        initialize_mixed_integer_system(m);
    } else {
        initialize_double_local_frames(m);
        initialize_double_period_jump(m);
        initialize_double_mixed_integer_system(m);
    }

    // Solve
    solve(m);
    return compute_rotation_form(m);
}

VectorX IntrinsicNRosyField::run_with_viewer(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V)
{
    // Initialize mixed integer system
    if (m.type[0] == 0) {
        initialize_local_frames(m);
        initialize_period_jump(m);
        initialize_mixed_integer_system(m);
    } else {
        initialize_double_local_frames(m);
        initialize_double_period_jump(m);
        initialize_double_mixed_integer_system(m);
    }

    // Solve
    solve(m);

    // Initialize viewer
    auto [V_double, F_mesh, F_halfedge] = generate_doubled_mesh(V, m, vtx_reindex);
    view_dual_graph(V, m, vtx_reindex, is_period_jump_fixed);
    VectorX kappa_mesh = generate_FV_halfedge_data(F_halfedge, kappa);
    VectorX period_jump_scaled(period_jump.size());
    for (int i = 0; i < period_jump.size(); ++i)
    {
        period_jump_scaled[i] = period_value[i] * period_jump[i];
    }
    VectorX period_jump_mesh = generate_FV_halfedge_data(F_halfedge, period_jump_scaled);
    VectorX period_value_mesh = generate_FV_halfedge_data(F_halfedge, period_value);
#ifdef ENABLE_VISUALIZATION
polyscope::init();
std::string mesh_handle = "intrinsic_field_mesh";
polyscope::registerSurfaceMesh(mesh_handle, V_double, F_mesh);
polyscope::getSurfaceMesh(mesh_handle)
    ->setBackFacePolicy(polyscope::BackFacePolicy::Cull);
polyscope::getSurfaceMesh(mesh_handle)
    ->addHalfedgeScalarQuantity(
        "kappa",
        convert_scalar_to_double_vector(kappa))
    ->setColorMap("coolwarm")
    ->setEnabled(false);
polyscope::getSurfaceMesh(mesh_handle)
    ->addHalfedgeScalarQuantity(
        "period jump",
        convert_scalar_to_double_vector(period_jump_mesh))
    ->setColorMap("coolwarm")
    ->setEnabled(false);
polyscope::getSurfaceMesh(mesh_handle)
    ->addHalfedgeScalarQuantity(
        "period value",
        convert_scalar_to_double_vector(period_value_mesh))
    ->setColorMap("coolwarm")
    ->setEnabled(false);
polyscope::getSurfaceMesh(mesh_handle)
    ->addFaceScalarQuantity(
        "theta",
        convert_scalar_to_double_vector(theta))
    ->setColorMap("coolwarm")
    ->setEnabled(true);
polyscope::show();
#endif

    return compute_rotation_form(m);
}

} // namespace Holonomy
} // namespace Penner
