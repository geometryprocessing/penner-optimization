#include "holonomy/core/intrinsic_field.h"

#include <gmm/gmm.h>
#include <CoMISo/Solver/ConstrainedSolver.hh>
#include <CoMISo/Solver/GMM_Tools.hh>
#include <CoMISo/Solver/MISolver.hh>
#include <CoMISo/Utils/StopWatch.hh>

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

}

void IntrinsicNRosyField::initialize_kappa(const Mesh<Scalar>& m)
{
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

void IntrinsicNRosyField::initialize_priority_kappa(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex)
{
    // Compute corner angles
    Optimization::corner_angles(m, he2angle, he2cot);

    // Compute the angle between reference halfedges across faces
    int num_halfedges = m.n_halfedges();
    kappa.setZero(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // Only process each edge once
        int hji = m.opp[hij];
        if (vtx_reindex[m.v_rep[m.to[hij]]] > vtx_reindex[m.v_rep[m.to[hji]]]) continue;

        // compute oriented angles between frames across edge eij
        kappa[hij] = compute_angle_between_frames(m, he2angle, hij);
        if (kappa[hij] <= -M_PI)
        {
            kappa[hij] += 2. * M_PI;
        }
        kappa[hji] = -kappa[hij];
        if (!(-M_PI < kappa[hij]  && kappa[hij]  <= M_PI)) {
            spdlog::error("{} out of range (-pi, pi])", kappa[hij]);
        }
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
}

void IntrinsicNRosyField::set_reference_halfedge(
    const Mesh<Scalar>& m,  
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXi& F, 
    const std::vector<int>& face_reindex,
    const Eigen::VectorXi& reference_corner)
{
    // For each face, select a reference halfedge, ensuring consistency of doubled reference
    int num_faces = m.n_faces();
    for (int f = 0; f < num_faces; ++f) {
        // get reference halfedge
        int hij = m.h[f];

        // only process original faces
        if (m.type[hij] == 2) continue;

        int vj = vtx_reindex[m.v_rep[m.to[hij]]];
        while (vj != F(face_reindex[f], 1)) {
            hij = m.n[hij];
        }

        // cycle both k and hij until the reference corner is found
        int k = 2;
        while (reference_corner[face_reindex[f]] != k)
        {
            hij = m.n[hij];
            k = (k + 1) % 3;
        }

        // Just use face to halfedge map for faces in original mesh
        face_reference_halfedge[f] = hij;
        if (m.type[hij] != 0)
        {
            int Rf = m.f[m.R[hij]]; // get reflection of the face
            face_reference_halfedge[Rf] = m.R[hij];
        }
    }

    // update kappa
    if (m.type[0] != 0)
    {
        initialize_double_kappa(m);
    } else {
        initialize_kappa(m);
    }
}

void IntrinsicNRosyField::initialize_double_kappa(const Mesh<Scalar>& m)
{
    // Compute corner angles
    Optimization::corner_angles(m, he2angle, he2cot);

    // Compute the angle between reference halfedges across faces
    int num_halfedges = m.n_halfedges();
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

void IntrinsicNRosyField::initialize_double_priority_kappa(const Mesh<Scalar>& m, const std::vector<int>& vtx_reindex)
{
    // Compute corner angles
    Optimization::corner_angles(m, he2angle, he2cot);

    // Compute the angle between reference halfedges across faces
    int num_halfedges = m.n_halfedges();
    kappa.setZero(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // Only process each edge once
        int hji = m.opp[hij];
        if (vtx_reindex[m.v_rep[m.to[hij]]] > vtx_reindex[m.v_rep[m.to[hji]]]) continue;
        if ((m.type[hij] == 2) && (m.type[hji] == 2)) continue;

        // compute oriented angles between frames across edge eij and reflected edge
        kappa[hij] = compute_angle_between_frames(m, he2angle, hij);
        if (kappa[hij] <= -M_PI)
        {
            kappa[hij] += 2. * M_PI;
        }
        kappa[hji] = -kappa[hij];
        kappa[m.R[hij]] = -kappa[hij];
        kappa[m.opp[m.R[hij]]] = kappa[hij];
        if (!(-M_PI < kappa[hij]  && kappa[hij]  <= M_PI)) {
            spdlog::error("{} out of range (-pi, pi])", kappa[hij]);
        }
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

class Rounder {
public:
    Rounder(
        const Mesh<Scalar>& m,
        const std::vector<int>& _var2he,
        const std::vector<int>& _halfedge_var_id,
        const std::vector<int>& _base_cones)
    : var2he(_var2he)
    , halfedge_var_id(_halfedge_var_id)
    , base_cones(_base_cones)
    , is_rounded(m.n_halfedges(), false)
    , values(m.n_halfedges(), 0)
    , to(vector_compose(m.v_rep, m.to))
    , opp(m.opp)
    , is_double(m.type[0] > 0)
    {
        int num_halfedges = m.n_halfedges();
        int num_vertices = m.n_ind_vertices();
        cone_period_jumps = std::vector<std::vector<int>>(num_vertices, std::vector<int>());
        for (int hij = 0; hij < num_halfedges; ++hij)
        {
            // only add variable period jumps
            if (halfedge_var_id[hij] == -1) continue;

            // add halfedge to tip
            int vj = m.v_rep[m.to[hij]];
            cone_period_jumps[vj].push_back(hij);

            // add opposite halfedge to base
            int hji = m.opp[hij];
            int vi = m.v_rep[m.to[hji]];
            cone_period_jumps[vi].push_back(hji);
        }
    }

    bool is_zero_cone(int hij)
    {
        int vj = to[hij];
        int cone = base_cones[vj];
        for (int h : cone_period_jumps[vj])
        {
            if ((h != hij) && (!is_rounded[h])) return false;
            cone += (is_double) ? (2 * values[h]) : values[h];
        }

        return (cone == 0);
    }

    int test_round(int id, double x)
    {
        int hij = var2he[id];
        int rounded_value = ((x)<0?int((x)-0.5):int((x)+0.5));
        int hji = opp[hij];
        values[hij] = rounded_value;
        values[hji] = -rounded_value;
        bool is_tip_cone = is_zero_cone(hij);
        bool is_base_cone = is_zero_cone(hji);
        
        if ((is_tip_cone) && (is_base_cone))
        {
            spdlog::error("Cone at both tip and base of period jump halfedge");
            return rounded_value;
        }
        if (is_tip_cone)
        {
            spdlog::trace("Cone at tip of period jump halfedge");
            return rounded_value + 1;
        }
        if (is_base_cone)
        {
            spdlog::trace("Cone at base of period jump halfedge");
            return rounded_value - 1;
        }

        return rounded_value;

    }

    int commit_round(int id, double x)
    {
        int rounded_value = test_round(id, x);
        int hij = var2he[id];
        int hji = opp[hij];
        values[hij] = rounded_value;
        values[hji] = -rounded_value;
        is_rounded[hij] = true;
        is_rounded[hji] = true;
        return rounded_value;
    }

private:
    std::vector<int> var2he;
    std::vector<int> halfedge_var_id;
    std::vector<int> base_cones;
    std::vector<bool> is_rounded;
    std::vector<int> values;
    std::vector<std::vector<int>> cone_period_jumps;
    std::vector<int> to;
    std::vector<int> opp;
    bool is_double;

};

class ConeMISolver : public COMISO::MISolver
{
public:
    ConeMISolver() {};
    void solve_cone_rounding( 
        CSCMatrix& _A, 
        Vecd&      _x, 
        Vecd&      _rhs, 
        Veci&      _to_round,
        Rounder& rounder) {
            // StopWatch
            COMISO::StopWatch sw;
            double time_search_next_integer = 0;

            // some statistics
            n_local_ = 0;
            n_cg_    = 0;
            n_full_  = 0;

            // reset cholmod step flag
            cholmod_step_done_ = false;

            Veci to_round(_to_round);
            // copy to round vector and make it unique
            std::sort(to_round.begin(), to_round.end());
            Veci::iterator last_unique;
            last_unique = std::unique(to_round.begin(), to_round.end());
            int r = last_unique - to_round.begin();
            to_round.resize( r);

            // initalize old indices
            Veci old_idx(_rhs.size());
            for(unsigned int i=0; i<old_idx.size(); ++i)
                old_idx[i] = i;

            if( initial_full_solution_)
            {
                if( noisy_ > 2) std::cerr << "initial full solution" << std::endl;
                direct_solver_.calc_system_gmm(_A);
                direct_solver_.solve(_x, _rhs);

                cholmod_step_done_ = true;

                ++n_full_;
            }

            // neighbors for local optimization
            Vecui neigh_i;

            // Vector for reduced solution
            Vecd xr(_x);

            // loop until solution computed
            for(unsigned int i=0; i<to_round.size(); ++i)
            {
                if( noisy_ > 0)
                {
                std::cerr << "Integer DOF's left: " << to_round.size()-(i+1) << " ";
                if( noisy_ > 1)
                    std::cerr << "residuum_norm: " << COMISO_GMM::residuum_norm( _A, xr, _rhs) << std::endl;
                }

                // position in round vector
                std::vector<int> tr_best;

                sw.start();

                RoundingSet rset;
                rset.set_threshold(multiple_rounding_threshold_);

                // find index yielding smallest rounding error
                for(unsigned int j=0; j<to_round.size(); ++j)
                {
                if( to_round[j] != -1)
                {
                int cur_idx = to_round[j];
                double rnd_error = fabs( rounder.test_round(old_idx[cur_idx], xr[cur_idx]) - xr[cur_idx]);

                rset.add(j, rnd_error);
                }
                }

                rset.get_ids( tr_best);

                time_search_next_integer += sw.stop();
            
                // nothing more to do?
                if( tr_best.empty() )
                break;

                if( noisy_ > 5)
                std::cerr << "round " << tr_best.size() << " variables simultaneously\n";

                // clear neigh for local update
                neigh_i.clear();

                for(unsigned int j = 0; j<tr_best.size(); ++j)
                {
                int i_cur = to_round[tr_best[j]];

                // store rounded value
                double rnd_x = rounder.commit_round(old_idx[i_cur], xr[i_cur]);
                _x[ old_idx[i_cur] ] = rnd_x;

                // compute neighbors
                Col col = gmm::mat_const_col(_A, i_cur);
                ColIter it  = gmm::vect_const_begin( col);
                ColIter ite = gmm::vect_const_end  ( col);
                for(; it!=ite; ++it)
                if(it.index() != (unsigned int)i_cur)
                neigh_i.push_back(it.index());

                // eliminate var
                COMISO_GMM::fix_var_csc_symmetric( i_cur, rnd_x, _A, xr, _rhs);
                to_round[tr_best[j]] = -1;
                }

                // 3-stage update of solution w.r.t. roundings
                // local GS / CG / SparseCholesky
                update_solution( _A, xr, _rhs, neigh_i);
            }

            // final full solution?
            if( final_full_solution_)
            {
                if( noisy_ > 2) std::cerr << "final full solution" << std::endl;

                if( gmm::mat_ncols( _A) > 0)
                {
                if(cholmod_step_done_)
                direct_solver_.update_system_gmm(_A);
                else
                direct_solver_.calc_system_gmm(_A);

                direct_solver_.solve( xr, _rhs);
                ++n_full_;
                }
            }

            // store solution values to result vector
            for(unsigned int i=0; i<old_idx.size(); ++i)
            {
                _x[ old_idx[i] ] = xr[i];
            }

            // output statistics
            if( stats_)
            {
                std::cerr << "\t" << __FUNCTION__ << " *** Statistics of MiSo Solver ***\n";
                std::cerr << "\t\t Number of CG    iterations  = " << n_cg_ << std::endl;
                std::cerr << "\t\t Number of LOCAL iterations  = " << n_local_ << std::endl;
                std::cerr << "\t\t Number of FULL  iterations  = " << n_full_ << std::endl;
                std::cerr << "\t\t Number of ROUNDING          = " << _to_round.size() << std::endl;
                std::cerr << "\t\t time searching next integer = " << time_search_next_integer / 1000.0 <<"s\n";
                std::cerr << std::endl;
        }

    }

};

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
    std::vector<int> var2he(x.size(), -1);
    for (int hij = 0; hij < m.n_halfedges(); ++hij) {
        int var_id = halfedge_var_id[hij];
        if (var_id != -1) {
            var_edges.push_back(var_id);
            var2he[var_id] = hij;
        }
    }

    // Empty constraints
    for (int i = 0; i < C.rows(); ++i) {
        for (int j = 0; j < C.cols(); ++j) {
            gmm_C(i, j) += (double)(C(i, j));
        }
    }

    // Solve system
    //COMISO::ConstrainedSolver cs;
    //cs.solve(gmm_C, gmm_A, x, gmm_b, var_edges, 0.0, false, true);
    gmm::csc_matrix< double > Acsc;
    gmm::copy( gmm_A, Acsc);
    std::vector<int> base_cones = generate_base_cones(m);
    Rounder rounder(m, var2he, halfedge_var_id, base_cones);
    ConeMISolver cs;
    cs.solve_cone_rounding(Acsc, x, gmm_b, var_edges, rounder);
    //COMISO::MISolver cs;
    //cs.solve(Acsc, x, gmm_b, var_edges);

    // Copy the face angles
    int num_faces = m.n_faces();
    for (int fi = 0; fi < num_faces; ++fi) {
        if (face_var_id[fi] != -1) {
            theta[fi] += x[face_var_id[fi]];
            if (m.type[m.h[fi]] != 0) {
                theta[m.f[m.R[m.h[fi]]]] -= x[face_var_id[fi]];
            }
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

void IntrinsicNRosyField::compute_principal_matchings(const Mesh<Scalar>& m)
{
    int num_halfedges = m.n_halfedges();
    VectorX rotation_form(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (hij < m.opp[hij]) continue;
        int hji = m.opp[hij];
        int f0 = m.f[hij];
        int f1 = m.f[hji];
        Scalar delta_theta = theta[f0] - theta[f1] + kappa[hij];
        period_jump[hij] = -round(delta_theta / period_value[hij]);
        period_jump[hji] = -period_jump[hij];
    }
}

std::vector<int> IntrinsicNRosyField::generate_base_cones(const Mesh<Scalar>& m) const
{
    // Compute the corner angles
    VectorX he2angle, he2cot;
    Optimization::corner_angles(m, he2angle, he2cot);

    std::vector<int> is_variable(m.n_halfedges(), false);
    for (int hij = 0; hij < m.n_halfedges(); hij++) {
        if (halfedge_var_id[hij] == -1) continue;
        is_variable[hij] = true;
        is_variable[m.opp[hij]] = true;
        if (m.type[hij] > 0)
        {
            is_variable[m.R[hij]] = true;
            is_variable[m.opp[m.R[hij]]] = true;
        }
    }

    int num_vertices = m.n_ind_vertices();
    std::vector<Scalar> Th_hat(num_vertices, 0);
    for (int hij = 0; hij < m.n_halfedges(); hij++) {
        Th_hat[m.v_rep[m.to[m.n[hij]]]] += he2angle[hij];
        Th_hat[m.v_rep[m.to[hij]]] += kappa[hij];
        if (is_variable[hij]) continue;

        Th_hat[m.v_rep[m.to[hij]]] += period_value[hij] * period_jump[hij];
    }

    std::vector<int> base_cones(num_vertices, 0);
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        base_cones[vi] = round(Th_hat[vi] / (M_PI / 2.));
    }

    return base_cones;
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
        if (use_trivial_boundary) {
            if ((m.type[hij] == 1) && (m.type[hji] == 2)) rotation_form[hij] = 0;
            if ((m.type[hij] == 2) && (m.type[hji] == 1)) rotation_form[hij] = 0;
        }
        //if (rotation_form[hij] > 6.) rotation_form[hij] -= (2. * M_PI);
        //if (rotation_form[hij] < -6.) rotation_form[hij] += (2. * M_PI);
        rotation_form[hji] = -rotation_form[hij];
    }
    assert(is_valid_one_form(m, rotation_form));

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
        bool has_boundary = false;
        do {
            vj = m.v_rep[m.to[hij]];

            // FIXME if ((m.R[m.opp[hij]] != hij) && (Th_hat[vj] > max_angle))
            if (Th_hat[vj] > max_angle)
            {
                h_opt = hij;
                max_angle = Th_hat[vj];
                spdlog::info("Max angle {} at {}", max_angle, vj);
            }
            if ((m.type[hij] == 1) && (m.type[m.opp[hij]] == 2))
            {
                spdlog::info("boundary edge found");
                has_boundary = true;
            }
            
            hij = m.opp[m.n[m.n[hij]]];
        } while (hij != h_start);

        // skip boundary edges
        // TODO: Make option
        if (has_boundary) continue;

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
        } else {
            spdlog::warn("Could not find candidate cone for correction");
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

void IntrinsicNRosyField::initialize(const Mesh<Scalar>& m)
{
    // Initialize mixed integer system
    if (m.type[0] == 0) {
        initialize_local_frames(m);
        initialize_kappa(m);
        initialize_period_jump(m);
        initialize_mixed_integer_system(m);
    } else {
        initialize_double_local_frames(m);
        initialize_double_kappa(m);
        initialize_double_period_jump(m);
        initialize_double_mixed_integer_system(m);
    }
}

VectorX IntrinsicNRosyField::run(const Mesh<Scalar>& m)
{
    initialize(m);
    solve(m);
    return compute_rotation_form(m);
}

void IntrinsicNRosyField::view(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V) const
{
    // Initialize viewer
    auto [V_double, F_mesh, F_halfedge] = generate_doubled_mesh(V, m, vtx_reindex);
    Optimization::view_dual_graph(V, m, vtx_reindex, is_period_jump_fixed);
    VectorX kappa_mesh = generate_FV_halfedge_data(F_halfedge, kappa);
    VectorX period_jump_scaled(period_jump.size());
    int num_halfedges = m.n_halfedges();
    VectorX rotation_form(num_halfedges);
    VectorX reference_rotation_form(num_halfedges);
    for (int hij = 0; hij < period_jump.size(); ++hij)
    {
        period_jump_scaled[hij] = period_value[hij] * period_jump[hij];

        int hji = m.opp[hij];
        int f0 = m.f[hij];
        int f1 = m.f[hji];
        rotation_form[hij] =
            theta[f0] - theta[f1] + kappa[hij] + period_value[hij] * period_jump[hij];
        reference_rotation_form[hij] = theta[f0] - theta[f1] + kappa[hij];
    }
    std::vector<int> base_cones = generate_base_cones(m);
    VectorX period_jump_mesh = generate_FV_halfedge_data(F_halfedge, period_jump_scaled);
    VectorX period_value_mesh = generate_FV_halfedge_data(F_halfedge, period_value);
    VectorX rotation_form_mesh = generate_FV_halfedge_data(F_halfedge, rotation_form);
    VectorX ref_rotation_form_mesh = generate_FV_halfedge_data(F_halfedge, reference_rotation_form);
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
    ->addVertexScalarQuantity("base_cones", base_cones)
    ->setColorMap("coolwarm")
    ->setEnabled(false);
polyscope::getSurfaceMesh(mesh_handle)
    ->addHalfedgeScalarQuantity(
        "period value",
        convert_scalar_to_double_vector(period_value_mesh))
    ->setColorMap("coolwarm")
    ->setEnabled(false);
polyscope::getSurfaceMesh(mesh_handle)
    ->addHalfedgeScalarQuantity(
        "rotation form",
        convert_scalar_to_double_vector(rotation_form_mesh))
    ->setColorMap("coolwarm")
    ->setEnabled(false);
polyscope::getSurfaceMesh(mesh_handle)
    ->addHalfedgeScalarQuantity(
        "reference rotation form",
        convert_scalar_to_double_vector(ref_rotation_form_mesh))
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
}

VectorX IntrinsicNRosyField::run_with_viewer(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXd& V)
{
    initialize(m);
    solve(m);
    view(m, vtx_reindex, V);

    return compute_rotation_form(m);
}

void IntrinsicNRosyField::get_field(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXi& F,
    const std::vector<int>& face_reindex,
    Eigen::VectorXi& reference_corner,
    Eigen::VectorXd& face_angle,
    Eigen::MatrixXd& corner_kappa,
    Eigen::MatrixXi& corner_period_jump) const
{
    int num_faces = m.n_faces();
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        // get reference halfedge
        int hij = face_reference_halfedge[fijk];

        // only process original faces
        if (m.type[hij] == 2) continue;

        // get local vertex index opposite reference halfedge
        int vk = vtx_reindex[m.v_rep[m.to[m.n[hij]]]];
        int k = 0;
        while (F(face_reindex[fijk], k) != vk)
        {
            k = (k + 1) % 3;
        }

        // record face angles
        reference_corner[face_reindex[fijk]] = k;
        face_angle[face_reindex[fijk]] = (double)(theta[fijk]);

        // get period jumps and rotations across edges
        for (int i = 0; i < 3; ++i)
        {
            corner_kappa(face_reindex[fijk], k) = (double)(kappa[hij]);
            corner_period_jump(face_reindex[fijk], k) = period_jump[hij];

            // increment local index and halfedge
            k = (k + 1) % 3;
            hij = m.n[hij];
        }
    }
}

void IntrinsicNRosyField::get_fixed_faces(
    const Mesh<Scalar>& m,
    const std::vector<int>& face_reindex,
    std::vector<bool>& is_fixed) const
{
    int num_faces = m.n_faces();
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        // get reference halfedge
        int hij = face_reference_halfedge[fijk];

        // only process original faces
        if (m.type[hij] == 2) continue;

        is_fixed[face_reindex[fijk]] = is_face_fixed[fijk];
    }
}

void IntrinsicNRosyField::set_field(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::MatrixXi& F, 
    const std::vector<int>& face_reindex,
    const Eigen::VectorXd& face_theta,
    const Eigen::MatrixXd& corner_kappa,
    const Eigen::MatrixXi& corner_period_jump)
{
    int num_faces = m.n_faces();
    for (int fijk = 0; fijk < num_faces; ++fijk)
    {
        // get reference halfedge
        int hij = m.h[fijk];

        // only process original faces
        if (m.type[hij] == 2) continue;

        // get local vertex index opposite reference halfedge
        int vk = vtx_reindex[m.v_rep[m.to[m.n[hij]]]];
        int k = 0;
        while (F(face_reindex[fijk], k) != vk)
        {
            k = (k + 1) % 3;
        }

        // record face angles
        theta[fijk] = Scalar(face_theta[face_reindex[fijk]]);

        // get period jumps and rotations across edges
        for (int i = 0; i < 3; ++i)
        {
            kappa[hij] = corner_kappa(face_reindex[fijk], k);
            period_jump[hij] = corner_period_jump(face_reindex[fijk], k);

            // increment local index and halfedge
            k = (k + 1) % 3;
            hij = m.n[hij];
        }

        // skip doubling on closed mesh
        if (m.type[hij] == 0) continue;

        // invert theta on reflected face
        int fjik = m.f[m.R[hij]];
        theta[fjik] = M_PI - theta[fijk];

        // invert kappa and period jumps on reflected edges
        for (int i = 0; i < 3; ++i)
        {
            kappa[m.R[hij]] = -kappa[hij];
            period_jump[m.R[hij]] = -period_jump[hij];

            // increment local index and halfedge
            k = (k + 1) % 3;
            hij = m.n[hij];
        }
    }
}

} // namespace Holonomy
} // namespace Penner
