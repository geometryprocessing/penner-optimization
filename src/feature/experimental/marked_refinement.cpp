#include "feature/experimental/marked_refinement.h"

#include "util/boundary.h"
#include "util/vector.h"
#include "holonomy/core/viewer.h"
#include "feature/core/component_mesh.h"

namespace Penner {
namespace Feature {


// Refine a single face, disregarding symmetry structure and independent vertices
//       k
//     / | \ .
//    /  |  \ .
//   /   l   \ .
//  /  /   \  \ .
// i ---------- j
int MarkedRefinementMesh::refine_single_face(int face_index)
{
    Mesh<Scalar>& m = get_mesh();

    // get halfedges and vertices of triangle by clockwise circulation
    int hij = m.h[face_index];
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int vi = m.to[hki];
    int vj = m.to[hij];
    int vk = m.to[hjk];

    // get new vertex, face and hafledge indices
    int vl = get_new_vertex();
    int fijl = face_index;
    int fjkl = get_new_face();
    int fkil = get_new_face();
    auto [hil, hli] = get_new_edge();
    auto [hjl, hlj] = get_new_edge();
    auto [hkl, hlk] = get_new_edge();

    // set connectivity for faces
    auto set_face = [&m](int h0, int h1, int h2, int f) {
        m.n[h0] = h1;
        m.n[h1] = h2;
        m.n[h2] = h0;
        m.f[h0] = f;
        m.f[h1] = f;
        m.f[h2] = f;
        m.h[f] = h0;
    };
    set_face(hij, hjl, hli, fijl);
    set_face(hjk, hkl, hlj, fjkl);
    set_face(hki, hil, hlk, fkil);

    // set connectivity for vertices
    m.to[hil] = vl;
    m.to[hli] = vi;
    m.to[hjl] = vl;
    m.to[hlj] = vj;
    m.to[hkl] = vl;
    m.to[hlk] = vk;
    m.out[vl] = hli;

    // set length of all edges to 1 and type to that of the face
    // TODO Use midpoint length; only topology used currently
    int type = m.type[hij];
    int type_input = m.type_input[hij];
    for (int h : {hil, hli, hjl, hlj, hkl, hlk}) {
        m.l[h] = 1.;
        m.type[h] = type;
        m.type_input[h] = type_input;
    }

    // refine all dual loops
    // TODO optimize linear pass
    int fij_opp = m.f[m.opp[hij]];
    int fjk_opp = m.f[m.opp[hjk]];
    int fki_opp = m.f[m.opp[hki]];
    auto& dual_loops = get_dual_loops();
    for (std::vector<int>& dual_loop : dual_loops) {
        int num_faces = dual_loop.size();
        auto loop_index = [&num_faces](int i) { return (i + num_faces) % num_faces; };
        std::vector<int> new_dual_loop = {};
        for (int i = 0; i < num_faces; ++i) {
            int curr_face = dual_loop[loop_index(i)];
            // handle case for refined face
            if (curr_face == face_index)
            {
                // add refined face opposite previous face
                int prev_face = dual_loop[loop_index(i - 1)];
                if (prev_face == fij_opp)
                    new_dual_loop.push_back(fijl);
                else if (prev_face == fjk_opp)
                    new_dual_loop.push_back(fjkl);
                else if (prev_face == fki_opp)
                    new_dual_loop.push_back(fkil);

                // add refined face opposite next face
                int next_face = dual_loop[loop_index(i + 1)];
                if (next_face == fij_opp)
                    new_dual_loop.push_back(fijl);
                else if (next_face == fjk_opp)
                    new_dual_loop.push_back(fjkl);
                else if (next_face == fki_opp)
                    new_dual_loop.push_back(fkil);
            }
            // if face not refined, just add to list
            else
            {
                new_dual_loop.push_back(curr_face);
            }
        }

        // overwrite dual loop
        dual_loop = new_dual_loop;
    }

    return vl;
}

// refine halfedge, without considering symmetry
int MarkedRefinementMesh::refine_single_halfedge(int halfedge_index)
{
    Mesh<Scalar>& m = get_mesh();

    // get halfedges and vertices of triangle by clockwise circulation
    int hij = halfedge_index;
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int hji = m.opp[hij];
    int hil = m.n[hji];
    int hlj = m.n[hil];
    int fijk = m.f[hij];
    int fjil = m.f[hji];
    int vi = m.to[hki];
    int vj = m.to[hij];
    int vk = m.to[hjk];
    int vl = m.to[hil];

    // get new vertex, face and hafledge indices
    int vm = get_new_vertex();
    int fmjk = fijk;
    int fmki = get_new_face();
    int fmil = fjil;
    int fmlj = get_new_face();
    int him = hij;
    int hmi = hji;
    auto [hmj, hjm] = get_new_edge();
    auto [hmk, hkm] = get_new_edge();
    auto [hml, hlm] = get_new_edge();

    // set connectivity for faces
    auto set_face = [&m](int h0, int h1, int h2, int f) {
        m.n[h0] = h1;
        m.n[h1] = h2;
        m.n[h2] = h0;
        m.f[h0] = f;
        m.f[h1] = f;
        m.f[h2] = f;
        m.h[f] = h0;
    };
    set_face(hmj, hjk, hkm, fmjk);
    set_face(hmk, hki, him, fmki);
    set_face(hmi, hil, hlm, fmil);
    set_face(hml, hlj, hjm, fmlj);

    // set connectivity for vertices
    m.to[hmi] = vi;
    m.to[hmj] = vj;
    m.to[hmk] = vk;
    m.to[hml] = vl;
    m.to[him] = vm;
    m.to[hjm] = vm;
    m.to[hkm] = vm;
    m.to[hlm] = vm;
    m.out[vm] = hmi;
    m.out[vi] = him;
    m.out[vj] = hjm;
    m_endpoints[vm] = {vi, vj};

    // set length of all edges to 1 and type to that of the face
    //int type_0 = m.type[hij];
    //int type_input = m.type_input[hij];
    //Scalar l = 1.;
    // TODO Use midpoint length

    // refine all dual loops
    //auto& dual_loops = get_dual_loops();
    //for (std::vector<int>& dual_loop : dual_loops) {
        // TODO
    //}

    return vm;
}

MarkedPennerConeMetric MarkedRefinementMesh::generate_marked_metric(
    const std::vector<Scalar>& kappa_hat) const
{
    const auto& refined_mesh = get_mesh();
    const auto& refined_basis_loops = get_dual_loops();
    std::vector<std::unique_ptr<DualLoop>> basis_loops;
    for (const auto& basis_loop : refined_basis_loops) {
        basis_loops.push_back(std::make_unique<Holonomy::DualLoopConnectivity>(
            Holonomy::DualLoopConnectivity(Holonomy::build_dual_path_from_face_sequence(refined_mesh, basis_loop))));
    }

    int num_halfedges = refined_mesh.n_halfedges();
    VectorX refined_metric_coords(refined_mesh.n_halfedges());
    for (int h = 0; h < num_halfedges; ++h) {
        refined_metric_coords[h] = 2. * log(refined_mesh.l[h]);
    }

    return MarkedPennerConeMetric(refined_mesh, refined_metric_coords, basis_loops, kappa_hat);
}

MarkedPennerConeMetric MarkedRefinementMesh::generate_dirichlet_metric(
    const std::vector<Scalar>& kappa_hat,
    const std::vector<int>& start_vertices,
    const MatrixX& boundary_constraint_system,
    const VectorX ell) const
{
    auto marked_metric = generate_marked_metric(kappa_hat);

    // generate boundary paths
    std::vector<BoundaryPath> boundary_paths = {};
    boundary_paths.reserve(start_vertices.size());
    for (int vi : start_vertices) {
        boundary_paths.push_back(BoundaryPath(marked_metric, vi));
    }

    return DirichletPennerConeMetric(
        marked_metric,
        boundary_paths,
        boundary_constraint_system,
        ell);
}



} // namespace Feature
} // namespace Penner