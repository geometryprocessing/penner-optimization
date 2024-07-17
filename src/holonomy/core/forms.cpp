#include "holonomy/core/forms.h"

#include "optimization/core/constraint.h"
#include <queue>

namespace Penner {
namespace Holonomy {

bool is_valid_one_form(const Mesh<Scalar>& m, const VectorX& one_form)
{
    int num_halfedges = m.n_halfedges();
    if (one_form.size() != num_halfedges) {
        return false;
    }

    // Check opposite halfedges are inverse
    for (int hij = 0; hij < num_halfedges; ++hij) {
        int hji = m.opp[hij];
        Scalar xij = one_form[hij];
        Scalar xji = one_form[hji];
        if (!float_equal(xij + xji, 0)) {
            spdlog::error("Edge pair ({}, {}) have form values ({}, {})", hij, hji, xij, xji);

            return false;
        }
    }

    // Check reflected edges are inverse
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (m.type[hij] < 1) break; // only consider reflection structure

        int hji = m.R[hij];
        Scalar xij = one_form[hij];
        Scalar xji = one_form[hji];
        if (!float_equal(xij + xji, 0)) {
            spdlog::error("Edge pair ({}, {}) have form values ({}, {})", hij, hji, xij, xji);

            return false;
        }
    }

    return true;
}

bool is_closed_one_form(const Mesh<Scalar>& m, const VectorX& one_form)
{
    // Check is one form valid
    if (!is_valid_one_form(m, one_form)) {
        return false;
    }

    // Check one form values of face sum to 0
    int num_faces = m.n_faces();
    for (int f = 0; f < num_faces; ++f) {
        int hij = m.h[f];
        int hjk = m.n[hij];
        int hki = m.n[hjk];

        Scalar xij = one_form[hij];
        Scalar xjk = one_form[hjk];
        Scalar xki = one_form[hki];
        Scalar sum = xij + xjk + xki;

        if (!float_equal(sum, 0)) {
            spdlog::info("Face 1-form sum is {}", sum);
            return false;
        }
    }

    return true;
}

MatrixX build_dual_loop_basis_one_form_matrix(
    const Mesh<Scalar>& m,
    const std::vector<std::unique_ptr<DualLoop>>& dual_loops)
{
    int num_loops = dual_loops.size();
    int num_halfedges = m.n_halfedges();

    // Columns of matrix are signed segment halfedge indicators
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_halfedges);
    for (int i = 0; i < num_loops; ++i) {
        for (const auto& dual_segment : *dual_loops[i]) {
            tripletList.push_back(T(dual_segment[0], i, -1.0));
            tripletList.push_back(T(dual_segment[1], i, 1.0));
        }
    }

    // Create the matrix from the triplets
    MatrixX one_form_matrix;
    one_form_matrix.resize(num_halfedges, num_loops);
    one_form_matrix.reserve(tripletList.size());
    one_form_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    return one_form_matrix;
}

MatrixX build_closed_one_form_matrix(
    const Mesh<Scalar>& m,
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    bool eliminate_vertex)
{
    // Initialize matrix triplet list
    int num_halfedges = m.n_halfedges();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(2 * num_halfedges);

    // Build v_rep
    std::vector<int> v_rep;
    int num_vertex_forms;
    if (eliminate_vertex) {
      Optimization::build_free_vertex_map(m, v_rep, num_vertex_forms);
    } else {
        v_rep = m.v_rep;
        num_vertex_forms = m.n_ind_vertices();
    }

    // Add vertex basis forms
    for (int h = 0; h < num_halfedges; ++h) {
        int v0 = v_rep[m.to[h]];
        if (v0 >= 0) {
            tripletList.push_back(T(h, v0, 1.0));
        }

        int v1 = v_rep[m.to[m.opp[h]]];
        if (v1 >= 0) {
            tripletList.push_back(T(h, v1, -1.0));
        }
    }

    // Add homology basis forms
    int num_loops = homology_basis_loops.size();
    for (int i = 0; i < num_loops; ++i) {
        for (const auto& dual_segment : *homology_basis_loops[i]) {
            tripletList.push_back(T(dual_segment[0], num_vertex_forms + i, -1.0));
            tripletList.push_back(T(dual_segment[1], num_vertex_forms + i, 1.0));
        }
    }

    // Create the matrix from the triplets
    MatrixX one_form_matrix(num_halfedges, num_vertex_forms + num_loops);
    one_form_matrix.reserve(tripletList.size());
    one_form_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    return one_form_matrix;
}

MatrixX build_one_form_integral_matrix(
    const Mesh<Scalar>& m,
    const std::vector<bool>& cut_h,
    std::vector<bool>& is_cut_h,
    int start_h)
{
    int num_halfedges = m.n_halfedges();
    std::vector<std::map<int, Scalar>> integral_matrix_lol(num_halfedges, std::map<int, Scalar>());

    bool cut_given = !cut_h.empty();
    is_cut_h = std::vector<bool>(m.n_halfedges(), false);

    // Initialize the per-halfedge integral with the starting halfedge
    int h = m.n[m.n[start_h]];
    integral_matrix_lol[h] = std::map<int, Scalar>();
    h = m.n[h];
    integral_matrix_lol[h][h] = 1.0;

    // Initialize list of halfedges to process and record of seen faces
    std::queue<int> halfedges_to_process;
    halfedges_to_process.push(h);
    auto is_face_seen = std::vector<bool>(m.n_faces(), false);
    is_face_seen[m.f[h]] = true;


    // Process opposite halfedge if it is not cut
    if (cut_given && cut_h[h]) {
        int ho = m.opp[h];
        is_cut_h[h] = true;
        is_cut_h[ho] = true;
    } else {
        // Copy per corner values across the edge
        int ho = m.opp[h];
        integral_matrix_lol[ho] = integral_matrix_lol[m.n[m.n[h]]];
        integral_matrix_lol[m.n[m.n[ho]]] = integral_matrix_lol[h];

        // Update records
        is_face_seen[m.f[ho]] = true;
        halfedges_to_process.push(ho);
    }

    // Process one face at a time until all halfedges are seen
    while (!halfedges_to_process.empty()) {
        // Get triangle halfedges for the next halfedge to process
        h = halfedges_to_process.front();
        halfedges_to_process.pop();
        int hn = m.n[h];
        int hp = m.n[hn];

        // Integrate over hn to finish defining halfedges for the face
        integral_matrix_lol[hn] = integral_matrix_lol[h];
        integral_matrix_lol[hn][hn] += 1.0;

        // Process edges of the face,
        for (int hc : {hn, hp}) {
            int ho = m.opp[hc];

            // Skip edges adjacent to seen faces or that are cut
            if (is_face_seen[m.f[ho]] || (cut_given && cut_h[ho])) {
                is_cut_h[hc] = true;
                is_cut_h[ho] = true;
                continue;
            }

            // Copy per corner values across the edge
            integral_matrix_lol[ho] = integral_matrix_lol[m.n[m.n[hc]]];
            integral_matrix_lol[m.n[m.n[ho]]] = integral_matrix_lol[hc];

            // Update records
            is_face_seen[m.f[ho]] = true;
            halfedges_to_process.push(ho);
        }
    }

    // Copy list of lists to triplets
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        for (auto it : integral_matrix_lol[h]) {
            tripletList.push_back(T(h, it.first, it.second));
        }
    }

    // Create the matrix from the triplets
    MatrixX integral_matrix;
    integral_matrix.resize(num_halfedges, num_halfedges);
    integral_matrix.reserve(tripletList.size());
    integral_matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    return integral_matrix;
}

VectorX integrate_one_form(
    const Mesh<Scalar>& m,
    const VectorX& one_form,
    const std::vector<bool>& cut_h,
    std::vector<bool>& is_cut_h,
    int start_h)
{
    assert(is_closed_one_form(m, one_form));

    // Integrate one form using the integration matrix
    MatrixX integral_matrix = build_one_form_integral_matrix(m, cut_h, is_cut_h, start_h);
    VectorX integrated_one_form = integral_matrix * one_form;

    // TODO: Validate integration
    return integrated_one_form;
}

// TODO: Differentiation code sketch
//    // validation phi values
//    for (int i = 0; i < m.n_halfedges(); i++) {
//        Scalar u0 = phi[i];
//        Scalar u1 = phi[m.n[m.n[i]]];
//        Scalar _xi = u0 - u1;
//        if (abs(_xi - xi[i]) > 1e-12)
//            std::cerr << std::setprecision(17) << "error (" << _xi << ", " << xi[i]
//                      << "): " << abs(_xi - xi[i]) << std::endl;
//    }

MatrixX build_integrated_one_form_scaling_matrix(const Mesh<Scalar>& m)
{
    // Generate map from halfedges hjk to halfedges hij and hjk with tips at the base
    // and tip of hjk respectively since the integrated one form has data at halfedge tip
    // corners
    int num_halfedges = m.n_halfedges();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(2 * num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        int hjk = m.n[hij];
        tripletList.push_back(T(hjk, hij, 1.0));
        tripletList.push_back(T(hjk, hjk, 1.0));
    }

    // Create the matrix from the triplets
    MatrixX scaling_matrix;
    scaling_matrix.resize(num_halfedges, num_halfedges);
    scaling_matrix.reserve(tripletList.size());
    scaling_matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    return scaling_matrix;
}

VectorX scale_halfedges_by_integrated_one_form(
    const Mesh<Scalar>& m,
    const VectorX& metric_coords,
    const VectorX& integrated_one_form)
{
    int num_halfedges = metric_coords.size();
    VectorX scaled_metric_coords(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // Scale NEXT halfedge hjk (as we can access the adjacent corner values from the tips of hij
        // and hjk)
        int hjk = m.n[hij];
        scaled_metric_coords[hjk] =
            metric_coords[hjk] + (integrated_one_form[hij] + integrated_one_form[hjk]);
    }

    return scaled_metric_coords;
}

VectorX scale_edges_by_zero_form(
    const Mesh<Scalar>& m,
    const VectorX& metric_coords,
    const VectorX& zero_form)
{
    int num_halfedges = metric_coords.size();
    VectorX scaled_metric_coords(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // Get adjacent vertices
        int hji = m.opp[hij];
        int vi = m.v_rep[m.to[hji]];
        int vj = m.v_rep[m.to[hij]];

        // Scale halfedge
        scaled_metric_coords[hij] = metric_coords[hij] + (zero_form[vi] + zero_form[vj]);
    }

    return scaled_metric_coords;
}


} // namespace Holonomy
} // namespace Penner