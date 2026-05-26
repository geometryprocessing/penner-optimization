#include "holonomy/core/dual_lengths.h"

#include "optimization/core/constraint.h"

#include <queue>

namespace Penner {
namespace Holonomy {

std::vector<Scalar> compute_dual_edge_lengths(const Mesh<Scalar>& m)
{
    int num_halfedges = m.n_halfedges();
    std::vector<Scalar> dual_edge_lengths(num_halfedges);

    // Compute cotangent angles
    VectorX he2angle, he2cot;
    Optimization::corner_angles(m, he2angle, he2cot);

    for (int hij = 0; hij < num_halfedges; ++hij) {
        // Only process halfedge with lower index
        if (hij > m.opp[hij]) continue;

        // Get the average of cotangents opposite the edge
        int hji = m.opp[hij];
        Scalar ratio = 0.5 * (he2cot[hij] + he2cot[hji]);

        // Set the dual edge length so that its ratio with the primal length is the average of
        // cotangents
        dual_edge_lengths[hij] = dual_edge_lengths[hji] = abs(ratio * m.l[hij]);
    }

    return dual_edge_lengths;
}

// Compute the length of the path from a face to a root in a dual tree (or forest)
Scalar compute_dual_path_distance_to_root(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    const DualTree& dual_tree,
    const std::vector<int>& e2he,
    int face_index)
{
    // Double distance is infinite
    if (m.type[m.h[face_index]] == 2) {
        return INF;
    }

    // Trace path to root and compute length
    Scalar dual_loop_length = 0.0;
    int curr_face_index = face_index;
    while (!dual_tree.is_root(curr_face_index)) {
        int edge_index = dual_tree.out(curr_face_index);
        curr_face_index = dual_tree.to(edge_index);
        dual_loop_length += weights[e2he[dual_tree.edge(edge_index)]];
    }

    return dual_loop_length;
}

// Check that dual path distances to roots are valid
bool is_valid_dual_path_distance_to_root(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    const DualTree& dual_tree,
    const std::vector<Scalar>& distances)
{
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(m, he2e, e2he);

    // Check each distance against direct computation
    int num_faces = m.n_faces();
    for (int fi = 0; fi < num_faces; ++fi) {
        Scalar distance_to_root =
            compute_dual_path_distance_to_root(m, weights, dual_tree, e2he, fi);
        if (!float_equal<Scalar>(distances[fi], distance_to_root, max(1e-10 * distance_to_root, 1e-10))) {
            spdlog::error(
                "computed distance {} and actual distance {} for {} differ",
                distances[fi],
                distance_to_root,
                fi);
            return false;
        }
    }

    return true;
}

// Compute the distances from dual vertices to the root of a dual spanning tree (or forest)
std::vector<Scalar> compute_dual_path_distances_to_root(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    const DualTree& dual_tree)
{
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(m, he2e, e2he);

    // Initialize face map with 0 distances
    int num_faces = m.n_faces();
    std::vector<Scalar> distances(num_faces, -1.0);

    // Set a priori distances
    for (int fi = 0; fi < num_faces; ++fi) {
        // Double distance is infinite
        if (m.type[m.h[fi]] == 2) {
            distances[fi] = INF;
            continue;
        }

        // Distance to a root is 0
        if (dual_tree.is_root(fi)) {
            distances[fi] = 0.0;
        }
    }

    // Assign length for each remaining faces iteratively
    for (int fi = 0; fi < num_faces; ++fi) {
        if (distances[fi] >= 0.0) continue;

        // Trace path to first known distance (root in base case)
        int curr_face_index = fi;
        std::vector<int> dual_path = {};
        while (distances[curr_face_index] < 0.0) {
            dual_path.push_back(curr_face_index);
            int edge_index = dual_tree.out(curr_face_index);
            curr_face_index = dual_tree.to(edge_index);
        }

        // Update lengths along reverse path
        for (auto itr = dual_path.rbegin(); itr != dual_path.rend(); ++itr) {
            int fj = *itr;
            int edge_index = dual_tree.out(fj);
            distances[fj] =
                distances[dual_tree.to(edge_index)] + weights[e2he[dual_tree.edge(edge_index)]];
        }
    }

    assert(is_valid_dual_path_distance_to_root(m, weights, dual_tree, distances));

    return distances;
}

std::vector<Scalar> compute_dual_loop_length_weights(
    const Mesh<Scalar>& m,
    const std::vector<Scalar>& weights,
    const DualTree& dual_tree)
{
    // Get edge maps
    std::vector<int> he2e;
    std::vector<int> e2he;
    build_edge_maps(m, he2e, e2he);

    // Initialize dual loop lengths to 0
    int num_halfedges = m.n_halfedges();
    std::vector<Scalar> dual_loop_lengths(num_halfedges, 0.0);

    // Precompute distance from faces to root
    std::vector<Scalar> distances = compute_dual_path_distances_to_root(m, weights, dual_tree);

    // Compute triangle areas
    for (int hij = 0; hij < num_halfedges; ++hij) {
        // Only process lower index halfedge in edge
        if (hij > m.opp[hij]) continue;

        // Skip edges in dual tree (default to zero)
        if (dual_tree.is_edge_in_tree(he2e[hij])) continue;

        // Compute length of homotopy cycle path from adding the edge to the dual tree
        dual_loop_lengths[hij] = distances[m.f[hij]] + distances[m.f[m.opp[hij]]] + weights[hij];
        dual_loop_lengths[m.opp[hij]] = dual_loop_lengths[hij];
    }

    return dual_loop_lengths;
}

} // namespace Holonomy
} // namespace Penner
