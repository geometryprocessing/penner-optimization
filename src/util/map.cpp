#include "util/map.h"

#include "util/vector.h"

#include <random>
#include <chrono>

namespace Penner {

std::vector<Scalar> vector_negate(const std::vector<Scalar>& v)
{
    int n = v.size();
    std::vector<Scalar> w(n);
    for (int i = 0; i < n; ++i) {
        w[i] = -v[i];
    }

    return w;
}

bool vector_contains_nan(const VectorX& v)
{
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (isnan(v(i))) return true;
    }

    return false;
}

std::vector<int> index_subset(size_t set_size, const std::vector<int>& subset_indices)
{
    std::vector<int> set_to_subset_mapping;
    Penner::compute_set_to_subset_mapping(set_size, subset_indices, set_to_subset_mapping);
    return set_to_subset_mapping;
}

int compute_map_range(const std::vector<int>& map)
{
    // get range of map
    int domain = map.size();
    int range = 0;
    for (int i = 0; i < domain; ++i) {
        if (range < (map[i] + 1)) {
            range = map[i] + 1;
        }
    }

    return range;
}

std::vector<int> invert_map(const std::vector<int>& map)
{
    // get range of map
    int domain = map.size();
    int range = compute_map_range(map);

    // invert map
    std::vector<int> inverse_map(range, -1);
    for (int i = 0; i < domain; ++i) {
        inverse_map[map[i]] = i;
    }

    return inverse_map;
}

std::vector<int> generate_permutation(int n, bool use_random_seed)
{
    // generate permuation for the given size
    std::vector<int> permutation;
    Penner::arange(n, permutation);
    if (use_random_seed)
    {
        auto rng = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
        std::shuffle(permutation.begin(), permutation.end(), rng);
    }
    else
    {
        auto rng = std::default_random_engine{};
        std::shuffle(permutation.begin(), permutation.end(), rng);
    }
    return permutation;
}

std::vector<int> shuffle_map_image(const std::vector<int>& map)
{
    // generate permuation for the map range
    int range = compute_map_range(map);
    std::vector<int> permutation = generate_permutation(range);

    // compute the map with the permutation
    return vector_compose(permutation, map);
}

bool is_invariant_under_permutation(const std::vector<int>& map, const std::vector<int>& perm)
{
    assert(map.size() == perm.size());

    // check if applying the permutation changes the image of any element under the map
    int n = map.size();
    for (int i = 0; i < n; ++i) {
        if (perm[i] < 0) return false;
        if (perm[i] >= n) return false;
        if (map[perm[i]] != map[i]) return false;
    }

    return true;
}


bool is_one_sided_inverse(
    const std::vector<int>& left_inverse,
    const std::vector<int>& right_inverse)
{
    long n = right_inverse.size();
    long m = left_inverse.size();
    for (long i = 0; i < n; ++i) {
        // Ignore negative indices
        if (right_inverse[i] < 0) {
            continue;
        }
        if ((right_inverse[i] >= m) || (left_inverse[right_inverse[i]] != i)) {
            return false;
        }
    }
    return true;
}

bool are_polygon_mesh_edges_valid(const std::vector<int>& next, const std::vector<int>& prev)
{
    if (next.size() != prev.size()) {
        spdlog::warn("next and prev are not inverse");
        return false;
    }

    // prev is a right and left inverse for next
    if ((!is_one_sided_inverse(next, prev)) || (!is_one_sided_inverse(prev, next))) {
        spdlog::warn("next and prev are not inverse");
        return false;
    }

    return true;
}


bool are_polygon_mesh_vertices_valid(
    const std::vector<int>& opp,
    const std::vector<int>& prev,
    const std::vector<int>& to,
    const std::vector<int>& out)
{
    if (prev.size() != to.size()) {
        return false;
    }
    long n_halfedges = to.size();

    // Generate per halfedge vertex circulation and from maps
    std::vector<int> circ(n_halfedges);
    std::vector<int> from(n_halfedges);
    for (long hi = 0; hi < n_halfedges; ++hi) {
        circ[hi] = prev[opp[hi]];
        from[hi] = to[opp[hi]];
    }

    // Build vertices from vertex circulation
    std::vector<std::vector<int>> vert;
    build_orbits(circ, vert);

    // Number of vertices in out match the number of orbits
    if (out.size() != vert.size()) {
        spdlog::warn("out does not have the right number of vertices");
        return false;
    }

    // to is invariant under circulation
    if (!is_invariant_under_permutation(to, circ)) {
        spdlog::warn("to is not invariant under vertex circulation");
        return false;
    }

    // out is a right inverse for from
    if (!is_one_sided_inverse(from, out)) {
        spdlog::warn("out is not a right inverse for from");
        return false;
    }

    return true;
}

bool are_polygon_mesh_faces_valid(
    const std::vector<int>& next,
    const std::vector<int>& he2f,
    const std::vector<int>& f2he)
{
    if (next.size() != he2f.size()) {
        return false;
    }

    // Build faces from next map
    std::vector<std::vector<int>> faces;
    build_orbits(next, faces);

    // Number of faces in f2he match the number of orbits
    if (f2he.size() != faces.size()) {
        spdlog::warn("f2he does not have the right number of faces");
        return false;
    }

    // he2f is invariant under next
    if (!is_invariant_under_permutation(he2f, next)) {
        spdlog::warn("he2f is not invariant under next");
        return false;
    }

    // f2he is a right inverse for he2f
    if (!is_one_sided_inverse(he2f, f2he)) {
        spdlog::warn("f2he is not a right inverse for he2f");
        return false;
    }

    return true;
}


} // namespace Penner
