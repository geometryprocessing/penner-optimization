#pragma once

#include "util/common.h"

#include <iomanip>

namespace Penner {

/**
 * @brief Compute the max of a std vector.
 *
 * @param[in] v: vector
 * @return max of v
 */
Scalar vector_max(const std::vector<Scalar>& v);

/**
 * @brief Negate every value of a vector of scalars
 *
 * @param[in] v: vector
 * @return negation of v
 */
std::vector<Scalar> vector_negate(const std::vector<Scalar>& v);

template <typename VectorType>
int argmax(const VectorType& v)
{
    int size = v.size();
    if (size == 0) return -1;
    int max_index = 0;
    for (int i = 1; i < size; ++i)
    {
        if (v[i] > v[max_index])
        {
            max_index = i;
        }
    }

    return max_index;
}

template <typename VectorType>
int argmin(const VectorType& v)
{
    int size = v.size();
    if (size == 0) return -1;
    int min_index = 0;
    for (int i = 1; i < size; ++i)
    {
        if (v[i] < v[min_index])
        {
            min_index = i;
        }
    }

    return min_index;
}

/**
 * @brief Determine if a vector contains a NaN
 * 
 * @param v: vector to check
 * @return true if the vector contains a NaN
 * @return false otherwise
 */
bool vector_contains_nan(const VectorX& v);

/**
 * @brief Compose two vectors
 *
 * @param[in] v: first vector
 * @param[in] w: second vector
 * @return composition i -> v[w[i]]
 */
template <typename VectorTypeOuter, typename VectorTypeInner>
VectorTypeOuter vector_compose(const VectorTypeOuter& v, const VectorTypeInner& w)
{
    int domain_size = w.size();
    VectorTypeOuter composition(domain_size);
    for (int i = 0; i < domain_size; ++i) {
        composition[i] = v[w[i]];
    }

    return composition;
}

/**
 * @brief Permute a vector by a permutation
 *
 * @param[in] v: vector to reindex
 * @param[in] reindex: permutation
 * @return composition i -> v[reindex[i]]
 */
template <typename VectorType, typename IndexVectorType>
VectorType vector_reindex(const VectorType& v, const IndexVectorType& reindex)
{
    int domain_size = reindex.size();
    VectorType composition(domain_size);
    for (int i = 0; i < domain_size; ++i) {
        composition[i] = v[reindex[i]];
    }

    return composition;
}

/**
 * @brief Permute a vector by the inverse of a permutation
 *
 * @param[in] v: vector to reindex
 * @param[in] reindex: permutation to invert
 * @return composition i -> v[reindex[i]]
 */
template <typename VectorType, typename IndexVectorType>
VectorType vector_inverse_reindex(const VectorType& v, const IndexVectorType& reindex)
{
    int domain_size = reindex.size();
    VectorType composition(domain_size);
    for (int i = 0; i < domain_size; ++i) {
        composition[reindex[i]] = v[i];
    }

    return composition;
}

/**
 * @brief Compute the range n of a map f: [0,...,m-1] -> [0,...,n-1]
 * 
 * @param map: integer index map
 * @return range of the map
 */
int compute_map_range(const std::vector<int>& map);

/**
 * @brief Invert a map (using a left inverse for noninvertible maps)
 * 
 * @param map: map to invert
 * @return left inverse of the map
 */
std::vector<int> invert_map(const std::vector<int>& map);

/**
 * @brief Generate a random permutation
 * 
 * @param n: size of the permutation
 * @return permutation vector
 */
std::vector<int> generate_permutation(int n);

/**
 * @brief Shuffle the image indices of a map
 * 
 * @param map: integer index map
 * @return shuffled index map
 */
std::vector<int> shuffle_map_image(const std::vector<int>& map);

/**
 * @brief Union a collection of n maps m_i: X_i -> Y_i into a single map m: X -> Y.
 *
 * We concatenate X = [X_1,...,X_n] and Y = [Y_1,...,Y_n], and we define m so that it maps
 * X_i to Y_i as m_i with appropriate index offsets in both the domain and range.
 *
 * @tparam type of the vector used for the map
 * @param maps: n maps to union
 * @param range_sizes: n sizes of Y_1,...,Y_n
 * @return union map
 */
template <typename VectorType>
VectorType union_maps(const std::vector<VectorType>& maps, const std::vector<int>& range_sizes)
{
    // Precompute the total map size
    int total_domain_size = 0;
    for (const auto& map : maps) {
        total_domain_size += map.size();
    }
    VectorType total_map(total_domain_size);

    // combine maps
    int num_maps = maps.size();
    int count = 0;
    int offset = 0;
    for (int i = 0; i < num_maps; ++i) {
        const auto& map = maps[i];

        // Add reindexed map entries to total map
        for (const auto& map_val : map) {
            total_map[count] = map_val + offset;
            count++;
        }

        // Increase offset
        offset += range_sizes[i];
    }

    return total_map;
}

/**
 * @brief Union a collection of n vectors m_i: X_i -> F into a single vector m: X -> F.
 *
 * We concatenate X = [X_1,...,X_n] and take F as a fixed value field, and we define m so that it
 * maps X_i to F as m_i with appropriate index offsets in the domain.
 *
 * @tparam type of the vector used for the map
 * @param vectors: n vectors to union
 * @return union vector
 */
template <typename VectorType>
VectorType union_vectors(const std::vector<VectorType>& vectors)
{
    // Precompute the total attribute domain size
    int total_domain_size = 0;
    for (const auto& vector : vectors) {
        total_domain_size += vector.size();
    }
    VectorType total_vector(total_domain_size);

    int count = 0;
    for (const auto& vector : vectors) {
        int vector_size = vector.size();
        total_vector.segment(count, vector_size) = vector;
        count += vector_size;
    }

    return total_vector;
}

/**
 * @brief Create a map from a set of a given size to a subset, with -1 for entries not in the subset.
 * 
 * @param set_size: size of the ambient set
 * @param subset_indices: indices of the subset in the set
 * @return map from set indices to subset indices
 */
std::vector<int> index_subset(size_t set_size, const std::vector<int>& subset_indices);

/**
 * @brief Check if a map is invariant under some permutation and thus descends to a well-defined
 * function on the orbits of the permutation
 *
 * @param map: map from {0,...,n-1} to {0,...,m-1}
 * @param perm: permutation of n elements
 * @return true iff the map is invariant under perm
 */
bool is_invariant_under_permutation(const std::vector<int>& map, const std::vector<int>& perm);

/**
 * @brief Check if two maps are one-sided inverses of each other, i.e., f(g(i)) = i
 *
 * The functions are allowed to have negative (denoting invalid) values; indices i where g(i) < 0
 * are skipped
 *
 * @param left_inverse: map f:{0,...,m-1}->Z
 * @param right_inverse: map g:{0,...,n-1}->{0,...,m-1}
 * @return true iff f composed with g is the identity where g is defined
 */
bool is_one_sided_inverse(
    const std::vector<int>& left_inverse,
    const std::vector<int>& right_inverse);

/**
 * @brief Check if the maps defining the edge connectivity of a polygonal mesh (next and prev)
 * are valid
 *
 * @param next: size #he vector, next halfedge id
 * @param prev: size #he vector, prev halfedge id
 * @return true iff the edge maps are valid
 */
bool are_polygon_mesh_edges_valid(const std::vector<int>& next, const std::vector<int>& prev);

/**
 * @brief Check if the maps defining the vertex connectivity of a polygonal mesh (to and out)
 * are valid
 *
 * @param prev: size #he vector, prev halfedge id
 * @param to: size #he vector, halfedge vertex tip id
 * @param out: size #v vector, arbitrary halfedge id outgoing from vertex
 * @return true iff the vertex maps are valid
 */
bool are_polygon_mesh_vertices_valid(
    const std::vector<int>& opp,
    const std::vector<int>& prev,
    const std::vector<int>& to,
    const std::vector<int>& out);

/**
 * @brief Check if the maps defining the face connectivity of a polygonal mesh (he2f and f2he)
 * are valid
 *
 * @param next: size #he vector, next halfedge id
 * @param he2f: size #he vector, face id adjacent to halfedge
 * @param f2he: size #f vector, arbitrary halfedge id adjacent to face
 * @return true iff the face maps are valid
 */
bool are_polygon_mesh_faces_valid(
    const std::vector<int>& next,
    const std::vector<int>& he2f,
    const std::vector<int>& f2he);

} // namespace Penner
