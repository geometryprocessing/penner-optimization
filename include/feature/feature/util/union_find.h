#pragma once

#include "feature/core/common.h"

namespace Penner {
namespace Feature {

/**
 * @brief Simple implementation of a union find data structure
 */
class UnionFind
{
public:
    /**
     * @brief Construct a new Union Find object of a given (fixed) size
     *
     * @param size: number of elements in the set
     */
    UnionFind(int size)
    {
        // Each element is originally a root
        arange(size, m_parent);
    }

    /**
     * @brief Determine if a given element is a root
     * 
     * @param index: index of an element
     * @return true iff the element is a root
     */
    bool is_root(int index) const { return (m_parent[index] == index); }

    /**
     * @brief Find the set root index of a given element
     * 
     * @param index: index of an element
     * @return root index for the element
     */
    int find_set(int index)
    {
        // Get root index of set
        int root = index;
        while (!is_root(root)) {
            root = m_parent[root];
        }

        // Overwrite existing parents for faster traversal later
        int write_index = index;
        while (m_parent[write_index] != root) {
            int parent_index = m_parent[write_index];
            m_parent[write_index] = parent_index;
            write_index = parent_index;
        }

        // Return root
        return root;
    }

    /**
     * @brief Union the sets containing two elements
     * 
     * @param first_index: index of the first element
     * @param second_index: index of the second element
     */
    void union_sets(int first_index, int second_index)
    {
        // Get set indices of two element indices
        int first_set = find_set(first_index);
        int second_set = find_set(second_index);

        // Do nothing if already same set
        if (first_set == second_set) return;

        // Union two sets arbitrarily
        // TODO Use better (size or rank) update rule
        m_parent[second_set] = first_set;
    }

    /**
     * @brief Count the total number of elements
     * 
     * @return number of elements
     */
    int count_elements() const { return m_parent.size(); }

    /**
     * @brief Count the total number of sets.
     * 
     * @return current number of sets
     */
    int count_sets()
    {
        // Count sets by counting roots
        int size = count_elements();
        int num_roots = 0;
        for (int i = 0; i < size; ++i) {
            if (is_root(i)) ++num_roots;
        }

        return num_roots;
    }

    /**
     * @brief Build labels for the set elements.
     * 
     * @return map from elements to set index
     */
    std::vector<int> index_sets()
    {
        // Count sets by counting roots
        int num_elements = count_elements();

        // Make map from roots to sets
        int count = 0;
        std::vector<int> set_index(num_elements, -1);
        for (int i = 0; i < num_elements; ++i) {
            if (is_root(i)) {
                set_index[i] = count;
                count++;
            }
        }

        // Index remaining elements
        for (int i = 0; i < num_elements; ++i) {
            set_index[i] = set_index[find_set(i)];
        }

        return set_index;
    }

    /**
     * @brief Build list of sets.
     * 
     * @return list of sets
     */
    std::vector<std::vector<int>> build_sets()
    {
        // build map from elements to sets
        std::vector<int> index = index_sets();

        // collect set elements
        int num_elements = count_elements();
        int num_sets = count_sets();
        std::vector<std::vector<int>> sets(num_sets, std::vector<int>({}));
        for (int i = 0; i < num_elements; ++i)
        {
            sets[index[i]].push_back(i);
        }

        return sets;
    }

private:
    std::vector<int> m_parent;
};

} // namespace Feature
} // namespace Penner