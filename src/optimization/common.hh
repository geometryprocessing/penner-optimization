#pragma once

#include "globals.hh"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <filesystem>

#include <igl/facet_components.h>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/ostream_sink.h"

namespace CurvatureMetric {

/// *************************
/// Floating Point Operations
/// *************************

/// Swap two doubles.
///
/// @param[in, out] a: first double to swap
/// @param[in, out] b: second double to swap
inline void
swap(
  double& a,
  double& b
) {
  std::swap<double>(a, b);
}

/// Get the max of two doubles.
///
/// @param[in] a: first double to max
/// @param[in] b: second double to max
/// @return max of a and b
inline double
max(
  const double& a,
  const double& b
) {
  return std::max(a, b);
}

/// Check if two values are equal, up to a tolerance.
///
/// @param[in] a: first value to compare
/// @param[in] b: second value to compare
/// @param[in] eps: tolerance for equality
/// @return true iff |a - b| < eps
inline bool
float_equal(Scalar a, Scalar b, Scalar eps = 1e-10)
{
  return (abs(a - b) < eps);
}

// Check if two vectors are component-wise equal, up to a tolerance.
//
/// @param[in] v: first vector to compare
/// @param[in] w: second vector to compare
/// @param[in] eps: tolerance for equality
/// @return true iff ||v - i||_inf < eps
inline bool
vector_equal(VectorX v, VectorX w, Scalar eps = 1e-10)
{
  // Check if the sizes are the same
  if (v.size() != w.size()) return false;

  // Check per element equality
  for (Eigen::Index i = 0; i < v.size(); ++i)
  {
    if (!float_equal(v[i], w[i], eps)) return false;
  }

  // Equal otherwise
  return true;
}

/// Check if a matrix contains a nan
///
/// @param[in] mat: matrix to check
/// @return true iff mat contains a nan
inline bool
matrix_contains_nan(const Eigen::MatrixXd& mat)
{
  for (Eigen::Index i = 0; i < mat.rows(); ++i) {
    for (Eigen::Index j = 0; j < mat.cols(); ++j) {
      if (std::isnan(mat(i, j)))
        return true;
    }
  }

  return false;
}

/// ************
/// Input/Output
/// ************

/// Join two filepaths.
///
/// @param[in] first_path: first path to join
/// @param[in] second_path: second path to join
/// @return combined path
inline std::filesystem::path
join_path(const std::filesystem::path& first_path,
          const std::filesystem::path& second_path)
{
  return first_path / second_path;
}

/// Read a vector from a file.
///
/// @param[in] filename: file with vector to read
/// @param[out] vec: vector from file
template <typename T>
void read_vector_from_file(
  const std::string &filename,
  std::vector<T> &vec
) {
  vec.clear();

  // Open file
  std::ifstream input_file(filename);
  if (!input_file) return;

  // Read file
  std::string line;
  while (std::getline(input_file, line))
  {
    std::istringstream iss(line);
    T value;
    iss >> value;
    vec.push_back(value);
  }

  // Close file
  input_file.close();
}

/// Read a vector of pairs from a file.
///
/// @param[in] filename: file with vector to read
/// @param[out] vec: vector from file
template <typename T>
void read_vector_of_pairs_from_file(
  const std::string &filename,
  std::vector<std::pair<T, T>> &vec
) {
  vec.clear();

  // Open file
  std::ifstream input_file(filename);
  if (!input_file) return;

  // Read file
  std::string line;
  while (std::getline(input_file, line))
  {
    std::istringstream iss(line);
    T first_value, second_value;
    iss >> first_value >> second_value;
    vec.push_back(std::make_pair(first_value, second_value));
  }

  // Close file
  input_file.close();
}

/// Format a vector as a human readable string
///
/// @param[in] vec: vector to format
/// @param[in] delim: deliminator between vector entries
/// @return formatted vector
template<typename T>
inline std::string formatted_vector(
  const std::vector<T> &vec,
  std::string delim=" "
) {
  std::stringstream vector_string;
  for (size_t i = 0; i < vec.size(); ++i)
  {
    vector_string << vec[i] << delim;
  }

  return vector_string.str();
}

/// Write a matrix to file.
///
/// @param[in] matrix: matrix to serialize
/// @param[in] filename: file to write the matrix to
inline void
write_matrix(
  const Eigen::MatrixXd& matrix,
  const std::string& filename
) {
  if (matrix.cols() == 0)
  {
    return;
  }

  // Open file
  std::ofstream output_file;
  output_file.open(filename);

  // Iterate over rows
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    // Iterate over columns of row i
    Scalar v = matrix(i, 0);
    output_file << std::fixed << std::setprecision(17) << v;
    for (Eigen::Index j = 1; j < matrix.cols(); ++j) {
      Scalar v = matrix(i, j);
      output_file << std::fixed << std::setprecision(17) << "," << v;
    }

    // Add newline to end of row
    output_file << std::endl;
  }

  // Close file
  output_file.close();
}

/// Write a sparse matrix to file in i,j,v format.
///
/// @param[in] matrix: matrix to serialize
/// @param[in] filename: file to write the matrix to
/// @param[in] format: format (csv or matlab) to write the matrix in
inline void
write_sparse_matrix(
  const MatrixX& matrix,
  const std::string& filename,
  std::string format="csv"
) {
  /// Open file
  std::ofstream output_file;
  output_file.open(filename);

  // Iterate over nonzero entries
  for (Eigen::Index k = 0; k < matrix.outerSize(); ++k) {
    for (MatrixX::InnerIterator it(matrix, k); it; ++it) {
      int i = it.row();
      int j = it.col();
      Scalar v = it.value();

      // CSV format has comma separated 0-indexed values
      if (format == "csv")
      {
        output_file << std::fixed << std::setprecision(17)
          << i << "," << j << "," << v << std::endl;
      }
      // MATLAB uses space separated 1-indexed values
      else if (format == "matlab")
      {
        output_file << std::fixed << std::setprecision(17)
          << (i + 1) << "  " << (j + 1) << "  " << v << std::endl;
      }
    }
  }

  // Close file
  output_file.close();
}

/// **************
/// Linear Algebra
/// **************

// Compute the sup norm of a vector.
//
/// @param[in] v: vector
/// @return sup norm of v
inline Scalar
sup_norm(const VectorX& v)
{
  Scalar norm_value = 0.0;
  for (Eigen::Index i = 0; i < v.size(); ++i)
  {
    norm_value = max(norm_value, abs(v[i]));
  }

  return norm_value;
}

// Compute the sup norm of a matrix.
//
/// @param[in] matrix: matrix
/// @return sup norm of the matrix
inline Scalar
matrix_sup_norm(
  const MatrixX& matrix
) {
  // Check for trivial matrices
  if (matrix.size() == 0) return 0;

  // Iterate to determine maximum abs value
  Scalar max_value = 0.0;
  for (Eigen::Index k = 0; k < matrix.outerSize(); ++k) {
    for (MatrixX::InnerIterator it(matrix, k); it; ++it) {
      max_value = std::max(max_value, abs(it.value()));
    }
  }

  return max_value;
}


/// Compute the Kronecker product of two vectors.
///
/// @param[in] vec_1: first vector to product
/// @param[in] vec_2: second vector to product
/// @return product of vec_1 and vec_2
inline VectorX
kronecker_product(const VectorX& vec_1,
                  const VectorX& vec_2)
{
  if (vec_1.size() != vec_2.size())
  {
    spdlog::error("Cannot multiply two vectors of different sizes");
    return VectorX();
  }

  // Build product component-wise
  size_t vec_size = vec_1.size();
  VectorX product(vec_size);
  for (size_t i = 0; i < vec_size; ++i)
  {
    product[i] = vec_1[i] * vec_2[i];
  }

  return product;
}

/// Create an identity sparse matrix of dimension nxn
///
/// @param[in] n: matrix dimension
/// @return nxn identity matrix
inline MatrixX
id_matrix(int n)
{
  // Build triplet lists for the identity
  std::vector<T> tripletList;
  tripletList.reserve(n);
  for (int i = 0; i < n; ++i) {
    tripletList.push_back(T(i, i, 1.0));
  }

  // Create the matrix from the triplets
  MatrixX id(n, n);
  id.reserve(tripletList.size());
  id.setFromTriplets(tripletList.begin(), tripletList.end());

  return id;
}

/// Create an empty IJV representation of a matrix with allocated space.
///
/// @param capacity: space to reserve in the IJV arrays
/// @return IJV representation of a matrix with reserved space
inline std::tuple<
  std::vector<int>,
  std::vector<int>,
  std::vector<Scalar>>
allocate_triplet_matrix(int capacity)
{
  std::vector<int> I;
  std::vector<int> J;
  std::vector<Scalar> V;
  I.reserve(capacity);
  J.reserve(capacity);
  V.reserve(capacity);

  return std::make_tuple(I, J, V);
}

/// Compute the condition number of a matrix.
///
/// Note that this is a slow and unstable operation for large matrices
///
/// @param[in] matrix: input matrix
/// @return condition number of the matrix
inline Scalar
compute_condition_number(
  const Eigen::MatrixXd matrix
) {
  // Check for square matrix
  if (matrix.rows() != matrix.cols()) return 0.0;
  
  // Compute condition number with singular values
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
  Scalar sigma_1 = svd.singularValues()(0);
  Scalar sigma_n = svd.singularValues()(svd.singularValues().size()-1);
  spdlog::trace("Min and max singular values are {} and {}", sigma_n, sigma_1);
  
  return sigma_1 / sigma_n;
}

VectorX solve_psd_system(const MatrixX& A, const VectorX&b);

/// ********************
/// Data Type Conversion
/// ********************

/// Convert standard template library vector to an Eigen vector.
///
/// @param[in] vector_std: standard template library vector to copy
/// @param[out] vector_eigen: copied vector
template <typename VectorScalar, typename MatrixScalar>
void
convert_std_to_eigen_vector(const std::vector<VectorScalar>& vector_std,
                            Eigen::Matrix<MatrixScalar, Eigen::Dynamic, 1>& vector_eigen)
{
  size_t vector_size = vector_std.size();
  vector_eigen.resize(vector_size);
  for (size_t i = 0; i < vector_size; ++i) {
    vector_eigen[i] = MatrixScalar(vector_std[i]);
  }
}

/// Convert standard template library vector of vectors to an Eigen matrix.
///
/// @param[in] matrix_vec: vector matrix to copy
/// @param[out] matrix: copied matrix 
template <typename VectorScalar, typename MatrixScalar>
void
convert_std_to_eigen_matrix(
  const std::vector<std::vector<VectorScalar>>& matrix_vec,
  Eigen::Matrix<MatrixScalar, Eigen::Dynamic, Eigen::Dynamic>& matrix
) {
  matrix.setZero(0, 0);
  if (matrix_vec.empty()) return;

  // Get dimensions of matrix
  int rows = matrix_vec.size();
  int cols = matrix_vec[0].size();
	matrix.resize(rows, cols);
  
  // Copy matrix by row
	for (int i = 0; i < rows; ++i)
	{
    // Check size validity
    if (static_cast<int>(matrix_vec[i].size()) != cols)
    {
      spdlog::error("Cannot copy vector of vectors of inconsistent sizes to a matrix");
      matrix.setZero(0, 0);
      return;
    }

    // Copy row
		for (int j = 0; j < cols; ++j)
		{
			matrix(i, j) = MatrixScalar(matrix_vec[i][j]);
		}
	}
}

/// Convert Eigen dense vector to sparse.
///
/// @param[in] vector_dense: input dense vector
/// @param[out] vector_sparse: output sparse vector
inline void
convert_dense_vector_to_sparse(
  const VectorX& vector_dense,
  MatrixX& vector_sparse
) {
  // Copy the vector to a triplet list
  int vec_size = vector_dense.size();
  typedef Eigen::Triplet<Scalar> T;
  std::vector<T> triplet_list;
  triplet_list.reserve(vec_size);
  for (int i = 0; i < vec_size; ++i)
  {
    triplet_list.push_back(T(i, 0, vector_dense[i]));
  }

  // Build the sparse vector from the triplet list
  vector_sparse.resize(vec_size, 1);
  vector_sparse.reserve(triplet_list.size());
  vector_sparse.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

/// Convert Eigen vector to standard template library vector.
///
/// @param[in] vector_eigen: Eigen vector to copy
/// @param[out] vector_std: copied vector
inline void
convert_eigen_to_std_vector(const VectorX& vector_eigen,
                            std::vector<Scalar>& vector_std)
{
  size_t vector_size = vector_eigen.size();
  vector_std.resize(vector_size);
  for (size_t i = 0; i < vector_size; ++i) {
    vector_std[i] = vector_eigen[i];
  }
}

/// Convert vector of scalars to doubles
inline Eigen::Matrix<double, Eigen::Dynamic, 1>
convert_scalar_to_double_vector(const VectorX& vector_scalar)
{
  int num_entries = vector_scalar.size();
  Eigen::Matrix<double, Eigen::Dynamic, 1> vector_double(num_entries);
  for (int i = 0; i < num_entries; ++i)
  {
    vector_double[i] = (double) (vector_scalar[i]);
  }

  return vector_double;
}

/// Convert vector of scalars to doubles
inline std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>>
convert_scalar_to_double_vector(const std::vector<VectorX>& vector_scalar)
{
  int num_entries = vector_scalar.size();
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> vector_double(num_entries);
  for (int i = 0; i < num_entries; ++i)
  {
    vector_double[i] = convert_scalar_to_double_vector(vector_scalar[i]);
  }

  return vector_double;
}

/// *****************
/// Vector Generation
/// *****************

/// Fill a vector with some value.
///
/// @param[in] size: size of the output vector
/// @param[in] value: fill value
/// @param[out] vec: vector to fill
template<typename T>
void
fill_vector(size_t size, T value, std::vector<T>& vec)
{
  vec.resize(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = value;
  }
}

/// Create a vector with values 0,1,...,n-1
///
/// @param[in] n: size of the output vector
/// @param[out] vec: output arangement vector
inline void
arange(size_t n, std::vector<int>& vec)
{
  vec.resize(n);
  for (size_t i = 0; i < n; ++i) {
    vec[i] = i;
  }
}

/// *******************
/// Subset Manipulation
/// *******************

/// Given a universe set and a list of subset indices, mask the corresponding
/// subset entries with 0.
///
/// @param[in] mask: list of indices to mask
/// @param[in, out] set: object to mask
inline void
mask_subset(const std::vector<int>& mask, VectorX& set)
{
  for (size_t i = 0; i < mask.size(); ++i) {
    set[mask[i]] = 0;
  }
}

/// Given a universe set and a list of subset indices, compute the corresponding
/// subset.
///
/// @param[in] universe: full set
/// @param[in] subset_indices: indices of the subset with respect to the
/// universe
/// @param[out] subset: subset corresponding to the indices
template<typename T>
void
compute_subset(const std::vector<T>& universe,
               const std::vector<int>& subset_indices,
               std::vector<T>& subset)
{
  size_t subset_size = subset_indices.size();
  subset.resize(subset_size);
  for (size_t i = 0; i < subset_size; ++i) {
    subset[i] = universe[subset_indices[i]];
  }
}

/// Given a universe set and a list of subset indices, compute the corresponding
/// subset.
///
/// @param[in] universe: full set
/// @param[in] subset_indices: indices of the subset with respect to the
/// universe
/// @param[out] subset: subset corresponding to the indices
inline void
compute_subset(const VectorX& universe,
               const std::vector<int>& subset_indices,
               VectorX& subset)
{
  subset.resize(subset_indices.size());
  for (size_t i = 0; i < subset_indices.size(); ++i)
  {
    subset[i] = universe[subset_indices[i]];
  }
}

/// Given a universe set size and a list of subset indices, compute the inverse
/// mapping from original subset indices to their location in the new subset or
/// -1 if the item has been removed
///
/// @param[in] set_size: size of the full set
/// @param[in] subset_indices: indices of the subset with respect to the
/// universe
/// @param[out] set_to_subset_mapping: subset corresponding to the indices
inline void
compute_set_to_subset_mapping(size_t set_size,
                              const std::vector<int>& subset_indices,
                              std::vector<int>& set_to_subset_mapping)
{
  size_t subset_size = subset_indices.size();
  fill_vector(set_size, -1, set_to_subset_mapping);
  for (size_t i = 0; i < subset_size; ++i) {
    set_to_subset_mapping[subset_indices[i]] = i;
  }
}

/// Write the subset with given indices into the universe.
///
/// @param[in] subset: subset corresponding to the indices
/// @param[in] subset_indices: indices of the subset with respect to the
/// universe
/// @param[in, out] set: full set to overwrite
inline void
write_subset(const VectorX& subset,
             const std::vector<int>& subset_indices,
             VectorX& set)
{
  size_t subset_size = subset.size();
  assert(subset_indices.size() == subset_size);
  for (size_t i = 0; i < subset_size; ++i) {
    set[subset_indices[i]] = subset[i];
  }
}

/// Given a matrix and lists of row and column indices, compute the
/// corresponding submatrix
///
/// @param[in] matrix: full matrix
/// @param[in] row_indices: indices of the rows to keep
/// @param[in] col_indices: indices of the cols to keep
/// @param[out] submatrix: corresponding submatrix
inline void
compute_submatrix(const MatrixX& matrix,
                  const std::vector<int>& row_indices,
                  const std::vector<int>& col_indices,
                  MatrixX& submatrix)
{
  size_t num_rows = matrix.rows();
  size_t num_cols = matrix.cols();
  size_t num_subrows = row_indices.size();
  size_t num_subcols = col_indices.size();
  assert(num_rows >= num_subrows);
  assert(num_cols >= num_subcols);

  // Get mappings from rows and columns to submatrix rows and columns
  std::vector<int> row_indexing_map, col_indexing_map;
  compute_set_to_subset_mapping(num_rows, row_indices, row_indexing_map);
  compute_set_to_subset_mapping(num_cols, col_indices, col_indexing_map);

  // To compute the sparse matrix subset, we get iterate over nonzero entries
  // and prune those that are not in the given indexing set while remapping kept
  // indices to their index in the subset
  typedef Eigen::Triplet<Scalar> T;
  std::vector<T> triplet_list;
  triplet_list.reserve(matrix.nonZeros());
  for (Eigen::Index k = 0; k < matrix.outerSize(); ++k) {
    for (MatrixX::InnerIterator it(matrix, k); it; ++it) {
      // Check if entry is in the submatrix
      int submatrix_row = row_indexing_map[it.row()];
      int submatrix_col = col_indexing_map[it.col()];
      if (submatrix_row < 0)
        continue;
      if (submatrix_col < 0)
        continue;

      // Add reindexed entry to the triplet list
      triplet_list.push_back(T(submatrix_row, submatrix_col, it.value()));
    }
  }
  submatrix.resize(num_subrows, num_subcols);
  submatrix.reserve(triplet_list.size());
  submatrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

/// *************
/// Index Vectors
/// *************

/// @brief From a vector of the indices, build a boolean array marking these indices as true
///
/// @param[in] index_vector: list of indices
/// @param[in] num_indices: total number of indices
/// @param[out] boolean_array: array of boolean values marking indices
template<typename Index>
void
convert_index_vector_to_boolean_array(const std::vector<Index>& index_vector,
                                      Index num_indices,
                                      std::vector<bool>& boolean_array)
{
  boolean_array.resize(num_indices, false);
  for (size_t i = 0; i < index_vector.size(); ++i) {
    boolean_array[index_vector[i]] = true;
  }
}

/// @brief From a boolean array, build a vector of the indices that are true.
///
/// @param[in] boolean_array: array of boolean values
/// @param[out] index_vector: indices where the array is true
template<typename Index>
void
convert_boolean_array_to_index_vector(const std::vector<bool>& boolean_array,
                                      std::vector<Index>& index_vector)
{
  size_t num_indices = boolean_array.size();
  index_vector.clear();
  index_vector.reserve(num_indices);
  for (size_t i = 0; i < num_indices; ++i) {
    if (boolean_array[i]) {
      index_vector.push_back(i);
    }
  }
}

/// @brief From a vector of the indices, build the complement of indices
///
/// @param[in] index_vector: list of indices 
/// @param[in] num_indices: total number of indices
/// @param[in] complement_vector: complement of indices
template<typename Index>
void
index_vector_complement(const std::vector<Index>& index_vector,
                        Index num_indices,
                        std::vector<Index>& complement_vector)
{
  // Build index boolean array
  std::vector<bool> boolean_array;
  convert_index_vector_to_boolean_array(
    index_vector, num_indices, boolean_array);

  // Build complement
  complement_vector.clear();
  complement_vector.reserve(num_indices - index_vector.size());
  for (Index i = 0; i < num_indices; ++i) {
    if (!boolean_array[i]) {
      complement_vector.push_back(i);
    }
  }
}

/// Given a boolean array, enumerate the true and false entries.
///
/// @param[in] boolean_array: array of boolean values
/// @param[out] true_entry_list: indices where the array is true
/// @param[out] false_entry_list: indices where the array is false
/// @param[out] array_to_list_map: map from indices to their position in the true or false lists
inline void
enumerate_boolean_array(
  const std::vector<bool>& boolean_array,
  std::vector<int>& true_entry_list,
  std::vector<int>& false_entry_list,
  std::vector<int>& array_to_list_map
) {
  int num_entries = boolean_array.size();
  true_entry_list.clear();
  true_entry_list.reserve(num_entries);
  false_entry_list.clear();
  false_entry_list.reserve(num_entries);

  // Iterate over the boolean array to enumerate the true and false entries
  for (int i = 0; i < num_entries; ++i)
  {
    if (boolean_array[i])
    {
      true_entry_list.push_back(i);
    }
    else
    {
      false_entry_list.push_back(i);
    }
  }

  // Generate the reverse map from array entries to list indices
  int num_true_entries = true_entry_list.size();
  int num_false_entries = false_entry_list.size();
  array_to_list_map.clear();
  array_to_list_map.resize(num_entries, -1);
  for (int i = 0; i < num_true_entries; ++i)
  {
    array_to_list_map[true_entry_list[i]] = i;
  }
  for (int i = 0; i < num_false_entries; ++i)
  {
    array_to_list_map[false_entry_list[i]] = i;
  }
}

/// ****
/// Mesh
/// ****

/// Compute the number of connected components of a mesh
///
/// @param[in] F: mesh faces
/// @return number of connected components
inline int
count_components(
  const Eigen::MatrixXi& F
) {
	Eigen::VectorXi face_components;
	igl::facet_components(F, face_components);
	return face_components.maxCoeff() + 1;
}

/// Given a face index matrix, reindex the vertex indices to removed unreferenced
/// vertex indices in O(|F|) time.
///
/// Note that libigl has function with similar behavior, but it is a O(|V| + |F|)
/// algorithm due to their bookkeeping method
///
/// @param[in] F: initial mesh faces
/// @param[out] FN: reindexed mesh faces
/// @param[out] new_to_old_map: map from new to old vertex indices
/// @return number of connected components
inline void
remove_unreferenced(
  const Eigen::MatrixXi& F,
  Eigen::MatrixXi& FN,
  std::vector<int>& new_to_old_map
) {
  int num_faces = F.rows();

  // Iterate over faces to find all referenced vertices in sorted order
  std::vector<int> referenced_vertices;
  for (int fi = 0; fi < num_faces; ++fi)
  {
    for (int j = 0; j < 3; ++j)
    {
      int vk = F(fi, j);
      referenced_vertices.push_back(vk);
    }
  }

  // Make the list of referenced vertices sorted and unique
  std::sort(referenced_vertices.begin(), referenced_vertices.end());
  auto last_sorted = std::unique(referenced_vertices.begin(), referenced_vertices.end()); 

  // Get the new to old map from the sorted referenced vertices list
  new_to_old_map.assign(referenced_vertices.begin(), last_sorted);

  // Build a (compact) map from old to new vertices
  int num_vertices = new_to_old_map.size();
  std::unordered_map<int, int> old_to_new_map;
  for (int k = 0; k < num_vertices; ++k)
  {
    int vk = new_to_old_map[k];
    old_to_new_map[vk] = k;
  }

  // Reindex the vertices in the face list
  FN.resize(num_faces, 3);
  for (int fi = 0; fi < num_faces; ++fi)
  {
    for (int j = 0; j < 3; ++j)
    {
      int vk = F(fi, j);
      int k = old_to_new_map[vk];
      FN(fi, j) = k;
    }
  }
}

/// Given a mesh with a parametrization, cut the mesh along the parametrization seams to
/// create a vertex set corresponding to the faces of the uv domain.
///
/// @param[in] V: mesh vertices
/// @param[in] F: mesh faces
/// @param[in] uv: parametrization vertices
/// @param[in] FT: parametrization faces
/// @param[in] V: cut mesh vertices
inline void
cut_mesh_along_parametrization_seams(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& uv,
  const Eigen::MatrixXi& FT,
  Eigen::MatrixXd& V_cut
) {
  int num_uv_vertices = uv.rows();
  int num_uv_faces = FT.rows();

  // Check input validity
  if (F.rows() != num_uv_faces)
  {
    spdlog::error("F and FT have a different number of faces");
    return;
  }

  // Copy by face index correspondences
  V_cut.resize(num_uv_vertices, 3);
  for (int f = 0; f < num_uv_faces; ++f)
  {
    for (int i = 0; i < 3; ++i)
    {
      int vi = F(f, i);
      int uvi = FT(f, i);
      V_cut.row(uvi) = V.row(vi);
    }
  }
}

} // namespace CurvatureMetric
