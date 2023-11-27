#include "energies.hh"

#include "area.hh"
#include "constraint.hh"
#include "projection.hh"
#include "targets.hh"
#include <igl/doublearea.h>
#include <igl/cotmatrix_entries.h>
#include <igl/edge_lengths.h>

/// FIXME Do cleaning pass

namespace CurvatureMetric {

void
first_invariant(const Mesh<Scalar>& m,
                const VectorX& target_log_length_coords,
                const VectorX& log_length_coords,
                VectorX& f2J1,
                MatrixX& J_f2J1,
                bool need_jacobian)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Compute areas and angles from the target metric
  VectorX he2areasq_target;
  VectorX he2angles_target;
  VectorX he2cot_target;
  areas_squared_from_log_lengths(m, target_log_length_coords, he2areasq_target);
  corner_angles(m, target_log_length_coords, he2angles_target, he2cot_target);

  // Convert mesh log edge lengths to a halfedge length array l for m
  int num_halfedges = he2e.size();
  VectorX l_sq(num_halfedges);
  for (int h = 0; h < num_halfedges; ++h) {
    l_sq[h] = exp(log_length_coords[he2e[h]]);
  }

  // Compute array from halfedges to term of the first invariant corresponding
  // to its length
  VectorX he2J1he(num_halfedges);
  for (int h = 0; h < num_halfedges; ++h) {
    Scalar area_target = sqrt(max(he2areasq_target[h], 0.0));
    he2J1he[h] = (he2cot_target[h] * l_sq[h]) / (2.0 * area_target);
  }

  // Sum over halfedges of faces to get array mapping faces to first invariant
  int num_faces = m.h.size();
  f2J1.setZero(num_faces);
  for (int h = 0; h < num_halfedges; ++h) {
    int f = m.f[h];
    f2J1[f] += he2J1he[h];
  }

  if (need_jacobian) {
    // Create list of triplets of Jacobian indices and values
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
      tripletList.push_back(T(m.f[h], he2e[h], he2J1he[h]));
    }

    // Build Jacobian from triplets
    J_f2J1.resize(num_faces, log_length_coords.size());
    J_f2J1.reserve(num_halfedges);
    J_f2J1.setFromTriplets(tripletList.begin(), tripletList.end());
  }
}

void
second_invariant_squared(const Mesh<Scalar>& m,
                         const VectorX& target_log_length_coords,
                         const VectorX& log_length_coords,
                         VectorX& f2J2sq,
                         MatrixX& J_f2J2sq,
                         bool need_jacobian)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Compute areas the metric and target metric
  VectorX he2areasq_target;
  VectorX he2areasq;
  areas_squared_from_log_lengths(m, target_log_length_coords, he2areasq_target);
  areas_squared_from_log_lengths(m, log_length_coords, he2areasq);

  // Compute array from faces to squared second invariant
  int num_faces = m.h.size();
  f2J2sq.setZero(num_faces);
  for (int f = 0; f < num_faces; ++f) {
    int h = m.h[f];
    f2J2sq[f] = he2areasq[h] / he2areasq_target[h];
  }

  if (need_jacobian) {
    // Compute derivatives of area squared
    VectorX he2areasqderiv;
    area_squared_derivatives_from_log_lengths(
      m, log_length_coords, he2areasqderiv);

    // Create list of triplets of Jacobian indices and values
    int num_halfedges = m.n.size();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
      Scalar J_f2J2sq_fe = he2areasqderiv[h] / he2areasq_target[h];
      tripletList.push_back(T(m.f[h], he2e[h], J_f2J2sq_fe));
    }

    // Build Jacobian from triplets
    J_f2J2sq.resize(num_faces, log_length_coords.size());
    J_f2J2sq.reserve(num_halfedges);
    J_f2J2sq.setFromTriplets(tripletList.begin(), tripletList.end());
  }
}

void
metric_distortion_energy(const Mesh<Scalar>& m,
                         const VectorX& target_log_length_coords,
                         const VectorX& log_length_coords,
                         VectorX& f2energy,
                         MatrixX& J_f2energy,
                         bool need_jacobian)
{
  // Compute metric invariants used to construct the energy
  VectorX f2J1;
  MatrixX J_f2J1;
  first_invariant(m,
                  target_log_length_coords,
                  log_length_coords,
                  f2J1,
                  J_f2J1,
                  need_jacobian);

  VectorX f2J2sq;
  MatrixX J_f2J2sq;
  second_invariant_squared(m,
                           target_log_length_coords,
                           log_length_coords,
                           f2J2sq,
                           J_f2J2sq,
                           need_jacobian);

  // Compute map from faces to metric distortion energy
  int num_faces = m.h.size();
  f2energy.setZero(num_faces);
  for (int f = 0; f < num_faces; ++f) {
    Scalar t1 = f2J1[f] - 1;
    Scalar t2 = f2J2sq[f];
    f2energy[f] = t1 * t1 - 2 * t2 + 1;
  }

  // Optionally compute the jacobian
  if (need_jacobian) {
    J_f2energy = 2.0 * (f2J1.asDiagonal() * J_f2J1 - J_f2J1 - J_f2J2sq);
  }
}

void
area_distortion_energy(const Mesh<Scalar>& m,
                       const VectorX& target_log_length_coords,
                       const VectorX& log_length_coords,
                       VectorX& f2energy,
                       MatrixX& J_f2energy,
                       bool need_jacobian)
{
  // Compute just the second metric invariant
  VectorX f2J2sq;
  MatrixX J_f2J2sq;
  second_invariant_squared(m,
                           target_log_length_coords,
                           log_length_coords,
                           f2J2sq,
                           J_f2J2sq,
                           need_jacobian);

  // Compute map from faces to energy
  int num_faces = m.h.size();
  f2energy.setZero(num_faces);
  for (int f = 0; f < num_faces; ++f) {
    f2energy[f] = (f2J2sq[f] - 1) * (f2J2sq[f] - 1);
  }

  // Optionally compute the Jacobian
  if (need_jacobian) {
    J_f2energy = 2.0 * (f2J2sq.asDiagonal() * J_f2J2sq - J_f2J2sq);
  }
}

void
symmetric_dirichlet_energy(const Mesh<Scalar>& m,
                           const VectorX& target_log_length_coords,
                           const VectorX& log_length_coords,
                           VectorX& f2energy,
                           MatrixX& J_f2energy,
                           bool need_jacobian)
{
  // Compute metric invariants used to construct the energy
  VectorX f2J1;
  MatrixX J_f2J1;
  first_invariant(m,
                  target_log_length_coords,
                  log_length_coords,
                  f2J1,
                  J_f2J1,
                  need_jacobian);

  VectorX f2J2sq;
  MatrixX J_f2J2sq;
  second_invariant_squared(m,
                           target_log_length_coords,
                           log_length_coords,
                           f2J2sq,
                           J_f2J2sq,
                           need_jacobian);

  // Compute map from faces to energy
  int num_faces = m.h.size();
  f2energy.setZero(num_faces);
  for (int f = 0; f < num_faces; ++f) {
    f2energy[f] = f2J1[f] * (1.0 + 1.0 / f2J2sq[f]);
  }

  // Optionally compute the Jacobian
  if (need_jacobian) {
    // Make temporary value arrays to simplify Jacobian calculations
    VectorX A;
    A.setZero(num_faces);
    for (int f = 0; f < num_faces; ++f) {
      A[f] = (1.0 + 1.0 / f2J2sq[f]);
    }
    VectorX B;
    B.setZero(num_faces);
    for (int f = 0; f < num_faces; ++f) {
      B[f] = f2J1[f] / (f2J2sq[f] * f2J2sq[f]);
    }

    // Compute Jacobian
    J_f2energy = A.asDiagonal() * J_f2J1 - B.asDiagonal() * J_f2J2sq;
  }
}

/// Compute the 3x3 energy matrix for the surface Hencky strain for a single face.
/// Length indices correspond to the opposite angle.
void
triangle_surface_hencky_strain_energy(
  const std::array<Scalar, 3> &lengths,
  const std::array<Scalar, 3> &cotangents,
  Scalar face_area,
  Eigen::Matrix<Scalar, 3, 3> &face_energy_matrix
) {
    // Extract values
    Scalar ljk = lengths[0];
    Scalar lki = lengths[1];
    Scalar lij = lengths[2];
    Scalar cotai = cotangents[0];
    Scalar cotaj = cotangents[1];
    Scalar cotak = cotangents[2];
    Scalar A = face_area;

    // Compute length and area ratios for clarity and (moderate) numerical
    // stability Default to 1e10 for A exactly 0
    Scalar rij = (A > 0) ? (lij * lij) / (2.0 * A) : 1e10;
    Scalar rjk = (A > 0) ? (ljk * ljk) / (2.0 * A) : 1e10;
    Scalar rki = (A > 0) ? (lki * lki) / (2.0 * A) : 1e10;
    Scalar ri = (A > 0) ? (lij * lki) / (2.0 * A) : 1e10;
    Scalar rj = (A > 0) ? (ljk * lij) / (2.0 * A) : 1e10;
    Scalar rk = (A > 0) ? (lki * ljk) / (2.0 * A) : 1e10;

    // Compute 3x3 energy matrix block for face f
    // FIXME: ri * cos(ai) = cot(ai), etc. Verify this and replace.
    // FIXME: Subtracting a constant 2 is bad for numerical stability. Look for
    // alternative.
    face_energy_matrix(0, 0) = A * (rij * rij) * ((4.0 * rk * rk) - 2.0);
    face_energy_matrix(1, 1) = A * (rjk * rjk) * ((4.0 * ri * ri) - 2.0);
    face_energy_matrix(2, 2) = A * (rki * rki) * ((4.0 * rj * rj) - 2.0);
    // Scalar Mi = (ri * ri) * ((4.0 * (rj * cos(aj)) * (rk * cos(ak))) - 2.0);
    // Scalar Mj = (rj * rj) * ((4.0 * (rk * cos(ak)) * (ri * cos(ai))) - 2.0);
    // Scalar Mk = (rk * rk) * ((4.0 * (ri * cos(ai)) * (rj * cos(aj))) - 2.0);
    // TODO Check this very carefully
    face_energy_matrix(0, 2) = A * (ri * ri) * ((4.0 * cotaj * cotak) - 2.0);
    face_energy_matrix(0, 1) = A * (rj * rj) * ((4.0 * cotak * cotai) - 2.0);
    face_energy_matrix(1, 2) = A * (rk * rk) * ((4.0 * cotai * cotaj) - 2.0);
    face_energy_matrix(2, 0) = face_energy_matrix(0, 2);
    face_energy_matrix(1, 0) = face_energy_matrix(0, 1);
    face_energy_matrix(2, 1) = face_energy_matrix(1, 2);
}

void
surface_hencky_strain_energy(const Mesh<Scalar>& m,
                             const VectorX& log_length_coords,
                             MatrixX& M)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Get angles and areas
  VectorX he2areasq;
  VectorX he2angles;
  VectorX he2cot;
  areas_squared_from_log_lengths(m, log_length_coords, he2areasq);
  corner_angles(m, log_length_coords, he2angles, he2cot);

  // Convert target mesh log edge lengths to a halfedge length array l for m
  int num_halfedges = he2e.size();
  VectorX l(num_halfedges);
  for (int h = 0; h < num_halfedges; ++h) {
    l[h] = exp(log_length_coords[he2e[h]] / 2.0);
  }

  // Iterate through faces and build block energy matrix
  int num_faces = m.h.size();
  std::vector<T> tripletList;
  tripletList.reserve(9 * num_faces);
  for (int f = 0; f < num_faces; ++f) {
    // Get face halfedges
    int hij = m.h[f];
    int hjk = m.n[hij];
    int hki = m.n[hjk];

    // Get length and angle information for face f
    std::array<Scalar, 3> lengths = { l[hjk], l[hki], l[hij] };
    std::array<Scalar, 3> cotangents = { he2cot[hjk], he2cot[hki], he2cot[hij] };
    Scalar face_area = sqrt(max(he2areasq[hij], 0.0));

    // Compute energy matrix for the face
    Eigen::Matrix<Scalar, 3, 3> face_energy_matrix;
    triangle_surface_hencky_strain_energy(
      lengths,
      cotangents,
      face_area,
      face_energy_matrix
    );

    // Add local entries to global matrix list
    for (Eigen::Index i = 0; i < 3; ++i)
    {
      for (Eigen::Index j = 0; j < 3; ++j)
      {
        tripletList.push_back(T(3 * f + i, 3 * f + j, face_energy_matrix(i, j)));
      }
    }
  }

  // Build matrix from triplets
  M.resize(3 * num_faces, 3 * num_faces);
  M.reserve(tripletList.size());
  M.setFromTriplets(tripletList.begin(), tripletList.end());
}

void
surface_hencky_strain_energy_inverse(const Mesh<Scalar>& m,
                             const VectorX& log_length_coords,
                             MatrixX& inverse_M)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Get angles and areas
  VectorX he2areasq;
  VectorX he2angles;
  VectorX he2cot;
  areas_squared_from_log_lengths(m, log_length_coords, he2areasq);
  corner_angles(m, log_length_coords, he2angles, he2cot);

  // Convert target mesh log edge lengths to a halfedge length array l for m
  int num_halfedges = he2e.size();
  VectorX l(num_halfedges);
  for (int h = 0; h < num_halfedges; ++h) {
    l[h] = exp(log_length_coords[he2e[h]] / 2.0);
  }

  // Iterate through faces and build block energy matrix
  int num_faces = m.h.size();
  std::vector<T> tripletList;
  tripletList.reserve(9 * num_faces);
  for (int f = 0; f < num_faces; ++f) {
    // Get face halfedges
    int hij = m.h[f];
    int hjk = m.n[hij];
    int hki = m.n[hjk];

    // Get length and angle information for face f
    std::array<Scalar, 3> lengths = { l[hjk], l[hki], l[hij] };
    std::array<Scalar, 3> cotangents = { he2cot[hjk], he2cot[hki], he2cot[hij] };
    Scalar face_area = sqrt(max(he2areasq[hij], 0.0));

    // Compute energy matrix for the face
    Eigen::Matrix<Scalar, 3, 3> face_energy_matrix;
    triangle_surface_hencky_strain_energy(
      lengths,
      cotangents,
      face_area,
      face_energy_matrix
    );

    // Invert energy matrix for the face
    Eigen::Matrix<Scalar, 3, 3> face_energy_matrix_inverse = face_energy_matrix.inverse();

    // Add local entries to global matrix list
    for (Eigen::Index i = 0; i < 3; ++i)
    {
      for (Eigen::Index j = 0; j < 3; ++j)
      {
        tripletList.push_back(T(3 * f + i, 3 * f + j, face_energy_matrix_inverse(i, j)));
      }
    }
  }

  // Build matrix from triplets
  inverse_M.resize(3 * num_faces, 3 * num_faces);
  inverse_M.reserve(tripletList.size());
  inverse_M.setFromTriplets(tripletList.begin(), tripletList.end());
}

void
first_invariant_vf(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv,
  VectorX &f2J1
) {
  int num_faces = F.rows();
  assert (F.rows() == F_uv.rows());

  // Embed uv coordinates in R3
  int num_uv_vertices = uv.rows();
  Eigen::MatrixXd uv_embed;
  uv_embed.setZero(num_uv_vertices, 3);
  uv_embed.block(0, 0, num_uv_vertices, 2) = uv;

  // Get per face values for the 3D embedding
  Eigen::VectorXd double_area_0;
  Eigen::MatrixXd cot_alpha_0;
  igl::doublearea(V, F, double_area_0);
  igl::cotmatrix_entries(V, F, cot_alpha_0);

  // Get per face values for the uv embedding
  Eigen::MatrixXd lengths;
  igl::edge_lengths(uv_embed, F_uv, lengths);

  // Compute the per face invariant
  f2J1.setZero(num_faces);
  for (int f = 0; f < num_faces; ++f)
  {
    for (int i = 0; i < 3; ++i)
    {
      f2J1[f] += (2.0 * cot_alpha_0(f, i) * lengths(f, i) * lengths(f, i)) / (double_area_0[f]);
    }
  }
}

void
second_invariant_vf(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv,
  VectorX &f2J2
) {
  int num_faces = F.rows();
  assert (F.rows() == F_uv.rows());

  // Embed uv coordinates in R3
  int num_uv_vertices = uv.rows();
  Eigen::MatrixXd uv_embed;
  uv_embed.setZero(num_uv_vertices, 3);
  uv_embed.block(0, 0, num_uv_vertices, 2) = uv;

  // Get per face values for the 3D embedding
  Eigen::VectorXd double_area_0, double_area;
  igl::doublearea(V, F, double_area_0);
  igl::doublearea(uv_embed, F_uv, double_area);

  // Compute the per face invariant
  f2J2.setZero(num_faces);
  for (int f = 0; f < num_faces; ++f)
  {
    f2J2[f] = double_area[f]/ double_area_0[f];
  }
}

void
symmetric_dirichlet_energy_vf(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv,
  VectorX& f2energy
) {
  int num_faces = F.rows();
  assert (F.rows() == F_uv.rows());

  // Compute metric invariants used to construct the energy
  VectorX f2J1, f2J2;
  first_invariant_vf(V, F, uv, F_uv, f2J1);
  second_invariant_vf(V, F, uv, F_uv, f2J2);

  // Compute map from faces to energy
  f2energy.setZero(num_faces);
  for (int f = 0; f < num_faces; ++f) {
    f2energy[f] = f2J1[f] * (1.0 + 1.0 / (f2J2[f] * f2J2[f]));
  }
}

VectorX
surface_hencky_strain_energy(const VectorX& area,
                                const MatrixX& cot_alpha,
                                const MatrixX& l,
                                const MatrixX& delta_ll)
{
  // Iterate through faces and build block energy matrix
  int num_faces = area.size();
  VectorX energy(num_faces);
  for (int f = 0; f < num_faces; ++f) {

    // Get length and angle information for face f
    Scalar lij = l.coeff(f, 0);
    Scalar ljk = l.coeff(f, 1);
    Scalar lki = l.coeff(f, 2);
    Scalar dllij = delta_ll.coeff(f, 0);
    Scalar dlljk = delta_ll.coeff(f, 1);
    Scalar dllki = delta_ll.coeff(f, 2);
    Scalar cotai = cot_alpha.coeff(f, 1);
    Scalar cotaj = cot_alpha.coeff(f, 2);
    Scalar cotak = cot_alpha.coeff(f, 0);
    Scalar A = area[f];

    // Compute length and area ratios for clarity and (moderate) numerical
    // stability Default to 1e10 for A exactly 0
    Scalar rij = (A > 0) ? (lij * lij) / (2.0 * A) : 1e10;
    Scalar rjk = (A > 0) ? (ljk * ljk) / (2.0 * A) : 1e10;
    Scalar rki = (A > 0) ? (lki * lki) / (2.0 * A) : 1e10;
    Scalar ri = (A > 0) ? (lij * lki) / (2.0 * A) : 1e10;
    Scalar rj = (A > 0) ? (ljk * lij) / (2.0 * A) : 1e10;
    Scalar rk = (A > 0) ? (lki * ljk) / (2.0 * A) : 1e10;

    // Compute 3x3 energy matrix block M for face f
    Scalar Mij = A * (rij * rij) * ((4.0 * rk * rk) - 2.0);
    Scalar Mjk = A * (rjk * rjk) * ((4.0 * ri * ri) - 2.0);
    Scalar Mki = A * (rki * rki) * ((4.0 * rj * rj) - 2.0);
    Scalar Mi = A * (ri * ri) * ((4.0 * cotaj * cotak) - 2.0);
    Scalar Mj = A * (rj * rj) * ((4.0 * cotak * cotai) - 2.0);
    Scalar Mk = A * (rk * rk) * ((4.0 * cotai * cotaj) - 2.0);

    // Compute energy for face
    Eigen::Matrix<Scalar, 3, 3> M;
    Eigen::Matrix<Scalar, 3, 1> dll;
    dll << dllij, dlljk, dllki;
    M << Mij, Mj, Mi, Mj, Mjk, Mk, Mi, Mk, Mki;
    energy[f] = (dll.transpose() * (M * dll))[0];
  }

  return energy;
}

// FIXME Rename these variables
// FIXME Ensure all pybind functions for the entire interface are in place
#ifdef PYBIND
std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
first_invariant_pybind(const Mesh<Scalar>& C,
                       const VectorX& lambdas_target_full,
                       const VectorX& lambdas_full,
                       bool need_jacobian)
{
  VectorX f2J1;
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> J_f2J1;
  first_invariant(
    C, lambdas_target_full, lambdas_full, f2J1, J_f2J1, need_jacobian);

  return std::make_tuple(f2J1, J_f2J1);
}

std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
second_invariant_squared_pybind(const Mesh<Scalar>& C,
                                const VectorX& lambdas_target_full,
                                const VectorX& lambdas_full,
                                bool need_jacobian)
{
  VectorX f2J2sq;
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> J_f2J2sq;
  second_invariant_squared(
    C, lambdas_target_full, lambdas_full, f2J2sq, J_f2J2sq, need_jacobian);

  return std::make_tuple(f2J2sq, J_f2J2sq);
}

VectorX
first_invariant_vf_pybind(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv
) {
  VectorX f2J1;
  first_invariant_vf(V, F, uv, F_uv, f2J1);
  return f2J1;
}

VectorX
second_invariant_vf_pybind(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F,
  const Eigen::MatrixXd &uv,
  const Eigen::MatrixXi &F_uv
) {
  VectorX f2J2;
  second_invariant_vf(V, F, uv, F_uv, f2J2);
  return f2J2;
}

std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
metric_distortion_energy_pybind(const Mesh<Scalar>& C,
                                const VectorX& lambdas_target_full,
                                const VectorX& lambdas_full,
                                bool need_jacobian)
{
  VectorX f2energy;
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> J_f2energy;
  metric_distortion_energy(
    C, lambdas_target_full, lambdas_full, f2energy, J_f2energy, need_jacobian);

  return std::make_tuple(f2energy, J_f2energy);
}

std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
area_distortion_energy_pybind(const Mesh<Scalar>& C,
                              const VectorX& lambdas_target_full,
                              const VectorX& lambdas_full,
                              bool need_jacobian)
{
  VectorX f2energy;
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> J_f2energy;
  area_distortion_energy(
    C, lambdas_target_full, lambdas_full, f2energy, J_f2energy, need_jacobian);

  return std::make_tuple(f2energy, J_f2energy);
}

std::tuple<VectorX, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>
symmetric_dirichlet_energy_pybind(const Mesh<Scalar>& C,
                                  const VectorX& lambdas_target_full,
                                  const VectorX& lambdas_full,
                                  bool need_jacobian)
{
  VectorX f2energy;
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> J_f2energy;
  symmetric_dirichlet_energy(
    C, lambdas_target_full, lambdas_full, f2energy, J_f2energy, need_jacobian);

  return std::make_tuple(f2energy, J_f2energy);
}

MatrixX
surface_hencky_strain_energy_pybind(const Mesh<Scalar>& m,
                                    const VectorX& lambdas_full)
{
  MatrixX M;
  surface_hencky_strain_energy(m, lambdas_full, M);

  return M;
}

VectorX
surface_hencky_strain_energy_vf(const VectorX& area,
                                const MatrixX& cot_alpha,
                                const MatrixX& l,
                                const MatrixX& delta_ll)
{
  return surface_hencky_strain_energy(area, cot_alpha, l, delta_ll);
}

#endif

}

