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
best_fit_conformal(const Mesh<Scalar>& m,
                   const VectorX& target_log_length_coords,
                   const VectorX& log_length_coords)
{
  // Construct psuedoinverse for the conformal scaling matrix and get the best
  // fit scale factors
  MatrixX B = conformal_scaling_matrix(m);
  MatrixX A = B.transpose() * B;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
  solver.compute(A);

  // Solve for the best fit conformal scale factor
  VectorX w = B.transpose() * (log_length_coords - target_log_length_coords);
  return solver.solve(w);
}

VectorX
scale_distortion_direction(const Mesh<Scalar>& m,
                           const VectorX& target_log_length_coords,
                           const VectorX& log_length_coords)
{
  // Compute the psuedoinverse for the conformal scaling matrix
  MatrixX B = conformal_scaling_matrix(m);
  MatrixX A = B.transpose() * B;

  // Solve for the gradient of the L2 norm of the best fit conformal scale
  // factor
  MatrixX L = A.transpose() * A;
  VectorX w = B.transpose() * (log_length_coords - target_log_length_coords);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
  solver.compute(L);
  VectorX g = solver.solve(w);
  return B * g;
}

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
face_halfedge_weight_matrix(const std::vector<Scalar>& face_weights,
                             MatrixX& M)
{
  // Iterate through faces and build diagonal energy matrix
  int num_faces = face_weights.size();
  std::vector<T> tripletList;
  tripletList.reserve(3 * num_faces);
  for (int f = 0; f < num_faces; ++f) {
    // Add local entries to global matrix list
    for (Eigen::Index i = 0; i < 3; ++i)
    {
      tripletList.push_back(T(3 * f + i, 3 * f + i, face_weights[f]));
    }
  }

  // Build matrix from triplets
  M.resize(3 * num_faces, 3 * num_faces);
  M.reserve(tripletList.size());
  M.setFromTriplets(tripletList.begin(), tripletList.end());
}


VectorX
surface_hencky_strain_energy_vf(const VectorX& area,
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

void
length_jacobian(const VectorX& log_length_coords, MatrixX& J_l)
{
  VectorX J_l_vec;
  J_l_vec.setZero(log_length_coords.size());
  for (int e = 0; e < log_length_coords.size(); ++e) {
    J_l_vec[e] = 2.0 / exp(log_length_coords[e]);
  }

  // The Jacobian is a diagonal matrix
  J_l = J_l_vec.asDiagonal();
}

MatrixX
generate_doubled_edge_to_face_he_matrix(const Mesh<Scalar>& m)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Create reindexing matrix
  int num_halfedges = m.n.size();
  int num_faces = m.h.size();
  std::vector<T> tripletList;
  tripletList.reserve(num_halfedges);
  for (int f = 0; f < num_faces; ++f) {
    // Get face halfedges
    int hij = m.h[f];
    int hjk = m.n[hij];
    int hki = m.n[hjk];

    // Get face projected edges
    int eij = he2e[hij];
    int ejk = he2e[hjk];
    int eki = he2e[hki];

    tripletList.push_back(T(3 * f + 0, eij, 1.0));
    tripletList.push_back(T(3 * f + 1, ejk, 1.0));
    tripletList.push_back(T(3 * f + 2, eki, 1.0));
  }
  MatrixX R;
  R.resize(3 * num_faces, e2he.size());
  R.reserve(tripletList.size());
  R.setFromTriplets(tripletList.begin(), tripletList.end());

  return R;
}

MatrixX
generate_edge_to_face_he_matrix(const Mesh<Scalar>& m)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Get reflection projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Create reindexing matrix
  int num_halfedges = m.n.size();
  int num_faces = m.h.size();
  std::vector<T> tripletList;
  tripletList.reserve(num_halfedges);
  for (int f = 0; f < num_faces; ++f) {
    // Get face halfedges
    int hij = m.h[f];
    int hjk = m.n[hij];
    int hki = m.n[hjk];

    // Get face projected edges
    int Eij = proj[he2e[hij]];
    int Ejk = proj[he2e[hjk]];
    int Eki = proj[he2e[hki]];

    tripletList.push_back(T(3 * f + 0, Eij, 1.0));
    tripletList.push_back(T(3 * f + 1, Ejk, 1.0));
    tripletList.push_back(T(3 * f + 2, Eki, 1.0));
  }
  MatrixX R;
  R.resize(3 * num_faces, embed.size());
  R.reserve(tripletList.size());
  R.setFromTriplets(tripletList.begin(), tripletList.end());

  return R;
}

VectorX
halfedge_function_to_vertices(const Mesh<Scalar>& m, const VectorX& g)
{
  // Initializes vector G to zero vector of length |V|
  VectorX G;
  G.setZero(m.n_ind_vertices());

  // Iterate over halfedges to compute the maximum
  int num_halfedges = m.n.size();
  for (int h = 0; h < num_halfedges; ++h) {
    G[m.v_rep[m.to[h]]] = max(G[m.v_rep[m.to[h]]], g[h]);
  }

  return G;
}

Scalar
compute_weighted_norm(
  const VectorX &weights,
  const VectorX &values
) {
  size_t num_values = values.size();
  assert( weights.size() == num_values );

  Scalar weighted_norm = 0;
  for (size_t i = 0; i < num_values; ++i)
  {
    weighted_norm += weights[i] * values[i] * values[i];
  }

  return weighted_norm;
}

void
compute_kronecker_product(
  const VectorX &first_vector,
  const VectorX &second_vector,
  VectorX &product_vector
) {
  size_t vector_size = first_vector.size();
  assert( second_vector.size() == vector_size );

  product_vector.resize(vector_size);
  for (size_t i = 0; i < vector_size; ++i)
  {
    product_vector[i] = first_vector[i] * second_vector[i];
  }
}

void
compute_face_area_weights(
  const Mesh<Scalar> &m,
  const VectorX &log_edge_lengths,
  VectorX &face_area_weights
) {
  // Compute area per halfedges
  VectorX he2area;
  areas_from_log_lengths(m, log_edge_lengths, he2area);

  // Reorganize areas to be per face
  int num_faces = m.h.size();
  face_area_weights.resize(num_faces);
  for (int f = 0; f < num_faces; ++f)
  {
    face_area_weights[f] = he2area[m.h[f]];
  }
  spdlog::trace("f to areas: {}", face_area_weights.transpose());
}

void
compute_edge_area_weights(
  const Mesh<Scalar> &m,
  const VectorX &log_edge_lengths,
  VectorX &edge_area_weights
) {
  // Build edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Compute area per halfedges
  VectorX he2area;
  areas_from_log_lengths(m, log_edge_lengths, he2area);

  // Compute edge weights as 1/3 of the adjacent face areas
  int num_edges = e2he.size();
  edge_area_weights.resize(num_edges);
  for (int e = 0; e < num_edges; ++e)
  {
    int h = e2he[e];
    edge_area_weights[e] = (1.0 / 3.0) * (he2area[h] + he2area[m.opp[h]]);
  }
}

// Compute the cone vertices of a closed mesh
void compute_cone_vertices(
  const Mesh<Scalar> &m,
  const ReductionMaps& reduction_maps,
  std::vector<int>& cone_vertices
) {
  cone_vertices.clear();
  int num_vertices = m.n_ind_vertices();

  // Closed meshes
  if (reduction_maps.bd_e.empty())
  {
    for (int v = 0; v < num_vertices; ++v)
    {
      if (!float_equal(m.Th_hat[v], 2 * M_PI))
      {
        cone_vertices.push_back(v);
      }
    }
  }
  // Open meshes
  else
  {
    // For open meshes, we iterate over edge endpoints with possible redundancy
    std::vector<bool> is_cone_vertex(num_vertices, false);

    // Iterate over interior edges
    int num_interior_edges = reduction_maps.int_e.size();
    for (int i = 0; i < num_interior_edges; ++i)
    {
      // Get two halfedges of the edge
      int E = reduction_maps.int_e[i];
      int e = reduction_maps.embed[E];
      int h = reduction_maps.e2he[e];
      int ho = m.opp[h];

      // Get two endpoint vertices
      int v_to = m.to[h];
      int v_fr = m.to[ho];

      // Regular interior vertices have doubled angle 4 pi
      is_cone_vertex[v_to] = (!float_equal(m.Th_hat[v_to], 4 * M_PI));
      is_cone_vertex[v_fr] = (!float_equal(m.Th_hat[v_fr], 4 * M_PI));
    }

    // Iterate over boundary edges
    int num_boundary_edges = reduction_maps.bd_e.size();
    for (int i = 0; i < num_boundary_edges; ++i)
    {
      // Get two halfedges of the edge
      int E = reduction_maps.bd_e[i];
      int e = reduction_maps.embed[E];
      int h = reduction_maps.e2he[e];
      int ho = m.opp[h];

      // Get two endpoint vertices
      int v_to = m.to[h];
      int v_fr = m.to[ho];

      // Regular boundary vertices have angle 2 pi
      is_cone_vertex[v_to] = (!float_equal(m.Th_hat[v_to], 2 * M_PI));
      is_cone_vertex[v_fr] = (!float_equal(m.Th_hat[v_fr], 2 * M_PI));
    }

    // Convert boolean list to index vector
    convert_boolean_array_to_index_vector(is_cone_vertex, cone_vertices);
  }
}

// Compute the cone adjacent faces of a closed mesh
void compute_cone_faces(
  const Mesh<Scalar> &m,
  const ReductionMaps& reduction_maps,
  std::vector<int>& cone_faces
) {
  // Compute the cone vertices
  std::vector<int> cone_vertices;
  compute_cone_vertices(m, reduction_maps, cone_vertices);

  // Get boolean mask for the cone vertices
  std::vector<bool> is_cone_vertex;
  int num_vertices = m.n_ind_vertices();
  convert_index_vector_to_boolean_array(cone_vertices, num_vertices, is_cone_vertex);

  // Compute the cone faces by iterating over the halfedges
  cone_faces.clear();
  int num_halfedges = m.n_halfedges();
  for (int h = 0; h < num_halfedges; ++h)
  {
    int v = m.v_rep[m.to[h]];
    if (is_cone_vertex[v])
    {
      int f = m.f[h];
      cone_faces.push_back(f);
    }
  }
}

void
compute_cone_face_weights(
  const Mesh<Scalar> &m,
  const ReductionMaps& reduction_maps,
  Scalar cone_weight,
  std::vector<Scalar>& face_weights
) {
  std::vector<int> cone_faces;
  compute_cone_faces(m, reduction_maps, cone_faces);
  spdlog::info("Weighting {} faces with {}", cone_faces.size(), cone_weight);
  face_weights = std::vector<Scalar>(m.h.size(), 1.0);
  for (size_t i = 0; i < cone_faces.size(); ++i)
  {
    face_weights[cone_faces[i]] = cone_weight;
  }
}

void
compute_boundary_face_weights(
  const Mesh<Scalar> &m,
  const ReductionMaps& reduction_maps,
  Scalar bd_weight,
  std::vector<Scalar>& face_weights
) {
  // Initialize face weights to 1
  face_weights = std::vector<Scalar>(m.h.size(), 1.0);

  // Iterate over boundary edges
  int num_boundary_edges = reduction_maps.bd_e.size();
  for (int i = 0; i < num_boundary_edges; ++i)
  {
    // Get two halfedges of the edge
    int E = reduction_maps.bd_e[i];
    int e = reduction_maps.embed[E];
    int h = reduction_maps.e2he[e];
    int ho = m.opp[h];

    // Get two faces adjacent to the edge
    int f = m.f[h];
    int fo = m.f[ho];

    // Set face weights
    face_weights[f] = bd_weight;
    face_weights[fo] = bd_weight;
  }

}

EnergyFunctor::EnergyFunctor(
  const Mesh<Scalar> &m,
  const VectorX &metric_target,
  const OptimizationParameters& opt_params
) {
  m_mesh = m;
  m_opt_params = opt_params;

  // Get reduction maps
  ReductionMaps reduction_maps(m, opt_params.fix_bd_lengths);

  // Set optimization parameters
  m_energy_choice = opt_params.energy_choice;
  m_lp_order = opt_params.p;
  m_two_norm_weight = opt_params.reg_factor;
  m_surface_hencky_strain_weight = 1.0;
  m_cone_weight = opt_params.cone_weight;
  m_bd_weight = opt_params.bd_weight;
  spdlog::info("Initializing {} energy functor", m_energy_choice);

  // Get reflection projection and embedding
  build_refl_proj(m, m_proj, m_embed);
  build_refl_matrix(m, m_projection);

  // Set target metric
  m_metric_target = metric_target;

  // Set original log edge lengths
  VectorX reduced_log_edge_lengths;
  compute_log_edge_lengths(m, reduced_log_edge_lengths);
  expand_reduced_function(m_proj, reduced_log_edge_lengths, m_log_edge_lengths);

  // Set area weights
  compute_edge_area_weights(m, m_log_edge_lengths, m_edge_area_weights);
  compute_face_area_weights(m, m_log_edge_lengths, m_face_area_weights);
  spdlog::trace("Edge area weights: {}", m_edge_area_weights.transpose());
  spdlog::trace("Face area weights: {}", m_face_area_weights.transpose());

  // Normalize area weights
  m_mesh_area = m_face_area_weights.sum();
  m_edge_area_weights /= m_mesh_area;
  m_face_area_weights /= m_mesh_area;
  spdlog::info("Mesh has area {}", m_mesh_area);
  spdlog::trace("Normalized edge area weights: {}", m_edge_area_weights.transpose());
  spdlog::trace("Normalized face area weights: {}", m_face_area_weights.transpose());

  // Compute quadratic energy matrix
  MatrixX R = generate_doubled_edge_to_face_he_matrix(m_mesh);
  MatrixX M;
  surface_hencky_strain_energy(m_mesh, m_log_edge_lengths, M);

  // Add cone face weights if nontrivial
  if (!float_equal(m_cone_weight, 1.0))
  {
    std::vector<Scalar> face_weights;
    compute_cone_face_weights(m, reduction_maps, m_cone_weight, face_weights);
    MatrixX M_face_weight;
    face_halfedge_weight_matrix(face_weights, M_face_weight);
    M = M * M_face_weight;
  }

  // Add boundary face weights if nontrivial
  if (!float_equal(m_bd_weight, 1.0))
  {
    std::vector<Scalar> face_weights;
    compute_boundary_face_weights(m, reduction_maps, m_bd_weight, face_weights);
    MatrixX M_face_weight;
    face_halfedge_weight_matrix(face_weights, M_face_weight);
    M = M * M_face_weight;
  }

  // Compute quadratic energy matrix_inverse
  // WARNING Only valid for vanilla surface hencky strain
  MatrixX inverse_M;
  surface_hencky_strain_energy_inverse(m_mesh, m_log_edge_lengths, inverse_M);

  m_quadratic_energy_matrix = R.transpose() * M * R;
  m_quadratic_energy_matrix_inverse = R.transpose() * inverse_M * R;
}

Scalar EnergyFunctor::compute_two_norm_energy(const VectorX& metric_coords) const
{
  VectorX difference = metric_coords - m_metric_target;
  return 0.5 * difference.squaredNorm();
}

Scalar EnergyFunctor::compute_p_norm_energy(const VectorX& metric_coords, int p) const
{
  int num_edges = metric_coords.size();
  VectorX difference = metric_coords - m_metric_target;
  Scalar energy = 0.0;
  for (int E = 0; E < num_edges; ++E)
  {
    // Iterate to compute the pth power of the edge difference absolute value
    Scalar term = 1.0;
    Scalar edge_difference = difference[E];
    for (int i = 0; i < p; ++i)
    {
      term *= abs(edge_difference);
    }

    // Add the term (weighted by 1/p)
    energy += term / static_cast<Scalar>(p);
  }
  return energy;
}

Scalar EnergyFunctor::compute_surface_hencky_strain_energy(const VectorX& metric_coords) const
{
  VectorX difference = metric_coords - m_metric_target;
  return 0.5 * (difference.dot((m_quadratic_energy_matrix * difference)));
}

Scalar EnergyFunctor::compute_scale_distortion_energy(const VectorX& metric_coords) const
{
  VectorX u = best_fit_conformal(m_mesh, m_metric_target, metric_coords);
  return 0.5 * (u.dot(u));
}

Scalar EnergyFunctor::compute_cone_energy(const VectorX& metric_coords) const
{
  VectorX constraint;
  MatrixX J_constraint;
  std::vector<int> flip_seq;
  bool need_jacobian = false;
  constraint_with_jacobian(m_mesh,
                           metric_coords,
                           constraint,
                           J_constraint,
                           flip_seq,
                           need_jacobian,
                           m_opt_params.use_edge_lengths);

  // Add L2 constraint error for fixed dof of the projection
  Scalar energy = 0;
  for (size_t v = 0; v < m_mesh.fixed_dof.size(); ++v) {
    if (m_mesh.fixed_dof[v]) {
      energy += 0.5 * constraint[v] * constraint[v];
    }
  }

  return energy;
}

Scalar EnergyFunctor::compute_symmetric_dirichlet_energy(const VectorX& metric_coords) const
{
  // Get per face symmetric dirichlet energy
  VectorX f2energy;
  MatrixX J_f2energy;
  bool need_jacobian = false;
  symmetric_dirichlet_energy(
    m_mesh,
    m_metric_target,
    metric_coords,
    f2energy,
    J_f2energy,
    need_jacobian
  );
  spdlog::trace(
    "Symmetric dirichlet face energy: {}",
    f2energy.transpose()
  );
  spdlog::trace(
    "Symmetric dirichlet face energy: {}",
    f2energy.transpose()
  );

  // Compute face integrated symmetric dirichlet energy
  return f2energy.dot(m_face_area_weights);
}

Scalar EnergyFunctor::energy(const VectorX& metric_coords) const
{
  if (m_energy_choice == "default")
  {
    // Compute energy terms
    Scalar two_norm_term = m_mesh_area * compute_two_norm_energy(metric_coords);
    Scalar surface_hencky_strain_term = compute_surface_hencky_strain_energy(metric_coords);
    spdlog::trace("Two norm term is {}", two_norm_term);
    spdlog::trace("Surface hencky strain term is {}", surface_hencky_strain_term);

    // Compute weighted energy
    Scalar weighted_energy = 0;
    weighted_energy += m_two_norm_weight * two_norm_term;
    weighted_energy += m_surface_hencky_strain_weight * surface_hencky_strain_term; 
    spdlog::trace("Weighted energy is {}", weighted_energy);
    
    return weighted_energy;
  }
  else if ((m_energy_choice == "p_norm") && (m_lp_order == 2))
  {
    return compute_two_norm_energy(metric_coords);
  }
  else if ((m_energy_choice == "p_norm") && (m_lp_order != 2))
  {
    return compute_p_norm_energy(metric_coords, m_lp_order);
  }
  else if (m_energy_choice == "surface_hencky_strain")
  {
    return compute_surface_hencky_strain_energy(metric_coords);
  }
  else if (m_energy_choice == "scale_distortion")
  {
    return compute_scale_distortion_energy(metric_coords);
  }
  else if (m_energy_choice == "sym_dirichlet")
  {
    return compute_symmetric_dirichlet_energy(metric_coords);
  }

  spdlog::error("No energy selected");
  return 0;
}

VectorX EnergyFunctor::compute_cone_gradient(const VectorX& metric_coords) const
{
  VectorX constraint;
  MatrixX J_constraint;
  std::vector<int> flip_seq;
  bool need_jacobian = true;
  constraint_with_jacobian(m_mesh,
                           metric_coords,
                           constraint,
                           J_constraint,
                           flip_seq,
                           need_jacobian,
                           m_opt_params.use_edge_lengths);

  // Add L2 constraint error gradient for fixed dof of the projection
  VectorX cone_gradient;
  cone_gradient.setZero(metric_coords.size());
  for (size_t v = 0; v < m_mesh.fixed_dof.size(); ++v) {
    if (m_mesh.fixed_dof[v]) {
      cone_gradient += constraint[v] * J_constraint.row(v);
    }
  }

  return cone_gradient;
}

VectorX EnergyFunctor::gradient(const VectorX& metric_coords) const
{
  if (m_energy_choice == "default")
  {
    VectorX difference = metric_coords - m_metric_target;
    VectorX surface_hencky_strain_gradient = m_quadratic_energy_matrix * difference;

    VectorX energy_gradient = m_mesh_area * m_two_norm_weight * difference;
    energy_gradient += m_surface_hencky_strain_weight * surface_hencky_strain_gradient; 
    return energy_gradient;
  }
  else if ((m_energy_choice == "p_norm") && (m_lp_order == 2))
  {
    VectorX difference = metric_coords - m_metric_target;
    return difference;
  }
  else if ((m_energy_choice == "p_norm") && (m_lp_order != 2))
  {
    int num_edges = metric_coords.size();
    VectorX difference = metric_coords - m_metric_target;
    VectorX p_norm_gradient(num_edges);
    for (int E = 0; E < num_edges; ++E)
    {
      // Iterate to compute the gradient of the edge difference as the signed
      // p-1th power of the absolute value
      Scalar edge_difference = difference[E];
      p_norm_gradient[E] = edge_difference;
      for (int i = 2; i < m_lp_order; ++i)
      {
        p_norm_gradient[E] *= abs(edge_difference);
      }
    }
    return p_norm_gradient;
  }
  else if (m_energy_choice == "surface_hencky_strain")
  {
    VectorX difference = metric_coords - m_metric_target;
    return m_quadratic_energy_matrix * difference;
  }
  else if (m_energy_choice == "scale_distortion")
  {
    return scale_distortion_direction(m_mesh, m_metric_target, metric_coords);
  }
  else if (m_energy_choice == "sym_dirichlet")
  {
    // Get per face symmetric dirichlet energy
    VectorX f2energy;
    MatrixX J_f2energy;
    bool need_jacobian = true;
    symmetric_dirichlet_energy(
      m_mesh,
      m_metric_target,
      metric_coords,
      f2energy,
      J_f2energy,
      need_jacobian
    );

    // Compute face integrated symmetric dirichlet energy Jacobian
    return (m_face_area_weights.transpose() * J_f2energy) * m_projection;
  }

  spdlog::error("No energy selected");
  return metric_coords;
}

MatrixX EnergyFunctor::hessian() const
{
  if ((m_energy_choice == "p_norm") && (m_lp_order == 2))
  {
    return id_matrix(m_metric_target.size());
  }
  else if (m_energy_choice == "surface_hencky_strain")
  {
    return m_quadratic_energy_matrix;
  }

  spdlog::error("Invalid energy for hessian selected");
  return id_matrix(m_metric_target.size());
}

MatrixX EnergyFunctor::hessian_inverse() const
{
  if ((m_energy_choice == "p_norm") && (m_lp_order == 2))
  {
    return id_matrix(m_metric_target.size());
  }
  else if (m_energy_choice == "surface_hencky_strain")
  {
    return m_quadratic_energy_matrix_inverse;
  }

  spdlog::error("Invalid energy for hessian selected");
  return id_matrix(m_metric_target.size());
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

MatrixX
length_jacobian_pybind(const VectorX& lambdas_full)
{
  MatrixX J_l;
  length_jacobian(lambdas_full, J_l);

  return J_l;
}
#endif

}

