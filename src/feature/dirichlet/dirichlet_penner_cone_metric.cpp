#include "feature/dirichlet/dirichlet_penner_cone_metric.h"

#include "util/boundary.h"

#include "holonomy/holonomy/constraint.h"
#include "holonomy/holonomy/newton.h"

#include "feature/feature/gluing.h"
#include "feature/dirichlet/constraint.h"

namespace Penner {
namespace Feature {

DirichletPennerConeMetric::DirichletPennerConeMetric()
    : MarkedPennerConeMetric(Mesh<Scalar>(), VectorX(0), {}, {})
    , ell_hat(VectorX(0))
    , use_relaxed_system(false)
    , m_boundary_paths({})
    , m_is_boundary_path_valid({})
    , h2bd({})
    , m_boundary_constraint_system(MatrixX(0, 0))
    , m_full_angle_constraint_system(MatrixX(0, 0))
    , m_relaxed_angle_constraint_system(MatrixX(0, 0))
{}

DirichletPennerConeMetric::DirichletPennerConeMetric(
    const Mesh<Scalar>& m,
    const VectorX& metric_coords,
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    const std::vector<Scalar>& kappa,
    const std::vector<BoundaryPath>& boundary_paths,
    const MatrixX& boundary_constraint_system,
    const VectorX& ell)
    : MarkedPennerConeMetric(m, metric_coords, homology_basis_loops, kappa)
    , ell_hat(ell)
    , use_relaxed_system(false)
    , m_boundary_paths(boundary_paths)
    , m_is_boundary_path_valid(boundary_paths.size(), false)
    , h2bd(m.n_halfedges(), -1)
    , m_boundary_constraint_system(boundary_constraint_system)
    , m_full_angle_constraint_system(Holonomy::build_free_vertex_system(m))
    , m_relaxed_angle_constraint_system(m_full_angle_constraint_system)
{}

DirichletPennerConeMetric::DirichletPennerConeMetric(
    const Mesh<Scalar>& m,
    const VectorX& metric_coords,
    const std::vector<std::unique_ptr<DualLoop>>& homology_basis_loops,
    const std::vector<Scalar>& kappa,
    const std::vector<BoundaryPath>& boundary_paths,
    const MatrixX& boundary_constraint_system,
    const VectorX& ell,
    const MatrixX& angle_constraint_system)
    : MarkedPennerConeMetric(m, metric_coords, homology_basis_loops, kappa)
    , ell_hat(ell)
    , use_relaxed_system(false)
    , m_boundary_paths(boundary_paths)
    , m_is_boundary_path_valid(boundary_paths.size(), false)
    , h2bd(m.n_halfedges(), -1)
    , m_boundary_constraint_system(boundary_constraint_system)
    , m_full_angle_constraint_system(Holonomy::build_free_vertex_system(m))
    , m_relaxed_angle_constraint_system(angle_constraint_system)
    {}

DirichletPennerConeMetric::DirichletPennerConeMetric(
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<BoundaryPath>& boundary_paths,
    const MatrixX& boundary_constraint_system,
    const VectorX& ell)
    : DirichletPennerConeMetric(
          marked_metric,
          marked_metric.get_reduced_metric_coordinates(),
          marked_metric.get_homology_basis_loops(),
          marked_metric.kappa_hat,
          boundary_paths,
          boundary_constraint_system,
          ell)
{}

DirichletPennerConeMetric::DirichletPennerConeMetric(const DirichletPennerConeMetric& dirichlet_metric)
    : MarkedPennerConeMetric(dirichlet_metric)
{
    copy_feature(dirichlet_metric);
}

void DirichletPennerConeMetric::operator=(const DirichletPennerConeMetric& dirichlet_metric)
{
    copy_connectivity(dirichlet_metric);
    copy_metric(dirichlet_metric);
    copy_holonomy(dirichlet_metric);
    copy_feature(dirichlet_metric);
}

bool DirichletPennerConeMetric::flip_ccw(int _h, bool Ptolemy)
{
    // Perform the flip in the base class
    bool success = MarkedPennerConeMetric::flip_ccw(_h, Ptolemy);

    // mark boundary path as invalid if it contains the flipped edge
    if (h2bd[_h] >= 0)
    {
        m_is_boundary_path_valid[h2bd[_h]] = false;
        m_is_boundary_path_valid[h2bd[opp[_h]]] = false;
    }

    return success;
}

bool DirichletPennerConeMetric::constraint(
    VectorX& constraint,
    MatrixX& J_constraint,
    bool need_jacobian,
    bool only_free_vertices) const
{
    // TODO Should support all vertices
    if (!only_free_vertices) {
        spdlog::warn("Only free vertex constraint supported");
    }

    compute_dirichlet_constraint_with_jacobian(*this, constraint, J_constraint, need_jacobian);
    return true;
}

VectorX DirichletPennerConeMetric::constraint(const VectorX& angles)
{
    return compute_dirichlet_constraint(*this, angles);
}

MatrixX DirichletPennerConeMetric::constraint_jacobian(const VectorX& cotangents)
{
    return compute_dirichlet_constraint_jacobian(*this, cotangents);
}

std::unique_ptr<Optimization::DifferentiableConeMetric>
DirichletPennerConeMetric::set_metric_coordinates(const VectorX& metric_coords) const
{
    return std::make_unique<DirichletPennerConeMetric>(DirichletPennerConeMetric(
        *this,
        metric_coords,
        m_homology_basis_loops,
        kappa_hat,
        m_boundary_paths,
        m_boundary_constraint_system,
        ell_hat,
        m_relaxed_angle_constraint_system));
}

std::unique_ptr<Optimization::DifferentiableConeMetric> DirichletPennerConeMetric::project_to_constraint(
    SolveStats<Scalar>& solve_stats,
    std::shared_ptr<Optimization::ProjectionParameters> proj_params) const
{
    // Copy projection parameters to newton method parameters
    Holonomy::NewtonParameters alg_params;
    alg_params.max_itr = proj_params->max_itr;
    alg_params.bound_norm_thres = proj_params->bound_norm_thres;
    alg_params.error_eps = proj_params->error_eps;
    alg_params.do_reduction = proj_params->do_reduction;
    alg_params.output_dir = proj_params->output_dir;
    alg_params.error_log = true;
    alg_params.solver = "ldlt"; // FIXME

    // Optimize all metric angles
    MatrixX identity = id_matrix(n_reduced_coordinates());
    Holonomy::NewtonLog log;
    MarkedPennerConeMetric optimized_metric =
        Holonomy::optimize_subspace_metric_angles_log(*this, identity, alg_params, log);

    // create dirichlet mesh with optimized metric coordinates
    solve_stats.n_solves = log.num_iter;
    return set_metric_coordinates(optimized_metric.get_metric_coordinates());
}

void DirichletPennerConeMetric::write_status_log(std::ostream& stream, bool write_header)
{
    // optionally write header for status
    if (write_header) {
        stream << "num_flips,";
        stream << "num_bd_paths,";
        stream << "num_bd_segments,";
        stream << "max_angle_error,";
        stream << "max_boundary_error,";
        stream << "sq_norm_angle_error,";
        stream << "sq_norm_boundary_error,";
        stream << std::endl;
    }

    // compute necessary status data
    VectorX alpha, cot_alpha;
    make_discrete_metric();
    get_corner_angles(alpha, cot_alpha);
    VectorX metric_constraint = compute_metric_constraint(*this, alpha);
    VectorX boundary_constraint = compute_boundary_constraint(*this);
    Scalar max_angle_error = metric_constraint.cwiseAbs().maxCoeff();
    Scalar max_boundary_error =
        (n_boundary_paths() == 0) ? 0. : boundary_constraint.cwiseAbs().maxCoeff();
    Scalar sq_norm_angle_error = metric_constraint.squaredNorm();
    Scalar sq_norm_boundary_error = boundary_constraint.squaredNorm();

    // write status row to file
    stream << num_flips() << ",";
    stream << n_boundary_paths() << ",";
    stream << n_boundary_segments() << ",";
    stream << std::scientific << std::setprecision(8) << max_angle_error << ",";
    stream << std::scientific << std::setprecision(8) << max_boundary_error << ",";
    stream << std::scientific << std::setprecision(8) << sq_norm_angle_error << ",";
    stream << std::scientific << std::setprecision(8) << sq_norm_boundary_error << ",";
    stream << std::endl;
}

int DirichletPennerConeMetric::n_boundary_segments()
{
    // sum over boundary paths
    int num_bd_segments = 0;
    const auto& boundary_paths = get_boundary_paths();
    for (const auto& boundary_path : boundary_paths) {
        num_bd_segments += boundary_path.size();
    }

    return num_bd_segments;
}

const std::vector<BoundaryPath>& DirichletPennerConeMetric::get_boundary_paths()
{
    // rebuild invalid boundary paths from scratch
    int num_bd_paths = m_boundary_paths.size();
    for (int i = 0; i < num_bd_paths; ++i) {
        int start_vertex = m_boundary_paths[i].get_start_vertex();
        m_boundary_paths[i] = BoundaryPath(*this, start_vertex);

        // TODO: add bookkeeping if profiling shows it is important
    }

    return m_boundary_paths;
}

std::vector<int> DirichletPennerConeMetric::get_path_starting_vertices() const
{
    int num_bd_paths = m_boundary_paths.size();
    std::vector<int> starting_vertices(num_bd_paths);
    for (int i = 0; i < num_bd_paths; ++i) {
        starting_vertices[i] = m_boundary_paths[i].get_start_vertex();
    }

    return starting_vertices;
}

// helper function to serialize a vector
template <typename VectorType>
void serialize_vector(std::ostream& output, const VectorType& v, const std::string& label)
{
    output << label;
    int n = v.size();
    for (int i = 0; i < n; ++i) {
        output << " " << v[i];
    }
    output << '\n';
}

void DirichletPennerConeMetric::serialize(std::ostream& output) const
{
    serialize_vector(output, n, "next");
    serialize_vector(output, opp, "opp");
    serialize_vector(output, Th_hat, "Th_hat");
    serialize_vector(output, kappa_hat, "kappa_hat");
    serialize_vector(output, ell_hat, "ell_hat");
}

// rebuild given boundary path
// TODO: optional bookkeeping; not currently used
void DirichletPennerConeMetric::reset_boundary_path(int bd_index)
{
    // unmark representative boundary path edges
    for (int hij : m_boundary_paths[bd_index].get_halfedge_path())
    {
        h2bd[hij] = -1;
    }

    // unmark transverse edges
    for (int hij : m_boundary_paths[bd_index].get_transverse_edges())
    {
        h2bd[hij] = -1;
    }

    // rebuild boundary path from scratch
    int start_vertex = m_boundary_paths[bd_index].get_start_vertex();
    m_boundary_paths[bd_index] = BoundaryPath(*this, start_vertex);

    // mark representative boundary path edges as part of this path
    for (int hij : m_boundary_paths[bd_index].get_halfedge_path())
    {
        h2bd[hij] = bd_index;
    }

    // mark transverse edges as part of this path
    for (int hij : m_boundary_paths[bd_index].get_transverse_edges())
    {
        h2bd[hij] = bd_index;
    }

    // mark path as valid
    m_is_boundary_path_valid[bd_index] = true;
}

// rebuild all boundary paths
// TODO: optional bookkeeping; not currently used
void DirichletPennerConeMetric::reset_boundary_paths()
{
    // Rebuild all boundary paths
    int num_bd_paths = m_boundary_paths.size();
    for (int i = 0; i < num_bd_paths; ++i) {
        reset_boundary_path(i);
    }
}

// copy feature data
void DirichletPennerConeMetric::copy_feature(const DirichletPennerConeMetric& m)
{
    m_boundary_paths = m.m_boundary_paths;
    m_is_boundary_path_valid = m.m_is_boundary_path_valid;
    h2bd = m.h2bd;
    m_boundary_constraint_system = m.m_boundary_constraint_system;
    m_full_angle_constraint_system = m.m_full_angle_constraint_system;
    m_relaxed_angle_constraint_system = m.m_relaxed_angle_constraint_system;
    ell_hat = m.ell_hat;
    use_relaxed_system = m.use_relaxed_system;
}

DirichletPennerConeMetric generate_dirichlet_metric_from_mesh(
    const MarkedPennerConeMetric& marked_metric,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map)
{
    // generate boundary identification
    auto [bd_pairs, boundary_he, he2bd] =
        generate_boundary_pairs(marked_metric, vtx_reindex, V_map);

    // Build boundary paths
    int num_bd = boundary_he.size();
    std::vector<BoundaryPath> boundary_paths = {};
    boundary_paths.reserve(num_bd);
    for (int i = 0; i < num_bd; ++i) {
        int vi = marked_metric.to[marked_metric.opp[boundary_he[i]]];
        boundary_paths.push_back(BoundaryPath(marked_metric, vi));
    }

    // Make boundary constraint system
    typedef Eigen::Triplet<int> ScalarTrip;
    std::vector<ScalarTrip> system_trips;
    int num_bd_pairs = bd_pairs.size();
    for (int i = 0; i < num_bd_pairs; ++i) {
        const auto& bd_pair = bd_pairs[i];
        int hij = bd_pair.first;
        int hji = bd_pair.second;
        system_trips.push_back(ScalarTrip(i, he2bd[hij], 1.));
        system_trips.push_back(ScalarTrip(i, he2bd[hji], -1.));
        spdlog::debug("Adding constraints l[{}]==l[{}]", hij, hji);
    }
    MatrixX boundary_constraint_system(num_bd_pairs, num_bd);
    boundary_constraint_system.setFromTriplets(system_trips.begin(), system_trips.end());
    VectorX ell = VectorX::Zero(num_bd_pairs);

    return DirichletPennerConeMetric(
        marked_metric,
        boundary_paths,
        boundary_constraint_system,
        ell);
}


} // namespace Feature
} // namespace Penner