#include "holonomy/interface.h"

#include "holonomy/core/boundary_basis.h"
#include "holonomy/core/homology_basis.h"
#include "holonomy/field/intrinsic_field.h"
#include "holonomy/core/quality.h"
#include "holonomy/holonomy/cones.h"
#include "holonomy/holonomy/holonomy.h"
#include "holonomy/holonomy/rotation_form.h"
#include "holonomy/similarity/energy.h"
#include "util/boundary.h"

#include "optimization/core/cone_metric.h"
#include "optimization/core/constraint.h"
#include "optimization/parameterization/interpolation.h"
#include "optimization/parameterization/refinement.h"
#include "util/io.h"
#include "util/vector.h"


#include <igl/facet_components.h>

#include "conformal_ideal_delaunay/ConformalIdealDelaunayMapping.hh"
#include "conformal_ideal_delaunay/ConformalInterface.hh"

#include "geometrycentral/surface/integer_coordinates_intrinsic_triangulation.h"
#include "geometrycentral/surface/intrinsic_triangulation.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

namespace Penner {
namespace Holonomy {

std::vector<int> extend_vtx_reindex(const Mesh<Scalar>& m, const std::vector<int>& vtx_reindex)
{
    return vector_compose(vtx_reindex, m.v_rep);
}

std::tuple<MarkedPennerConeMetric, std::vector<int>> generate_marked_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Th_hat,
    const VectorX& rotation_form,
    std::vector<int> free_cones,
    MarkedMetricParameters marked_metric_params)
{
    // Convert VF mesh to halfedge
    bool fix_boundary = false;
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m = FV_to_double<Scalar>(
        V,
        F,
        uv,
        F_uv,
        Th_hat,
        vtx_reindex,
        indep_vtx,
        dep_vtx,
        v_rep,
        bnd_loops,
        free_cones,
        fix_boundary);

    // Check for invalid cones
    if (!validate_cones(m)) {
        spdlog::info("Fixing invalid cones");
        fix_cones(m);
    }

    // Use halfedge mesh method
    MarkedPennerConeMetric marked_metric =
        generate_marked_metric_from_mesh(m, rotation_form, marked_metric_params);
    if (marked_metric_params.remove_symmetry) {
        vtx_reindex = extend_vtx_reindex(m, vtx_reindex);
    }

    return std::make_tuple(marked_metric, vtx_reindex);
}

VectorX generate_penner_coordinates(const Mesh<Scalar>& m)
{
    // Make copy of mesh delaunay
    Mesh<Scalar> m_copy = m;
    VectorX scale_factors;
    scale_factors.setZero(m.n_ind_vertices());
    bool use_ptolemy_flip = false;
    DelaunayStats del_stats;
    SolveStats<Scalar> solve_stats;
    ConformalIdealDelaunay<Scalar>::MakeDelaunay(
        m_copy,
        scale_factors,
        del_stats,
        solve_stats,
        use_ptolemy_flip);

    // Get flip sequence
    const auto& flip_sequence = del_stats.flip_seq;
    for (auto iter = flip_sequence.rbegin(); iter != flip_sequence.rend(); ++iter) {
        int flip_index = *iter;
        if (flip_index < 0) {
            flip_index = -flip_index - 1;
        }
        m_copy.flip_ccw(flip_index);
        m_copy.flip_ccw(flip_index);
        m_copy.flip_ccw(flip_index);
    }

    // Get metric coordinates from copy
    int num_halfedges = m.n_halfedges();
    VectorX metric_coords(num_halfedges);
    for (int h = 0; h < num_halfedges; ++h) {
        metric_coords[h] = 2.0 * log(m_copy.l[h]);
    }

    return metric_coords;
}

void generate_basis_loops(
    const Mesh<Scalar>& m,
    std::vector<std::unique_ptr<DualLoop>>& basis_loops,
    MarkedMetricParameters marked_metric_params,
    std::vector<int> marked_halfedges)
{
    // (optionally) generate dual loops on the surface
    // If the mesh is a trivial torus, don't add constraints
    int num_basis_loops = 0;
    if (is_trivial_torus(m))
    {
        spdlog::warn("Trivial torus seen");
    }
    if (marked_metric_params.remove_loop_constraints) return;
    if ((marked_metric_params.remove_trivial_torus) && (is_trivial_torus(m))) return;
    if (is_trivial_torus(m))
    {
        spdlog::warn("Adding constraints for trivial torus");
    }

    spdlog::debug("Adding holonomy constraints");
    HomologyBasisGenerator holonomy_basis_generator(m, 0, marked_metric_params.weighting);
    BoundaryBasisGenerator boundary_basis_generator(m);
    if (!marked_halfedges.empty()){
        boundary_basis_generator.avoid_marked_halfedges(marked_halfedges);
    }

    int num_homology_basis_loops = holonomy_basis_generator.n_homology_basis_loops();
    int num_basis_boundaries = boundary_basis_generator.n_basis_boundaries();
    spdlog::debug(
        "Adding {} homology and {} boundary constraints",
        num_homology_basis_loops,
        num_basis_boundaries);

    // Optionally remove some basis loops
    if (marked_metric_params.max_loop_constraints >= 0) {
        num_homology_basis_loops =
            std::min<int>(num_homology_basis_loops, marked_metric_params.max_loop_constraints);
    }

    // Optionally remove some boundary loops
    if (marked_metric_params.max_boundary_constraints >= 0) {
        num_basis_boundaries =
            std::min<int>(num_basis_boundaries, marked_metric_params.max_boundary_constraints);
    }

    // Initialize basis list loop and lambda to add loops
    basis_loops.reserve(num_basis_loops);
    auto add_basis_loop = [&](const std::vector<DualSegment>& basis_loop) {
        // increment count
        num_basis_loops++;

        // Use custom data structure for dual loop tracking
        if (marked_metric_params.use_connectivity) {
            basis_loops.push_back(std::make_unique<DualLoopConnectivity>(DualLoopConnectivity(basis_loop)));
        }
        // Use simpler list representation
        else {
            basis_loops.push_back(std::make_unique<DualLoopList>(DualLoopList(basis_loop)));
        }
    };

    // Add homology basis loops
    for (int i = 0; i < num_homology_basis_loops; ++i) {
        //auto basis_loop = holonomy_basis_generator.construct_homology_basis_loop(i);
        //add_basis_loop(build_dual_path_from_face_sequence(m, basis_loop));
        add_basis_loop(holonomy_basis_generator.construct_homology_basis_dual_path(i));
    }

    // Add boundary basis loops
    for (int i = 0; i < num_basis_boundaries; ++i) {
        auto basis_loop = boundary_basis_generator.construct_boundary_path_basis_loop(i);
        add_basis_loop(build_dual_path_from_face_sequence(m, basis_loop));
    }
}

Optimization::DiscreteMetric generate_discrete_metric(const Mesh<Scalar>& m) {
    // Build initial metric and target metric from edge lengths
    VectorX scale_factors;
    scale_factors.setZero(m.n_ind_vertices());
    bool is_hyperbolic = false;
    Optimization::InterpolationMesh<Scalar> interpolation_mesh(m, scale_factors, is_hyperbolic);

    // Get initial log length coordinates
    VectorX log_length_coords = interpolation_mesh.get_halfedge_metric_coordinates();
    return Optimization::DiscreteMetric(m, log_length_coords);
}


std::vector<Scalar> compute_kappa(
    const Mesh<Scalar>& discrete_metric,
    const VectorX& rotation_form,
    const std::vector<std::unique_ptr<DualLoop>>& basis_loops)
{
    // Compute the corner angles
    VectorX he2angle, he2cot;
    Optimization::corner_angles(discrete_metric, he2angle, he2cot);

    // Compute rotation angles along dual loops if loop constraints are needed
    int num_basis_loops = basis_loops.size();
    std::vector<Scalar> kappa(num_basis_loops);
    for (int i = 0; i < num_basis_loops; ++i) {
        // Compute field rotation and metric holonomy
        Scalar rotation = compute_dual_loop_rotation(discrete_metric, rotation_form, *basis_loops[i]);
        Scalar holonomy = compute_dual_loop_holonomy(discrete_metric, he2angle, *basis_loops[i]);

        // Constraint is the difference of the holonomy and rotation
        kappa[i] = holonomy - rotation;
        spdlog::debug("Holonomy constraint {} is {}", i, kappa[i]);
    }

    return kappa;
}

void make_free_interior(Mesh<Scalar>& m) {
    m.fixed_dof = std::vector<bool>(m.n_ind_vertices(), true);
    auto bd_vertices = find_boundary_vertices(m);
    for (int vi : bd_vertices) {
        m.fixed_dof[m.v_rep[vi]] = false;
    }

    // handle trivial interior case
    int num_bd_vertices = bd_vertices.size();
    if (num_bd_vertices == m.n_ind_vertices()) {
        m.fixed_dof[0] = true;
    }
}

void remove_symmetry(const Mesh<Scalar>& _m, Mesh<Scalar>& m) {
    m.Th_hat = std::vector<Scalar>(m.n_vertices(), 0.);
    m.fixed_dof = std::vector<bool>(m.n_vertices(), false);
    arange(m.n_vertices(), m.v_rep);
    int num_halfedges = m.n_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij) {
        m.type[hij] = 0;
        // m.R[hij] = 0;

        // split interior cones
        m.Th_hat[m.v_rep[m.to[hij]]] = _m.Th_hat[_m.v_rep[_m.to[hij]]] / 2.;
        if (_m.type[hij] == 2) {
            m.fixed_dof[m.v_rep[m.to[hij]]] = true;
        } else {
            m.fixed_dof[m.v_rep[m.to[hij]]] = _m.fixed_dof[_m.v_rep[_m.to[hij]]];
        }
    }

    std::vector<int> bd_vertices = find_boundary_vertices(_m);
    for (int vi : bd_vertices) {
        m.Th_hat[m.v_rep[vi]] = _m.Th_hat[_m.v_rep[vi]];
        m.fixed_dof[m.v_rep[vi]] = _m.fixed_dof[_m.v_rep[vi]];
    }
}

MarkedPennerConeMetric generate_marked_metric_from_mesh(
    const Mesh<Scalar>& _m,
    const VectorX& rotation_form,
    MarkedMetricParameters marked_metric_params,
    std::vector<int> marked_halfedges)
{
    // Optionally remove symmetry structure
    // TODO: Need to remake cone angles with half values
    Mesh<Scalar> m = _m;

    // Get initial log length coordinates
    Optimization::DiscreteMetric discrete_metric = generate_discrete_metric(m);
    VectorX log_length_coords = discrete_metric.get_metric_coordinates();

    // compute basis loops
    std::vector<std::unique_ptr<DualLoop>> basis_loops;
    generate_basis_loops(m, basis_loops, marked_metric_params, marked_halfedges);
    std::vector<Scalar> kappa = compute_kappa(discrete_metric, rotation_form, basis_loops);

    // optional modifications
    if (marked_metric_params.free_interior) make_free_interior(m);
    if (marked_metric_params.remove_symmetry) remove_symmetry(_m, m);

    // Build initial metric coordinates
    VectorX metric_coords;
    if (marked_metric_params.use_initial_zero) {
        int num_halfedges = m.n_halfedges();
        metric_coords = VectorX::Zero(num_halfedges);
    } else if (marked_metric_params.use_log_length) {
        metric_coords = log_length_coords;
    } else {
        metric_coords = generate_penner_coordinates(m);
    }

    return MarkedPennerConeMetric(m, metric_coords, basis_loops, kappa);
}

std::tuple<Mesh<Scalar>, std::vector<int>> generate_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Th_hat,
    std::vector<int> free_cones)
{
    // Convert VF mesh to halfedge
    bool fix_boundary = false;
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m = FV_to_double<Scalar>(
        V,
        F,
        uv,
        F_uv,
        Th_hat,
        vtx_reindex,
        indep_vtx,
        dep_vtx,
        v_rep,
        bnd_loops,
        free_cones,
        fix_boundary);
    return std::make_tuple(m, vtx_reindex);
}

std::tuple<MarkedPennerConeMetric, std::vector<int>, VectorX, std::vector<Scalar>>
infer_marked_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    bool use_intrinsic,
    MarkedMetricParameters marked_metric_params)
{
    auto [frame_field, field_Th_hat] = generate_cross_field(V, F);

    // Convert VF mesh to halfedge
    std::vector<int> vtx_reindex_mesh, indep_vtx, dep_vtx, v_rep, bnd_loops;
    std::vector<int> free_cones(0);
    bool fix_boundary = false;
    Mesh<Scalar> m = FV_to_double<Scalar>(
        V,
        F,
        V,
        F,
        field_Th_hat,
        vtx_reindex_mesh,
        indep_vtx,
        dep_vtx,
        v_rep,
        bnd_loops,
        free_cones,
        fix_boundary);

    // Generate rotation form and cones
    VectorX rotation_form;
    if (use_intrinsic) {
        FieldParameters field_params;
        rotation_form = generate_intrinsic_rotation_form(m, field_params);
    } else {
        rotation_form =
            generate_rotation_form_from_cross_field(m, vtx_reindex_mesh, V, F, frame_field);
    }
    bool has_boundary = bnd_loops.size() >= 1;
    std::vector<Scalar> Th_hat =
        generate_cones_from_rotation_form(m, vtx_reindex_mesh, rotation_form, has_boundary);

    // Generate marked mesh
    auto [marked_metric, vtx_reindex] =
        generate_marked_metric(V, F, V, F, Th_hat, rotation_form, free_cones, marked_metric_params);
    if (marked_metric_params.remove_symmetry) {
        vtx_reindex = extend_vtx_reindex(m, vtx_reindex);
    }

    return std::make_tuple(marked_metric, vtx_reindex, rotation_form, Th_hat);
}

std::tuple<VectorX, std::vector<Scalar>> generate_intrinsic_rotation_form(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const FieldParameters& field_params)
{
    // generate halfedge mesh
    std::vector<int> vtx_reindex;
    std::vector<int> free_cones(0);
    std::vector<Scalar> Th_hat = std::vector<Scalar>(V.rows(), 2 * M_PI);
    bool fix_boundary = false;
    bool use_discrete_metric = true;
    std::unique_ptr<DifferentiableConeMetric> cone_metric = Optimization::generate_initial_mesh(
        V,
        F,
        V,
        F,
        Th_hat,
        vtx_reindex,
        free_cones,
        fix_boundary,
        use_discrete_metric);

    // compute rotation form
    VectorX rotation_form = generate_intrinsic_rotation_form(*cone_metric, field_params);
    //VectorX rotation_form = generate_intrinsic_rotation_form(*cone_metric, vtx_reindex, V, field_params);

    // generate cones from the rotation form
    bool has_bd = (cone_metric->type[0] != 0);
    Th_hat = generate_cones_from_rotation_form(*cone_metric, vtx_reindex, rotation_form, has_bd);

    return std::make_tuple(rotation_form, Th_hat);
}

std::tuple<MarkedPennerConeMetric, VectorX, std::vector<Scalar>> generate_refined_marked_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    double min_angle,
    MarkedMetricParameters marked_metric_params)
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    // Get input geometry
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geometry;
    std::tie(mesh, geometry) = makeManifoldSurfaceMeshAndGeometry(V, F);

    // Flip edges to get the intrinsic Delaunay triangulation
    std::unique_ptr<IntrinsicTriangulation> intTri(
        new IntegerCoordinatesIntrinsicTriangulation(*mesh, *geometry));

    // Make the mesh delaunay and refine
    intTri->flipToDelaunay();
    intTri->delaunayRefine(min_angle);
    intTri->intrinsicMesh->compress();

    // Build NOB representation with lengths
    int num_halfedges = intTri->intrinsicMesh->nHalfedges();
    HalfedgeData<size_t> he_indices = intTri->intrinsicMesh->getHalfedgeIndices();
    std::vector<int> next_he(num_halfedges, -1);
    std::vector<int> opp(num_halfedges, -1);
    std::vector<int> bnd_loops = {};
    std::vector<Scalar> l(num_halfedges, 0.0);
    intTri->requireEdgeLengths();
    for (Halfedge he : intTri->intrinsicMesh->halfedges()) {
        size_t he_index = he_indices[he];
        next_he[he_index] = he_indices[he.next()];
        opp[he_index] = he_indices[he.twin()];
        l[he_index] = intTri->edgeLengths[he.edge()];
    }

    // Build the connectivity arrays from the NOB arrays
    Connectivity C;
    NOB_to_connectivity(next_he, opp, bnd_loops, C);

    // Create trivial reflection information
    std::vector<char> type(num_halfedges, 0);
    std::vector<int> R(num_halfedges, 0);

    // Create a halfedge structure for the mesh
    int num_vertices = C.out.size();
    Mesh<Scalar> m;
    m.n = C.n;
    m.to = C.to;
    m.f = C.f;
    m.h = C.h;
    m.out = C.out;
    m.opp = C.opp;
    m.type = type;
    m.type_input = type;
    m.R = R;
    m.l = l;
    m.Th_hat = std::vector<Scalar>(num_vertices, 2 * M_PI);
    m.v_rep = range(0, num_vertices);
    m.fixed_dof = std::vector<bool>(num_vertices, false);
    m.fixed_dof[0] = true;

    // Get rotation form and corresponding cones
    FieldParameters field_params;
    VectorX rotation_form = generate_intrinsic_rotation_form(m, field_params);
    std::vector<Scalar> Th_hat = generate_cones_from_rotation_form(m, rotation_form);
    m.Th_hat = Th_hat;

    // Check for invalid cones
    if (!validate_cones(m)) {
        spdlog::info("Fixing invalid cones");
        fix_cones(m);
    }

    // Set cones and check Guass Bonnet
    GaussBonnetCheck(m);

    // Get initial marked mesh for optimization
    auto marked_metric = generate_marked_metric_from_mesh(m, rotation_form, marked_metric_params);

    return std::make_tuple(marked_metric, rotation_form, Th_hat);
}

std::tuple<SimilarityPennerConeMetric, std::vector<int>> generate_similarity_metric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<Scalar>& Th_hat,
    const VectorX& rotation_form,
    std::vector<int> free_cones,
    MarkedMetricParameters marked_metric_params)
{
    // Convert VF mesh to halfedge
    bool fix_boundary = false;
    std::vector<int> vtx_reindex, indep_vtx, dep_vtx, v_rep, bnd_loops;
    Mesh<Scalar> m = FV_to_double<Scalar>(
        V,
        F,
        uv,
        F_uv,
        Th_hat,
        vtx_reindex,
        indep_vtx,
        dep_vtx,
        v_rep,
        bnd_loops,
        free_cones,
        fix_boundary);

    // Use halfedge method with appropriate parameters
    SimilarityPennerConeMetric similarity_metric =
        generate_similarity_metric_from_mesh(m, rotation_form, marked_metric_params);

    return std::make_tuple(similarity_metric, vtx_reindex);
}

SimilarityPennerConeMetric generate_similarity_metric_from_mesh(
    const Mesh<Scalar>& m,
    const VectorX& rotation_form,
    MarkedMetricParameters marked_metric_params)
{
    // Generate the base underlying marked metric
    MarkedPennerConeMetric marked_metric =
        generate_marked_metric_from_mesh(m, rotation_form, marked_metric_params);

    // Use initial zero harmonic form coordinates (corresponding to a metric)
    VectorX harmonic_form_coords = VectorX::Zero(marked_metric.n_homology_basis_loops());

    return SimilarityPennerConeMetric(
        m,
        marked_metric.get_reduced_metric_coordinates(),
        marked_metric.get_homology_basis_loops(),
        marked_metric.kappa_hat,
        harmonic_form_coords);
}

void regularize_metric(MarkedPennerConeMetric& marked_metric, double max_triangle_quality)
{
    // Get initial mesh quality
    VectorX reduced_metric_coords = marked_metric.get_reduced_metric_coordinates();
    VectorX mesh_quality = compute_mesh_quality(marked_metric);
    spdlog::info("Initial quality is {}", mesh_quality.maxCoeff());

    // Get average
    int num_edges = marked_metric.n_edges();
    Scalar average_initial_coord = reduced_metric_coords.mean();
    spdlog::info("Average metric coordinate is {}", average_initial_coord);

    // Regularize
    bool changed = false;
    while (mesh_quality.maxCoeff() > max_triangle_quality) {
        reduced_metric_coords = 0.9 * reduced_metric_coords;
        marked_metric.change_metric(marked_metric, reduced_metric_coords);
        mesh_quality = compute_mesh_quality(marked_metric);
        spdlog::info("Quality is {}", mesh_quality.maxCoeff());
        changed = true;
    }

    // Make sure average is unchanged if regularized
    if (changed) {
        Scalar difference = average_initial_coord - reduced_metric_coords.mean();
        reduced_metric_coords += VectorX::Constant(num_edges, difference);
        marked_metric.change_metric(marked_metric, reduced_metric_coords);
        mesh_quality = compute_mesh_quality(marked_metric);
        spdlog::info("Final quality is {}", mesh_quality.maxCoeff());
        spdlog::info("Final average is {}", reduced_metric_coords.mean());
    }
}

void optimize_triangle_quality(MarkedPennerConeMetric& marked_metric, double max_triangle_quality)
{
    std::vector<int> flip_seq = {};
    marked_metric.make_discrete_metric();
    flip_seq = marked_metric.get_flip_sequence();
    VectorX mesh_quality = compute_mesh_quality(marked_metric);
    for (auto iter = flip_seq.rbegin(); iter != flip_seq.rend(); ++iter) {
        int h = *iter;
        spdlog::trace("Flipping {} cw", h);
        marked_metric.flip_ccw(h, true);
        marked_metric.flip_ccw(h, true);
        marked_metric.flip_ccw(h, true);
    }
    spdlog::info("Initial quality is {}", mesh_quality.maxCoeff());

    // Regularize until quality is sufficiently low
    while (mesh_quality.maxCoeff() > max_triangle_quality) {
        marked_metric.make_discrete_metric();
        flip_seq = marked_metric.get_flip_sequence();
        mesh_quality = compute_mesh_quality(marked_metric);
        spdlog::info("New quality is {}", mesh_quality.maxCoeff());

        LogTriangleQualityEnergy energy(marked_metric);
        // FIXME TriangleQualityEnergy energy(marked_metric);
        VectorX gradient = energy.EnergyFunctor::gradient(marked_metric);
        spdlog::info("Gradient in range [{}, {}]", gradient.minCoeff(), gradient.maxCoeff());
        gradient /= (gradient.norm() + 1e-10);
        VectorX reduced_metric_coords = marked_metric.get_reduced_metric_coordinates();
        marked_metric.change_metric(marked_metric, reduced_metric_coords - gradient, true, false);

        // Undo any flips to make Delaunay
        for (auto iter = flip_seq.rbegin(); iter != flip_seq.rend(); ++iter) {
            int h = *iter;
            spdlog::trace("Flipping {} cw", h);
            marked_metric.flip_ccw(h, true);
            marked_metric.flip_ccw(h, true);
            marked_metric.flip_ccw(h, true);
        }
    }
}

} // namespace Holonomy
} // namespace Penner