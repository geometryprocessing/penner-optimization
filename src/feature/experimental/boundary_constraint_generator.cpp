#include "feature/experimental/boundary_constraint_generator.h"

#include "util/boundary.h"
#include "holonomy/core/viewer.h"
#include "feature/feature/features.h"
#include "feature/core/component_mesh.h"
#include "feature/feature/gluing.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#endif

namespace Penner {
namespace Feature {

BoundaryConstraintGenerator::BoundaryConstraintGenerator(Mesh<Scalar> m)
    : m_mesh(m)
    , m_out({})
{
    // find starting halfedges for all all boundary components
    std::vector<int> boundary_components = find_boundary_components(m);

    // build trivial features
    int num_features = boundary_components.size();
    int num_segments = 0; // will find incrementally
    int num_halfedges = m.n_halfedges();
    m_feature_start.resize(num_features);
    m_feature_end.resize(num_features);
    m_next_feature.resize(num_features);
    m_prev_feature.resize(num_features);
    m_to.resize(num_features);
    m_segment.resize(num_halfedges);
    for (int i = 0; i < num_features; ++i) {
        // build trivial loop
        m_next_feature[i] = i;
        m_prev_feature[i] = i;
        m_to[i] = -1;

        // build feature data
        m_feature_start[i] = num_segments;
        int h_start = boundary_components[i];
        std::vector<int> boundary_component = build_boundary_component(m, h_start);
        for (int h : boundary_component) {
            // mark new segment index for the boundary halfedge
            m_segment[h] = num_segments;

            // add segment topology data
            // WARNING: need to overwrite next and prev later
            m_feature.push_back(i);
            m_halfedge.push_back(h);
            m_next_segment.push_back(num_segments + 1);
            m_prev_segment.push_back(num_segments - 1);
            m_target_length.push_back(m.l[h]);
            m_pair.push_back(-1);
            ++num_segments;
        }
        m_feature_end[i] = num_segments - 1;

        // mark first and last segments as end of chain
        m_prev_segment[m_feature_start[i]] = -1;
        m_next_segment[m_feature_end[i]] = -1;
    }
}

void BoundaryConstraintGenerator::pair_boundary_edges(
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& V_map)
{
    // generate boundary identification
    const auto& m = get_mesh();
    auto [bd_pairs, boundary_he, he2bd] = generate_boundary_pairs(m, vtx_reindex, V_map);

    // pair each boundary segment
    for (const auto& bd_pair : bd_pairs) {
        int hij = bd_pair.first;
        int hji = bd_pair.second;
        m_pair[segment(hij)] = segment(hji);
        m_pair[segment(hji)] = segment(hij);
    }
}

void BoundaryConstraintGenerator::add_junction_after_segment(int segment_index)
{
    int feature_index = feature(segment_index);
    int next_feature_index = next_feature(feature_index);
    int next_segment_index = next_segment(segment_index);
    int end = feature_end(feature_index);
    int to_junction_index = to(feature_index);

    // add junction to trivial loop
    if (to_junction_index == -1) {
        int new_junction_index = count_junctions();
        m_to[feature_index] = new_junction_index;
        m_out.push_back(feature_index);

        int h_start = halfedge(segment_index);
        std::vector<int> boundary_component = build_boundary_component(get_mesh(), h_start);
        int num_bd = boundary_component.size();
        for (int i = 0; i < num_bd - 1; ++i) {
            int hij = boundary_component[i];
            int hjk = boundary_component[i + 1];
            m_next_segment[m_segment[hij]] = m_segment[hjk];
        }
        for (int i = 1; i < num_bd + 1; ++i) {
            int hli = boundary_component[(i - 1) % num_bd];
            int hij = boundary_component[i % num_bd];
            int hjk = boundary_component[(i + 1) % num_bd];
            m_next_segment[m_segment[hij]] = m_segment[hjk];
            m_prev_segment[m_segment[hij]] = m_segment[hli];
        }
        m_feature_start[feature_index] = m_segment[boundary_component[1 % num_bd]];
        m_feature_end[feature_index] = m_segment[boundary_component[0]];
        m_prev_segment[m_feature_start[feature_index]] = -1;
        m_next_segment[m_feature_end[feature_index]] = -1;

        return;
    }

    // get new indices
    int new_feature_index = count_features();
    int new_junction_index = count_junctions();

    // split segment connectivity
    m_next_segment[segment_index] = -1;
    m_prev_segment[next_segment_index] = -1;

    // add new feature
    m_feature_start.push_back(next_segment_index);
    m_feature_end.push_back(end);
    m_next_feature.push_back(next_feature_index);
    m_prev_feature.push_back(feature_index);
    m_to.push_back(to_junction_index);

    // modify current feature
    m_feature_end[feature_index] = segment_index;
    m_next_feature[feature_index] = new_feature_index;
    m_to[feature_index] = new_junction_index;

    // add junction
    m_out.push_back(new_feature_index);

    // mark new feature segments
    int curr_segment_index = next_segment_index;
    m_feature[curr_segment_index] = new_feature_index;
    do {
        curr_segment_index = m_next_segment[curr_segment_index];
        m_feature[curr_segment_index] = new_feature_index;
    } while (curr_segment_index != end);
}

int BoundaryConstraintGenerator::split_segment(int segment_index)
{
    // get relevant indices
    int feature_index = feature(segment_index);
    int next_segment_index = next_segment(segment_index);
    int end = feature_end(feature_index);
    int halfedge_index = halfedge(segment_index);

    // add new segment
    int new_segment_index = count_segments();
    m_next_segment[segment_index] = new_segment_index;
    m_next_segment.push_back(next_segment_index);
    m_prev_segment.push_back(segment_index);
    m_halfedge.push_back(halfedge_index);
    m_feature.push_back(feature_index);

    // check if end segment has changed
    if (end == segment_index) {
        m_feature_end[feature_index] = new_segment_index;
    }

    return new_segment_index;
}

void BoundaryConstraintGenerator::mark_cones_as_junctions()
{
    spdlog::debug("Marking cones as junctions");
    const auto& m = get_mesh();

    int num_segments = count_segments();
    for (int i = 0; i < num_segments; ++i) {
        int hij = halfedge(i); // constant even with feature modification
        int vj = m.v_rep[m.to[hij]];
        if (!float_equal<Scalar>(m.Th_hat[vj], 2 * M_PI)) {
            spdlog::debug("Adding junction after segment {}", i);
            add_junction_after_segment(i);
        }
    }
}

std::vector<Scalar> BoundaryConstraintGenerator::enumerate_feature_cones(int feature_index) const
{
    spdlog::debug("Enumerating cones on feature {}", feature_index);
    const auto& m = get_mesh();
    std::vector<Scalar> cones = {};

    // iterate over feature, counting cones
    SegmentIterator iter = get_feature_iterator(feature_index);
    for (; !iter.is_end(); ++iter) {
        int hij = *iter;
        int vj = m.v_rep[m.to[hij]];
        Scalar cone_angle = m.Th_hat[vj];
        if (!float_equal<Scalar>(cone_angle, 2 * M_PI)) {
            cones.push_back(cone_angle);
        }
    }

    return cones;
}

std::vector<Scalar> BoundaryConstraintGenerator::enumerate_interior_cones() const
{
    spdlog::debug("Enumerating interior cones");
    const auto& m = get_mesh();
    std::vector<Scalar> cones = {};

    // mark boundary vertices
    int num_vertices = m.n_ind_vertices();
    int num_segments = count_segments();
    std::vector<bool> is_boundary_vertex(num_vertices, false);
    for (int i = 0; i < num_segments; ++i) {
        int hij = halfedge(i);
        int vj = m.v_rep[m.to[hij]];
        is_boundary_vertex[vj] = true;
    }

    // iterate over vertices, skipping boundary
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (is_boundary_vertex[vi]) continue;

        Scalar cone_angle = m.Th_hat[vi];
        if (!float_equal<Scalar>(cone_angle, 4 * M_PI)) {
            cones.push_back(cone_angle / 2.);
        }
    }

    return cones;
}

void BoundaryConstraintGenerator::distribute_feature_length(
    int feature_index,
    Scalar target_feature_length)
{
    const auto& m = get_mesh();
    SegmentIterator iter = get_feature_iterator(feature_index);
    Scalar feature_length = compute_feature_length(feature_index);
    for (; !iter.is_end(); ++iter) {
        // get length of mesh halfedge corresponding to the segment
        int h = *iter;
        Scalar target_segment_length = (target_feature_length / feature_length) * m.l[h];
        set_target_length(m_segment[h], target_segment_length);
    }
}

void BoundaryConstraintGenerator::set_uniform_feature_lengths(Scalar target_feature_length)
{
    int num_features = count_features();
    Scalar total_feature_length = 0.;
    for (int i = 0; i < num_features; ++i) {
        total_feature_length += compute_feature_length(i);
    }
    if (target_feature_length == 0.) {
        target_feature_length = total_feature_length / num_features;
    }
    for (int i = 0; i < num_features; ++i) {
        distribute_feature_length(i, target_feature_length);
    }
}

Scalar BoundaryConstraintGenerator::compute_feature_length(int feature_index) const
{
    // get mesh and iterator for the feature
    const auto& m = get_mesh();
    SegmentIterator iter = get_feature_iterator(feature_index);

    // iterate to compute feature length
    Scalar feature_length = 0.;
    for (; !iter.is_end(); ++iter) {
        // get length of mesh halfedge corresponding to the segment
        int h = *iter;
        feature_length += m.l[h];
    }

    return feature_length;
}

std::vector<int> BoundaryConstraintGenerator::compute_feature_halfedges(int feature_index) const
{
    std::vector<int> feature_halfedges = {};
    SegmentIterator iter = get_feature_iterator(feature_index);
    for (; !iter.is_end(); ++iter) {
        // get halfedge corresponding to the segment
        int h = *iter;
        feature_halfedges.push_back(h);
    }

    return feature_halfedges;
}

void BoundaryConstraintGenerator::view(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex)
{
    const Mesh<Scalar>& m = get_mesh();
    auto [V_double, F, F_halfedge] = Optimization::generate_doubled_mesh(V, m, vtx_reindex);

    // build feature coloring
    int num_halfedges = m.n_halfedges();
    int num_features = count_features();
    VectorX is_feature_halfedge = VectorX::Zero(num_halfedges);
    for (int i = 0; i < num_features; ++i) {
        std::vector<int> feature_halfedges = compute_feature_halfedges(i);
        for (int h : feature_halfedges) {
            is_feature_halfedge(h) = i + 1;
        }
    }
    VectorX FV_feature_halfedges = Optimization::generate_FV_halfedge_data(F_halfedge, is_feature_halfedge);

    auto [cone_positions, cone_values] = Optimization::generate_cone_vertices(V, vtx_reindex, m);

#ifdef ENABLE_VISUALIZATION
    polyscope::init();

    // add mesh
    std::string mesh_handle = "boundary constraint generator";
    polyscope::registerSurfaceMesh(mesh_handle, V_double, F);
    polyscope::getSurfaceMesh(mesh_handle)
        ->addHalfedgeScalarQuantity(
            "features",
            convert_scalar_to_double_vector(FV_feature_halfedges))
        ->setEnabled(true);

    // add cones
    polyscope::registerPointCloud(mesh_handle + "_cones", cone_positions);
    polyscope::getPointCloud(mesh_handle + "_cones")
        ->addScalarQuantity("index", cone_values)
        ->setColorMap("coolwarm")
        ->setMapRange({-M_PI, M_PI})
        ->setEnabled(true);

    polyscope::show();

#endif
}

std::tuple<MatrixX, VectorX> BoundaryConstraintGenerator::build_boundary_constraint_system() const
{
    int num_segments = count_segments();
    int num_halfedges = count_halfedges();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_segments);
    VectorX ell(num_segments);
    for (int i = 0; i < num_segments; ++i) {
        int h = halfedge(i);
        assert(is_valid_halfedge_index(h));
        tripletList.push_back(T(i, h, 1.));
        ell[i] = 2. * log(get_target_length(i));
        spdlog::debug("Setting constraint {} = {}", i, ell[i]);
    }

    // Create the matrix from the triplets
    MatrixX system_matrix;
    system_matrix.resize(num_segments, num_halfedges);
    system_matrix.reserve(tripletList.size());
    system_matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    return std::make_tuple(system_matrix, ell);
}

} // namespace Feature
} // namespace Penner