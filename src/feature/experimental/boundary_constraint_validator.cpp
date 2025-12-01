#include "feature/experimental/boundary_constraint_validator.h"

#include "util/boundary.h"
#include "holonomy/core/viewer.h"
#include "feature/feature/features.h"
#include "feature/core/component_mesh.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#endif

namespace Penner {
namespace Feature {

BoundaryConstraintValidator::BoundaryConstraintValidator(Mesh<Scalar> m)
    : BoundaryConstraintGenerator(m)
    , m_path_start_segment({})
    , m_path_end_segment({})
    , m_path_reverse_start_segment({})
    , m_path_reverse_end_segment({})
    , m_path_holonomy({})
{
    m_path = std::vector<int>(count_segments(), -1);
    m_reverse_path = std::vector<int>(count_segments(), -1);
}

void BoundaryConstraintValidator::add_boundary_path_segments(
    const std::vector<std::unique_ptr<DualLoop>>& basis_loops,
    const std::vector<Scalar>& kappa_hat)
{
    spdlog::debug("Marking boundary paths");
    const auto& m = get_mesh();
    int num_basis_loops = basis_loops.size();

    for (int i = 0; i < num_basis_loops; ++i) {
        // get halfedges at start and end of dual path
        int start_h = -1;
        int end_h = -1;
        int current_type = m.type[m.opp[(*basis_loops[i]->begin())[0]]];
        for (const auto& dual_segment : *basis_loops[i]) {
            int h0 = dual_segment[0];
            if ((m.type[h0] == 1) && (current_type == 2)) {
                start_h = h0;
                current_type = 1;
            }
            if ((m.type[h0] == 2) && (current_type == 1)) {
                end_h = m.opp[h0];
                current_type = 2;
            }
        }

        // skip interior loops
        if ((start_h < 0) || (end_h < 0)) continue;
        spdlog::debug("Adding loop from {} to {}", start_h, end_h);

        // split segments
        //  <- s11 -  | | <- s10 -
        //            ^ v
        //   - s00 -> | |  - s01 ->
        int s00 = segment(start_h);
        int s10 = segment(end_h);
        int s01 = split_segment(s00);
        int s11 = split_segment(s10);

        // add forward path
        m_path[s00] = count_paths();
        m_reverse_path[s10] = count_paths();
        m_path_start_segment.push_back(s00);
        m_path_end_segment.push_back(s11);
        m_path_reverse_start_segment.push_back(s10);
        m_path_reverse_end_segment.push_back(s01);
        m_path_holonomy.push_back(kappa_hat[i]);
    }
}

int BoundaryConstraintValidator::split_segment(int segment_index)
{
    m_path.push_back(-1);
    m_reverse_path.push_back(-1);
    return BoundaryConstraintGenerator::split_segment(segment_index);
}

void BoundaryConstraintValidator::add_feature(
    int feature_index,
    Vector2& d,
    Scalar& total_curvature,
    std::vector<Eigen::Triplet<Scalar>>& tripletList) const
{
    spdlog::debug("Feature start direction: ({}, {})", d[0], d[1]);
    const Mesh<Scalar>& m = get_mesh();
    Matrix2x2 R;
    SegmentIterator iter = get_feature_iterator(feature_index);
    for (; !iter.is_end(); ++iter) {
        // get edge tangent vector for the current halfedge
        int h = *iter;
        tripletList.push_back(T(0, segment(h), d[0]));
        tripletList.push_back(T(1, segment(h), d[1]));

        // rotate direction by the angle at the tip of the halfedge
        int vi = m.to[h];
        Scalar theta = m.Th_hat[m.v_rep[vi]] / 2. - M_PI;
        total_curvature += theta;
        R = compute_rotation(theta);
        d = R * d;
    }

    spdlog::debug("Feature end direction: ({}, {})", d[0], d[1]);
}

void BoundaryConstraintValidator::add_free_connection(
    const std::array<int, 5>& virtual_segments,
    bool reverse,
    Vector2& d,
    std::vector<Vector2>& segment_directions,
    std::vector<int>& segment_indices) const
{
    spdlog::debug("Connection start direction: ({}, {})", d[0], d[1]);

    // rotate direction by -pi/2 (pointing into interior w/ ccw orientation)
    Scalar theta = -M_PI / 2.;
    Matrix2x2 R = compute_rotation(theta);
    d = R * d;

    if (!reverse) {
        spdlog::debug("Adding forward connection");
        for (auto itr = virtual_segments.begin(); itr != virtual_segments.end(); ++itr) {
            // add virtual segment length
            int segment = *itr;
            segment_directions.push_back(d);
            segment_indices.push_back(segment);

            // rotate direction by -pi/2
            theta = -M_PI / 2.;
            R = compute_rotation(theta);
            d = R * d;
        }
    } else {
        spdlog::debug("Adding reverse connection");
        for (auto itr = virtual_segments.rbegin(); itr != virtual_segments.rend(); ++itr) {
            // add virtual segment length
            int segment = *itr;
            segment_directions.push_back(d);
            segment_indices.push_back(segment);

            // rotate direction by pi/2
            theta = M_PI / 2.;
            R = compute_rotation(theta);
            d = R * d;
        }

        // correct final direction
        theta = M_PI;
        R = compute_rotation(theta);
        d = R * d;
    }

    spdlog::debug("Connection end direction: ({}, {})", d[0], d[1]);
}

void BoundaryConstraintValidator::add_fixed_connection(
    const std::vector<int>& virtual_segments,
    bool is_counterclockwise,
    bool reverse,
    Vector2& d,
    std::vector<Vector2>& segment_directions,
    std::vector<int>& segment_indices) const
{
    spdlog::debug("Connection start direction: ({}, {})", d[0], d[1]);

    // rotate direction by pi/2 (pointing into interior w/ ccw orientation)
    Matrix2x2 R = compute_rotation(M_PI / 2.);
    d = R * d;

    // add holonomy rotations
    Scalar theta = (is_counterclockwise) ? (M_PI / 2.) : (-M_PI / 2.);
    if (!reverse) {
        spdlog::debug("Adding forward connection");
        for (auto itr = virtual_segments.begin(); itr != virtual_segments.end(); ++itr) {
            // add virtual segment length
            int segment = *itr;
            segment_directions.push_back(d);
            segment_indices.push_back(segment);

            // rotate direction by pi/2 (with orientation determined by the index sign)
            R = compute_rotation(theta);
            d = R * d;
        }
        // correct direction
        R = compute_rotation(-theta);
        d = R * d;

    } else {
        spdlog::debug("Adding reverse connection");
        for (auto itr = virtual_segments.rbegin(); itr != virtual_segments.rend(); ++itr) {
            // add virtual segment length
            int segment = *itr;
            segment_directions.push_back(d);
            segment_indices.push_back(segment);

            // rotate direction by pi/2 (with reverse orientation of forward connection)
            R = compute_rotation(-theta);
            d = R * d;
        }

        // correct final direction
        R = compute_rotation(theta);
        d = R * d;
    }

    // rotate into interior
    R = compute_rotation(M_PI / 2.);
    d = R * d;

    spdlog::debug("Connection end direction: ({}, {})", d[0], d[1]);
}

// helper function for per-component validity checks
// TODO: currently picks an arbitrary component; should do for an indexed component if keep
std::tuple<int, std::vector<Vector2>, std::vector<int>>
BoundaryConstraintValidator::build_component_layout() const
{
    const Mesh<Scalar>& m = get_mesh();
    int num_segments = count_segments();
    int num_features = count_features();
    std::vector<Vector2> segment_directions;
    std::vector<int> segment_indices;
    int variable_count = num_segments;
    int num_paths = count_paths();
    spdlog::debug("Adding {} features with {} segments", num_features, num_segments);

    // build virtual segments for paths
    std::vector<std::vector<int>> virtual_segments(num_paths);
    for (int i = 0; i < num_paths; ++i) {
        int num_turns = (int)(abs(round(path_holonomy(i) / (M_PI / 2.))));
        int num_segments = num_turns + 1;
        virtual_segments[i].resize(num_segments);
        std::iota(virtual_segments[i].begin(), virtual_segments[i].end(), variable_count);
        variable_count += virtual_segments[i].size();
    }

    // initialize direction tracking
    Vector2 d = {1, 0};
    Scalar theta;
    Matrix2x2 R;

    // iterate from arbitrary starting segment
    int start_segment = 0;
    int segment_index = start_segment;
    do {
        // add segment length to constraint
        segment_directions.push_back(d);
        segment_indices.push_back(segment_index);

        // add forward connection if start segment
        if (is_path_start(segment_index)) {
            int path_index = path(segment_index);
            bool is_counterclockwise = (path_holonomy(path_index) > 0);
            bool reverse = false;
            add_fixed_connection(
                virtual_segments[path_index],
                is_counterclockwise,
                reverse,
                d,
                segment_directions,
                segment_indices);
            segment_index = path_end_segment(path_index);
        }
        // add reverse connection if reverse start segment
        else if (is_reverse_path_start(segment_index)) {
            int path_index = reverse_path(segment_index);
            bool is_counterclockwise = (path_holonomy(path_index) > 0);
            bool reverse = true;
            add_fixed_connection(
                virtual_segments[path_index],
                is_counterclockwise,
                reverse,
                d,
                segment_directions,
                segment_indices);
            segment_index = path_reverse_end_segment(path_index);
        }
        // add cone rotation if end of feature
        else if (is_feature_end(segment_index)) {
            int h = halfedge(segment_index);
            int vi = m.to[h];
            theta = m.Th_hat[m.v_rep[vi]] / 2. - M_PI;
            R = compute_rotation(theta);
            d = R * d;
            segment_index = feature_start(next_feature(feature(segment_index)));
        }
        // just iterate to next segment otherwise
        else {
            segment_index = next_segment(segment_index);
        }
    } while (segment_index != start_segment);

    // add interior cones
    std::vector<Scalar> interior_cones = enumerate_interior_cones();
    for (Scalar cone_angle : interior_cones) {
        // get new virtual segments for cone connection
        std::array<int, 5> cone_virtual_segments;
        std::iota(cone_virtual_segments.begin(), cone_virtual_segments.end(), variable_count);
        variable_count += cone_virtual_segments.size();

        // add forward connection
        add_free_connection(cone_virtual_segments, false, d, segment_directions, segment_indices);

        // rotate direction by the cone angle defect
        Scalar theta = cone_angle - (2. * M_PI);
        Matrix2x2 R = compute_rotation(theta);
        d = R * d;

        // add reverse connection
        add_free_connection(cone_virtual_segments, true, d, segment_directions, segment_indices);
    }

    return std::make_tuple(variable_count, segment_directions, segment_indices);
}

void BoundaryConstraintValidator::view_component_layout(const VectorX& ell) const
{
    auto [variable_count, segment_directions, segment_indices] = build_component_layout();

    // build layout vertices
    int num_layout_edges = segment_directions.size();
    Eigen::MatrixXd layout_vertices(num_layout_edges, 2);
    Eigen::MatrixXd inflated_vertices(num_layout_edges, 3);
    Eigen::MatrixXi layout_edges(num_layout_edges, 2);
    Vector2 v = {0, 0};
    for (int i = 0; i < num_layout_edges; ++i) {
        // layout current vertex
        layout_vertices.row(i) << (double)(v[0]), (double)(v[1]);
        inflated_vertices.row(i) << (double)(v[0]), (double)(v[1]), (0.1 * i);
        layout_edges.row(i) << i, (i + 1) % num_layout_edges;

        // update v
        Vector2 d = segment_directions[i];
        Scalar l = ell[segment_indices[i]];
        v = v + l * d;
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerPointCloud2D("component vertices", layout_vertices);
    polyscope::registerCurveNetwork2D("component layout", layout_vertices, layout_edges);
    polyscope::registerCurveNetwork("inflated layout", inflated_vertices, layout_edges);
    polyscope::show();
#endif
}

std::tuple<MatrixX, VectorX> BoundaryConstraintValidator::build_component_validity_system() const
{
    auto [variable_count, segment_directions, segment_indices] = build_component_layout();

    // build system validity matrix for the layout
    int num_layout_edges = segment_directions.size();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_layout_edges);
    for (int i = 0; i < num_layout_edges; ++i) {
        for (int j : {0, 1}) {
            tripletList.push_back(T(j, segment_indices[i], segment_directions[i][j]));
        }
    }

    // create the matrix from the triplets
    MatrixX system_matrix;
    system_matrix.resize(2, variable_count);
    system_matrix.reserve(tripletList.size());
    system_matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    // sum of oriented lengths should be zero
    VectorX ell = VectorX::Zero(2);

    return std::make_tuple(system_matrix, ell);
}

std::tuple<MatrixX, VectorX> BoundaryConstraintValidator::build_boundary_validity_system() const
{
    int num_segments = count_segments();
    int num_features = count_features();
    const Mesh<Scalar>& m = get_mesh();
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_segments);
    std::vector<Scalar> ell = {};
    int constraint_count = 0;
    int variable_count = num_segments;
    std::vector<bool> is_segment_seen(num_segments, false);
    for (int i = 0; i < num_segments; ++i) {
        // only process each segment once
        if (is_segment_seen[i]) continue;
        is_segment_seen[i] = true;

        if (pair(i) >= 0) {
            is_segment_seen[pair(i)] = true; // mark pair as seen

            // add equality constraint
            tripletList.push_back(T(constraint_count, i, 1.));
            tripletList.push_back(T(constraint_count, pair(i), -1.));
            ell.push_back(0);
            ++constraint_count;
            spdlog::debug("Setting constraint {} - {} = {}", i, pair(i), ell.back());

        } else {
            tripletList.push_back(T(constraint_count, i, 1.));
            ell.push_back(get_target_length(i));
            ++constraint_count;
            spdlog::debug("Setting constraint {} = {}", i, ell.back());
        }
    }

    Vector2 d = {1, 0};

    std::vector<bool> is_feature_seen(num_features, false);
    for (int i = 0; i < num_features; ++i) {
        if (is_feature_seen[i]) continue; // skip seen features

        // iterate over feature component to build polygon closure condition
        int feature_index = i;
        Scalar total_curvature = 0.;
        int h, vi;
        Scalar theta = 0.;
        Matrix2x2 R;
        do {
            SegmentIterator iter = get_feature_iterator(feature_index);
            for (; !iter.is_end(); ++iter) {
                // get edge tangent vector for the current halfedge
                h = *iter;
                tripletList.push_back(T(constraint_count + 0, segment(h), d[0]));
                tripletList.push_back(T(constraint_count + 1, segment(h), d[1]));

                // rotate direction by the angle at the tip of the halfedge
                vi = m.to[h];
                theta = m.Th_hat[m.v_rep[vi]] / 2. - M_PI;
                total_curvature += theta;
                R = compute_rotation(theta);
                d = R * d;
            }
            is_feature_seen[feature_index] = true;
            feature_index = next_feature(feature_index);
        } while (feature_index != i);

        // add additional cone variable if total geodesic curvature is not 2 pi
        if (!float_equal<Scalar>(total_curvature, -2. * M_PI)) {
            spdlog::debug("Adding virtual edge for interior cones");
            // add virtual edge with angle theta/2
            // note that the use of theta/2 is somewhat arbitrary
            // we could use any angle in [0, theta]
            int virtual_segment = variable_count;
            R = compute_rotation(-theta / 2.);
            d = R * d;
            tripletList.push_back(T(constraint_count + 0, virtual_segment, d[0]));
            tripletList.push_back(T(constraint_count + 1, virtual_segment, d[1]));

            // add virtual edge with missing geodesic curvature angle
            Scalar curvature_defect = -2 * M_PI - total_curvature;
            spdlog::debug("Curvature defect of {}", curvature_defect);
            R = compute_rotation(curvature_defect);
            d = R * d;
            tripletList.push_back(T(constraint_count + 0, virtual_segment, d[0]));
            tripletList.push_back(T(constraint_count + 1, virtual_segment, d[1]));

            ++variable_count;
        }

        // closure constraint: sum of edge lengths should be zero
        ell.push_back(0);
        ell.push_back(0);
        constraint_count += 2;
    }

    // create the matrix from the triplets
    MatrixX system_matrix;
    system_matrix.resize(constraint_count, variable_count);
    system_matrix.reserve(tripletList.size());
    system_matrix.setFromTriplets(tripletList.begin(), tripletList.end());

    // convert ell to an eigen type
    VectorX ell_eigen;
    convert_std_to_eigen_vector(ell, ell_eigen);

    return std::make_tuple(system_matrix, ell_eigen);
}

} // namespace Feature
} // namespace Penner