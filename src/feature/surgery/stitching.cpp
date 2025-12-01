#include "feature/surgery/stitching.h"

#include "holonomy/core/viewer.h"
#include "optimization/parameterization/refinement.h"
#include "optimization/parameterization/layout.h"
#include "util/io.h"
#include "util/vector.h"
#include "util/vf_mesh.h"

#include "igl/remove_unreferenced.h"
#include "igl/boundary_loop.h"

#ifdef ENABLE_VISUALIZATION
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#endif

namespace Penner {
namespace Feature {

// compute the barycentric coordiante of p on the line p0:p1
double compute_segment_coordinate(const Eigen::VectorXd& p0, const Eigen::VectorXd& p1, const Eigen::VectorXd& p)
{
    Eigen::VectorXd d0 = (p - p0);
    Eigen::VectorXd d1 = (p1 - p);
    return (d0.norm() / (d0.norm() + d1.norm()));
}

// compute the barycentric coordinates of a list of vertices on the edge between vi and vj
std::vector<double>
compute_edge_coordinates(const Eigen::MatrixXd& V, int vi, int vj, const std::vector<int>& v_points)
{
    // get endpoints
    Eigen::VectorXd pi = V.row(vi);
    Eigen::VectorXd pj = V.row(vj);

    // comptue coordinates for edge
    int num_points = v_points.size();
    std::vector<double> edge_coords(num_points, -1.);
    for (int k = 0; k < num_points; ++k) {
        int vk = v_points[k];
        Eigen::VectorXd pk = V.row(vk);
        edge_coords[k] = compute_segment_coordinate(pi, pj, pk);
    }

    return edge_coords;
}

// compute uniform barycentric coordinates for a given number of points
std::vector<double> compute_uniform_coordinates(int num_points)
{
    std::vector<double> edge_coords(num_points, -1.);
    for (int k = 0; k < num_points; ++k) {
        edge_coords[k] = (k + 1.) / (num_points + 1.);
    }

    return edge_coords;
}

// get the face and local corner index of a corner opposite an edge
std::pair<int, int> get_face_corner(
    const std::vector<std::array<int, 3>> F,
    const Eigen::SparseMatrix<int>& vv2f,
    int vi,
    int vj)
{
    int f = vv2f.coeff(vi, vj) - 1;
    if ((F[f][1] == vi) && (F[f][2] == vj)) return std::make_pair(f, 0);
    if ((F[f][2] == vi) && (F[f][1] == vj)) return std::make_pair(f, 0);
    if ((F[f][0] == vi) && (F[f][2] == vj)) return std::make_pair(f, 1);
    if ((F[f][2] == vi) && (F[f][0] == vj)) return std::make_pair(f, 1);
    return std::make_pair(f, 2);
}

// get the vertex index opposite an edge
int get_face_corner_index(
    const std::vector<std::array<int, 3>> F,
    const Eigen::SparseMatrix<int>& vv2f,
    int vi,
    int vj)
{
    auto [f, i] = get_face_corner(F, vv2f, vi, vj);
    return F[f][i];
}

// check if all vertices are distinct
bool is_valid_face(const std::array<int, 3> f)
{
    if (f[0] == f[1]) return false;
    if (f[1] == f[2]) return false;
    if (f[2] == f[0]) return false;
    return true;
}

// check if all faces are valid (distinct vertices)
bool is_valid_face_list(const std::vector<std::array<int, 3>> F)
{
    for (const auto& f : F) {
        if (!is_valid_face(f)) return false;
    }

    return true;
}

// generate vertex map after refinement
// WARNING: assumes added vertices have index at end
Eigen::VectorXi generate_refined_vertex_map(const Eigen::VectorXi& V_map, int num_refined_vertices)
{
    Eigen::VectorXi Vr_map(num_refined_vertices);
    int num_cut_vertices = V_map.size();
    int refined_vertex_count = V_map.maxCoeff() + 1;
    for (int vi = 0; vi < num_refined_vertices; ++vi) {
        if (vi < num_cut_vertices) {
            Vr_map[vi] = V_map[vi];
        } else {
            Vr_map[vi] = refined_vertex_count;
            ++refined_vertex_count;
        }
    }

    return Vr_map;
}

// Method class to splice together an overlay on a cut mesh.
class EdgeSplicer
{
public:
    /**
     * @brief Construct a new Edge Splicer object with a refined mesh with overlay data and gluing.
     * 
    * @param V: refined cut mesh vertices
    * @param F: refined cut mesh faces
    * @param uv: refined cut mesh uv vertices
    * @param F_uv: refined cut mesh uv faces
    * @param Fn_to_F: map from refined faces to original faces
    * @param endpoints: map from refined vertices to original edge endpoints
    * @param V_map: gluing map from original cut mesh vertices to closed mesh vertices
     */
    EdgeSplicer(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const Eigen::MatrixXd& uv,
        const Eigen::MatrixXi& F_uv,
        const std::vector<int>& Fn_to_F,
        const std::vector<std::pair<int, int>>& endpoints,
        const Eigen::VectorXi& V_map);

    /**
     * @brief Set the feature corners.
     * 
     * @param is_feature: mask of refined mesh corners opposite features
     */
    void set_feature_corners(const Eigen::MatrixXi& is_feature);

    /**
     * @brief Main method to splice edges once initialized.
     * 
     */
    void splice_edges();

    /**
     * @brief Get overlay mesh data for the stitched mesh.
     * 
     * @return stitched cut mesh vertices
     * @return stitched cut mesh faces
     * @return stitched cut mesh uv vertices
     * @return stitched cut mesh uv faces
     * @return map from stitched jfaces to original faces
     * @return map from stitched vertices to original edge endpoints
     */
    std::tuple<
        Eigen::MatrixXd,
        Eigen::MatrixXi,
        Eigen::MatrixXd,
        Eigen::MatrixXi,
        std::vector<int>,
        std::vector<std::pair<int, int>>>
    get_overlay_mesh() const;

    /**
     * @brief Get the stitched feature corners.
     * 
     * @return mask of stitched mesh corners opposite features
     */
    Eigen::MatrixXi get_feature_corners() const;

    // Use uniform barycentric coordinates if true
    bool use_uniform_bc;

private:
    // refined mesh data
    Eigen::MatrixXd m_V;
    Eigen::MatrixXd m_uv;
    std::vector<std::array<int, 3>> refined_F, refined_Fuv;
    std::vector<std::array<double, 2>> refined_uv;
    std::vector<int> m_Fn_to_F;
    std::vector<std::pair<int, int>> m_endpoints;

    // overlay mesh construction
    Eigen::MatrixXi F_orig;
    Eigen::MatrixXi F_uv_orig;
    std::vector<std::array<std::vector<int>, 3>> corner_v_points;
    std::vector<std::array<std::vector<int>, 3>> corner_uv_points;
    std::vector<std::array<std::vector<double>, 3>> corner_edge_coords;

    // feature data
    std::vector<std::array<int, 3>> is_feature_corner;

    // glued mesh halfedge
    std::vector<int> opposite;
    std::vector<std::pair<int, int>> he_to_corner;

    // vertex maps
    Eigen::SparseMatrix<int> vv2f;
    Eigen::VectorXi refined_V_map;

    // class to iterate over an edge while stitching
    class EdgeIterator
    {
    public:
        // build the iterator for the given halfedge
        EdgeIterator(EdgeSplicer& parent, int halfedge_index)
            : m_parent(parent)
            , hij(halfedge_index)
            , hji(parent.opposite[hij])
            , fijk(parent.he_to_corner[hij].first)
            , fjil(parent.he_to_corner[hji].first)
            , k(parent.he_to_corner[hij].second)
            , l(parent.he_to_corner[hji].second)
            , hij_coords(parent.corner_edge_coords[fijk][k])
            , hji_coords(parent.corner_edge_coords[fjil][l])
        {
            // initialize edge iterator indices
            hij_iter = 0;
            hij_end = hij_coords.size();
            int num_hji_splits = hji_coords.size();
            hji_iter = num_hji_splits - 1;
            hji_end = -1;

            // get vertices of halfedge face
            v_start_hij = parent.F_orig(fijk, (k + 1) % 3);
            v_end_hij = parent.F_orig(fijk, (k + 2) % 3);
            v_start_hji = parent.F_orig(fjil, (l + 2) % 3);
            v_end_hji = parent.F_orig(fjil, (l + 1) % 3);

            // get uv vertices of halfedge face
            uv_start_hij = parent.F_uv_orig(fijk, (k + 1) % 3);
            uv_end_hij = parent.F_uv_orig(fijk, (k + 2) % 3);
            uv_start_hji = parent.F_uv_orig(fjil, (l + 2) % 3);
            uv_end_hji = parent.F_uv_orig(fjil, (l + 1) % 3);

            // log face corner information
            SPDLOG_TRACE("refining hij = ({}, {})", v_start_hij, v_end_hij);
            SPDLOG_TRACE("midpoints are {}", formatted_vector(corner_v_points[fijk][k]));
            SPDLOG_TRACE("refining hji = ({}, {})", v_start_hji, v_end_hji);
            SPDLOG_TRACE("midpoints are {}", formatted_vector(corner_v_points[fjil][l]));
        }

        // determine if at end of segment
        bool is_segment_end() const { return (hij_iter == hij_end); }

        // determine if at end of opposite segment
        bool is_opposite_segment_end() const { return (hji_iter == hji_end); }

        // determine if at end of both segments
        bool is_end() const { return (is_segment_end() && is_opposite_segment_end()); }

        // iterate (unless at end)
        void iterate_segment()
        {
            // do nothing if at end
            if (is_segment_end()) return;

            // iterate segment, tracking base explicitly
            ++hij_iter;
        }

        // iterate (unless at end)
        void iterate_opposite_segment()
        {
            // do nothing if at end
            if (is_opposite_segment_end()) return;

            // iterate segment, tracking base explicitly
            --hji_iter;
        }

        // getters
        int get_start_vertex() const { return v_start_hij; }
        int get_end_vertex() const { return v_end_hij; }
        int get_opposite_start_vertex() const { return v_start_hji; }
        int get_opposite_end_vertex() const { return v_end_hji; }
        int get_start_uv_vertex() const { return uv_start_hij; }
        int get_end_uv_vertex() const { return uv_end_hij; }
        int get_opposite_start_uv_vertex() const { return uv_start_hji; }
        int get_opposite_end_uv_vertex() const { return uv_end_hji; }

        // get refined vertex at tip of current segment, or original vertex if at end
        int get_segment_tip() const
        {
            return (hij_iter != hij_end) ? m_parent.corner_v_points[fijk][k][hij_iter] : v_end_hij;
        }

        // get refined vertex at tip of current opposite segment, or original vertex if at end
        int get_opposite_segment_tip() const
        {
            return (hji_iter != hji_end) ? m_parent.corner_v_points[fjil][l][hji_iter] : v_end_hji;
        }

        // get refined uv vertex at tip of current segment, or original vertex if at end
        int get_segment_uv_tip() const
        {
            return (hij_iter != hij_end) ? m_parent.corner_uv_points[fijk][k][hij_iter]
                                         : uv_end_hij;
        }

        // get refined uv vertex at tip of current opposite segment, or original vertex if at end
        int get_opposite_segment_uv_tip() const
        {
            return (hji_iter != hji_end) ? m_parent.corner_uv_points[fjil][l][hji_iter]
                                         : uv_end_hji;
        }

        // get barycentric coordinate for end of current segment, or 1 if at end
        double get_segment_coord() const
        {
            return (hij_iter != hij_end) ? hij_coords[hij_iter] : 1.;
        }

        // get barycentric coordinate for end of current opposite segment, or 1 if at end
        double get_opposite_segment_coord() const
        {
            return (hji_iter != hji_end) ? (1. - hji_coords[hji_iter]) : 1.;
        }

        // determine if segment tips are the same (should only happen if at endof segment or extreme numerics)
        bool are_segment_tips_equal() const
        {
            return (get_segment_tip() == get_opposite_segment_tip());
        }

        // determine if segment tips are within some epsilon threshold but not exactly equal
        bool are_segment_tips_approx_equal() const
        {
            // check if exactly equal
            if (are_segment_tips_equal()) return false;
            if ((is_opposite_segment_end()) && (is_segment_end())) return false;

            // check if at end of segment and opposite segment is near 1
            if (is_segment_end()) {
                double t_hji = 1. - hji_coords[hji_iter];
                return (float_equal(t_hji, 1., 1e-8));
            }

            // check if at end of opposite segment and segment is near 1
            if (is_opposite_segment_end()) {
                double t_hij = hij_coords[hij_iter];
                return (float_equal(t_hij, 1., 1e-8));
            }

            // check if barycentric coordinates of vertices are close
            double t_hij = hij_coords[hij_iter];
            double t_hji = 1. - hji_coords[hji_iter];
            if (float_equal(t_hij, t_hji, 1e-8)) {
                return true;
            }

            return false;
        }

        // check if the segment is currently ahead
        bool is_segment_tip_ahead() const
        {
            // check if iterators are at endpoints
            if (are_segment_tips_equal()) return false;
            if (is_segment_end()) return true;
            if (is_opposite_segment_end()) return false;

            // compare edge coordinates, reversing orientation for opposite segment
            // break ties in segments favor
            double t_hij = hij_coords[hij_iter];
            double t_hji = 1. - hji_coords[hji_iter];
            return (t_hij >= t_hji);
        }

        // check if the opposite segment is currently ahead
        bool is_opposite_segment_tip_ahead() const
        {
            // check if iterators are at endpoints
            if (are_segment_tips_equal()) return false;
            if (is_opposite_segment_end()) return true;
            if (is_segment_end()) return false;

            // compare edge coordinates, reversing orientation for opposite segment
            double t_hij = hij_coords[hij_iter];
            double t_hji = 1. - hji_coords[hji_iter];
            return (t_hij < t_hji);
        }

    private:
        EdgeSplicer& m_parent;
        int hij, hji, fijk, fjil, k, l;
        const std::vector<double>&hij_coords, hji_coords;


        int hij_iter, hij_end, hji_iter, hji_end;
        int v_start_hij, v_end_hij, v_start_hji, v_end_hji;
        int uv_start_hij, uv_end_hij, uv_start_hji, uv_end_hji;
    };

    // get vertex positions
    const Eigen::MatrixXd& get_vertices() const { return m_V; }

    // identify the tips of the current segments
    void merge_segment(EdgeIterator& edge_iterator)
    {
        int v_tip_hij = edge_iterator.get_segment_tip();
        int v_tip_hji = edge_iterator.get_opposite_segment_tip();
        refined_V_map[v_tip_hij] = v_tip_hji;
    }

    // compute the uv area of a face
    Scalar compute_area(int f)
    {
        int vi = refined_Fuv[f][0];
        int vj = refined_Fuv[f][1];
        int vk = refined_Fuv[f][2];
        Eigen::Vector2d pi(refined_uv[vi][0], refined_uv[vi][1]);
        Eigen::Vector2d pj(refined_uv[vj][0], refined_uv[vj][1]);
        Eigen::Vector2d pk(refined_uv[vk][0], refined_uv[vk][1]);

        return Optimization::signed_area(pi, pj, pk);
    }

    // split the current segment, returning the current stitched vertex and corresponding uv points
    std::tuple<int, int, int> split_segment(EdgeIterator& edge_iterator, int segment_base_vertex)
    {
        // get vertices of the split triangles adjacent to the halfedge segment
        int v_base_hij = segment_base_vertex;
        int v_tip_hij = edge_iterator.get_segment_tip();
        int v_tip_hji = edge_iterator.get_opposite_segment_tip();
        int v_edge = v_tip_hji;
        int v_corner_hij = get_face_corner_index(refined_F, vv2f, v_base_hij, v_tip_hij);

        // get adjacent face of the segment to split
        auto [f_old, f_old_corner] = get_face_corner(refined_F, vv2f, v_base_hij, v_tip_hij);
        int f_new = refined_F.size();

        // refine current face
        refined_F[f_old] = {v_corner_hij, v_edge, v_tip_hij};
        refined_F.push_back({v_corner_hij, v_base_hij, v_edge});
        m_Fn_to_F.push_back(m_Fn_to_F[f_old]);
        std::array<int, 3> old_is_feature = is_feature_corner[f_old];
        is_feature_corner[f_old] = {
            old_is_feature[f_old_corner],
            old_is_feature[(f_old_corner + 1) % 3],
            false};
        is_feature_corner.push_back(
            {old_is_feature[f_old_corner], false, old_is_feature[(f_old_corner + 2) % 3]});

        // update edge to face map
        set_VV_edge_faces(v_corner_hij, v_edge, v_tip_hij, f_old);
        set_VV_edge_faces(v_corner_hij, v_base_hij, v_edge, f_new);
        vv2f.coeffRef(v_base_hij, v_tip_hij) = 0;

        // add new midpoint for the uv domain face
        double t = edge_iterator.get_opposite_segment_coord();
        Eigen::Vector2d p0 = m_uv.row(edge_iterator.get_start_uv_vertex());
        Eigen::Vector2d p1 = m_uv.row(edge_iterator.get_end_uv_vertex());
        Eigen::Vector2d p = (1. - t) * p0 + t * p1;
        int uvm = add_uv_edge_point(f_old, f_old_corner, p);
        int uvm_opp = edge_iterator.get_opposite_segment_uv_tip();

#ifdef CHECK_VALIDITY
        // check the areas of the refined faces
        Scalar area;
        area = compute_area(f_old);
        if (area < 1e-10)
        {
            spdlog::warn("refined face of area {}", area);
        }
        area = compute_area(f_new);
        if (area < 1e-10)
        {
            spdlog::warn("refined face of area {}", area);
        }
#endif

        return {v_edge, uvm, uvm_opp};
    }

    // split the current opposite segment, returning the current stitched vertex and corresponding uv points
    std::tuple<int, int, int> split_opposite_segment(
        EdgeIterator& edge_iterator,
        int opposite_segment_base_vertex)
    {
        // get vertices of the split triangles adjacent to the opposite halfedge segment
        int v_base_hji = opposite_segment_base_vertex;
        int v_tip_hij = edge_iterator.get_segment_tip();
        int v_tip_hji = edge_iterator.get_opposite_segment_tip();
        int v_edge = v_tip_hij;
        int v_corner_hji = get_face_corner_index(refined_F, vv2f, v_tip_hji, v_base_hji);

        // get adjacent face of the segment to split
        auto [f_old, f_old_corner] = get_face_corner(refined_F, vv2f, v_tip_hji, v_base_hji);
        int f_new = refined_F.size();

        // refine current face
        refined_F[f_old] = {v_corner_hji, v_edge, v_base_hji};
        refined_F.push_back({v_corner_hji, v_tip_hji, v_edge});
        m_Fn_to_F.push_back(m_Fn_to_F[f_old]);
        std::array<int, 3> old_is_feature = is_feature_corner[f_old];
        is_feature_corner[f_old] = {
            old_is_feature[f_old_corner],
            old_is_feature[(f_old_corner + 1) % 3],
            false};
        is_feature_corner.push_back(
            {old_is_feature[f_old_corner], false, old_is_feature[(f_old_corner + 2) % 3]});

        // update edge to face map
        set_VV_edge_faces(v_corner_hji, v_edge, v_base_hji, f_old);
        set_VV_edge_faces(v_corner_hji, v_tip_hji, v_edge, f_new);
        vv2f.coeffRef(v_tip_hji, v_base_hji) = 0;

        // add new midpoint for the uv domain face
        double t = edge_iterator.get_segment_coord();
        Eigen::Vector2d p0 = m_uv.row(edge_iterator.get_opposite_start_uv_vertex());
        Eigen::Vector2d p1 = m_uv.row(edge_iterator.get_opposite_end_uv_vertex());
        Eigen::Vector2d p = (1. - t) * p0 + t * p1;
        int uvm_opp = add_uv_edge_point(f_old, f_old_corner, p);
        int uvm = edge_iterator.get_segment_uv_tip();

#ifdef CHECK_VALIDITY
        Scalar area;
        area = compute_area(f_old);
        if (area < 1e-10)
        {
            spdlog::info("refined face of area {}", area);
        }
        area = compute_area(f_new);
        if (area < 1e-10)
        {
            spdlog::info("refined face of area {}", area);
        }
#endif

        return {v_edge, uvm, uvm_opp};
    }

    // splice a given edge containing the halfedge
    bool splice_edge(int halfedge_index)
    {
        // initialize an iterator for the edge
        int hij = halfedge_index;
        EdgeIterator edge_iterator(*this, hij);

        // track base vertex of the active segments on the halfedges where the edge has already been spliced
        // NOTE: this is not tracked with the iterator as it is not invariant under splicing
        // the tip of the active segment, in contrast, is invariant and not invalidated by splicing
        int v_base_hij = edge_iterator.get_start_vertex();
        int v_base_hji = edge_iterator.get_opposite_start_vertex();
        int v_base = edge_iterator.get_start_vertex();
        int v_tip = edge_iterator.get_end_vertex();
        std::vector<int> edge_vertices = {};
        std::vector<int> hij_uv_vertices = {};
        std::vector<int> hji_uv_vertices = {};
        int uvm, uvm_opp;

        // loop until both queues processed
        double t = 0.;
        bool allow_collapse = false; // TODO: make optional
        while (!edge_iterator.is_end()) {
            // check if near degenerate split occuring
            if (edge_iterator.are_segment_tips_approx_equal()) {
                spdlog::warn("Near degenerate split");
            }

            // no refinement needed if active segments have same tip
            if (edge_iterator.are_segment_tips_equal()) {
                // get geometry for current tip (current base of the zipper)
                t = edge_iterator.get_segment_coord();
                v_base_hij = edge_iterator.get_segment_tip();
                v_base_hji = edge_iterator.get_opposite_segment_tip();
                uvm = edge_iterator.get_segment_uv_tip();
                uvm_opp = edge_iterator.get_opposite_segment_uv_tip();

                // iterate segments
                edge_iterator.iterate_segment();
                edge_iterator.iterate_opposite_segment();
            }
            // optionally allow a collapse of approximately equal segment tips
            else if ((allow_collapse) && (edge_iterator.are_segment_tips_approx_equal())) {
                spdlog::warn("Merging segments");
                // get uv geometry for tip (don't add new uv points)
                t = edge_iterator.get_segment_coord();
                uvm = edge_iterator.get_segment_uv_tip();
                uvm_opp = edge_iterator.get_opposite_segment_uv_tip();

                // identify the tips
                merge_segment(edge_iterator);

                // get the now joined tip
                v_base_hij = edge_iterator.get_segment_tip();
                v_base_hji = edge_iterator.get_opposite_segment_tip();

                // iterate both segments
                edge_iterator.iterate_segment();
                edge_iterator.iterate_opposite_segment();

                // ensure the new vertex position is consistent with the uv position
                m_V.row(v_base_hij) = ((1 - t) * m_V.row(v_base)) + (t * m_V.row(v_tip));
            }
            // refine current active segment if the tip is farther along the edge than the opposite
            else if (edge_iterator.is_segment_tip_ahead()) {
                t = edge_iterator.get_opposite_segment_coord();

                // split the current segment and increment the base of splicing to the new vertex
                std::tie(v_base_hij, uvm, uvm_opp) = split_segment(edge_iterator, v_base_hij);
                v_base_hji = v_base_hij;
                edge_iterator.iterate_opposite_segment();

                // ensure the new vertex position is consistent with the uv position
                m_V.row(v_base_hij) = ((1 - t) * m_V.row(v_base)) + (t * m_V.row(v_tip));
            }
            // refine opposite active segment if the tip is farther along the edge
            else if (edge_iterator.is_opposite_segment_tip_ahead()) {
                t = edge_iterator.get_segment_coord();

                // split the current opposite segment and increment the base of splicing to the new vertex
                std::tie(v_base_hji, uvm, uvm_opp) =
                    split_opposite_segment(edge_iterator, v_base_hji);
                v_base_hij = v_base_hji;
                edge_iterator.iterate_segment();

                // ensure the new vertex position is consistent with the uv position
                m_V.row(v_base_hij) = ((1 - t) * m_V.row(v_base)) + (t * m_V.row(v_tip));
            }
            // error case: should not occur
            else
            {
                spdlog::info("Invalid edge midpoint");
                return false;
            }

            // store indices of the current splicing base 
            edge_vertices.push_back(v_base_hij);
            hij_uv_vertices.push_back(uvm);
            hji_uv_vertices.push_back(uvm_opp);
        }

        // optionally use uniform barycentric coordinates to stabilize later optimzations
        if (use_uniform_bc) {
            int num_midpoints = edge_vertices.size();
            std::vector<double> uniform_coords = compute_uniform_coordinates(num_midpoints);
            for (int m = 0; m < num_midpoints; ++m) {
                double t = uniform_coords[m];
                m_V.row(edge_vertices[m]) = ((1 - t) * m_V.row(v_base)) + (t * m_V.row(v_tip));
            }
        }

        return true;
    }

    // update the map from edges to faces for a given triangle
    void set_VV_edge_faces(int vi, int vj, int vk, int fijk)
    {
        vv2f.coeffRef(vi, vj) = fijk + 1;
        vv2f.coeffRef(vj, vk) = fijk + 1;
        vv2f.coeffRef(vk, vi) = fijk + 1;
    }

    // add a uv edge point opposite a corner with given barycentric coordiante
    int add_uv_edge_point(int fijk, int k, double t)
    {
        // get local connectivity
        int uvi = refined_Fuv[fijk][(k + 1) % 3];
        int uvj = refined_Fuv[fijk][(k + 2) % 3];
        int uvk = refined_Fuv[fijk][(k + 0) % 3];
        int uvm = refined_uv.size();

        // build new connectivity
        refined_Fuv[fijk] = {uvk, uvm, uvj};
        refined_Fuv.push_back({uvk, uvi, uvm});

        // build uv point
        Eigen::Vector2d UVi = {refined_uv[uvi][0], refined_uv[uvi][1]};
        Eigen::Vector2d UVj = {refined_uv[uvj][0], refined_uv[uvj][1]};
        Eigen::Vector2d inserted_point = (1 - t) * UVi + t * UVj;
        refined_uv.push_back({inserted_point[0], inserted_point[1]});

        // return new point index
        return uvm;
    }

    // add a uv edge point opposite a corner with given position
    int add_uv_edge_point(int fijk, int k, const Eigen::Vector2d& inserted_point)
    {
        // get local connectivity
        int uvi = refined_Fuv[fijk][(k + 1) % 3];
        int uvj = refined_Fuv[fijk][(k + 2) % 3];
        int uvk = refined_Fuv[fijk][(k + 0) % 3];
        int uvm = refined_uv.size();

        // build new connectivity
        refined_Fuv[fijk] = {uvk, uvm, uvj};
        refined_Fuv.push_back({uvk, uvi, uvm});

        // build uv point
        refined_uv.push_back({inserted_point[0], inserted_point[1]});

        // return new point index
        return uvm;
    }

    // build overlay endpoints for the stitched mesh
    std::vector<std::pair<int, int>> get_endpoints() const
    {
        // apply refined gluing map to endpoints, checking and preserving -1 indices
        int num_refined_vertices = refined_V_map.size();
        int num_stitched_vertices = refined_V_map.maxCoeff() + 1;
        std::vector<std::pair<int, int>> endpoints(num_stitched_vertices); 
        for (int vi = 0; vi < num_refined_vertices; ++vi) {
            auto [vj, vk] = m_endpoints[vi];
            endpoints[refined_V_map[vi]].first = (vj >= 0) ? refined_V_map[vj] : -1;
            endpoints[refined_V_map[vi]].second = (vk >= 0) ? refined_V_map[vk] : -1;
        }

        return endpoints;
    }
};

EdgeSplicer::EdgeSplicer(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<int>& Fn_to_F,
    const std::vector<std::pair<int, int>>& endpoints,
    const Eigen::VectorXi& V_map)
    : use_uniform_bc(false)
    , m_V(V)
    , m_uv(uv)
    , m_Fn_to_F(Fn_to_F)
    , m_endpoints(endpoints)
{
    // build list of split faces per original face
    std::vector<std::vector<int>> F_to_Fn = Optimization::build_F_to_Fn(Fn_to_F);

    // For each original face, get the overlay vertices corresponding to the face
    int num_faces = F_to_Fn.size();
    Eigen::MatrixXi halfedge_map;
    Optimization::build_faces(
        F,
        F_uv,
        F_to_Fn,
        endpoints,
        F_orig,
        corner_v_points,
        F_uv_orig,
        corner_uv_points,
        halfedge_map);

    // reglue cut edges of the mesh using the vertex identification
    Eigen::MatrixXi F_glued(num_faces, 3);
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        for (int i = 0; i < 3; ++i) {
            F_glued(fijk, i) = V_map[F_orig(fijk, i)];
        }
    }

    // build halfedge pairings for the glued mesh
    std::vector<int> next_he;
    std::vector<int> vtx_reindex;
    std::vector<int> bnd_loops;
    std::vector<std::vector<int>> corner_to_he;
    FV_to_NOB(F_glued, next_he, opposite, bnd_loops, vtx_reindex, corner_to_he, he_to_corner);

    // get barycentric coordinates on each edge
    corner_edge_coords.resize(num_faces);
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;
            int uvj = F_uv_orig(fijk, j);
            int uvk = F_uv_orig(fijk, k);
            corner_edge_coords[fijk][i] =
                compute_edge_coordinates(uv, uvj, uvk, corner_uv_points[fijk][i]);
        }
    }

    // initialize map from vertex-vertex edge pairs to faces
    vv2f = generate_VV_to_face_map(F);

    // initialize extendable refined mesh faces as list of lists
    int num_refined_faces = F.rows();
    refined_F.reserve(2 * num_refined_faces);
    refined_Fuv.reserve(2 * num_refined_faces);
    is_feature_corner.reserve(2 * num_refined_faces);
    for (int fijk = 0; fijk < num_refined_faces; ++fijk) {
        refined_F.push_back({F(fijk, 0), F(fijk, 1), F(fijk, 2)});
        refined_Fuv.push_back({F_uv(fijk, 0), F_uv(fijk, 1), F_uv(fijk, 2)});
        is_feature_corner.push_back({false, false, false});
    }

    // initialize extendable refined mesh uv vertices
    // (3D vertex list needs to be reduced, not extended)
    refined_uv.reserve(2 * m_uv.rows());
    int num_refined_uv = m_uv.rows();
    for (int uvi = 0; uvi < num_refined_uv; ++uvi) {
        refined_uv.push_back({m_uv(uvi, 0), m_uv(uvi, 1)});
    }

    // generate initial refined vertex map
    refined_V_map = generate_refined_vertex_map(V_map, V.rows());
}

void EdgeSplicer::set_feature_corners(const Eigen::MatrixXi& is_feature)
{
    int num_faces = is_feature.rows();
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        for (int i = 0; i < 3; ++i) {
            is_feature_corner[fijk][i] = is_feature(fijk, i);
        }
    }
}

void EdgeSplicer::splice_edges()
{
    int num_halfedges = opposite.size();
    for (int hij = 0; hij < num_halfedges; ++hij) {
        if (hij < opposite[hij]) continue;
        bool success = splice_edge(hij);
        if (!success) return;
    }
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    std::vector<int>,
    std::vector<std::pair<int, int>>>
EdgeSplicer::get_overlay_mesh() const
{
    // build mesh with removed duplicates
    Eigen::MatrixXd V_r = matrix_inverse_reindex_domain(m_V, refined_V_map);
    Eigen::MatrixXi F_r =
        matrix_reindex_range(convert_std_to_eigen_matrix(refined_F), refined_V_map);

    // convert refined face matrices
    Eigen::MatrixXi Fuv_r = convert_std_to_eigen_matrix(refined_Fuv);
    Eigen::MatrixXd uv_r;
    convert_std_to_eigen_matrix(refined_uv, uv_r);

    // get refinement data
    std::vector<int> Fn_to_F_r = m_Fn_to_F;
    std::vector<std::pair<int, int>> endpoints_r = get_endpoints();

#ifdef CHECK_VALIDITY
    // check validity
	std::vector<std::vector<int>> loops;
	igl::boundary_loop(F_r, loops);
    if (loops.size() > 0) spdlog::warn("{} loops in stitched mesh", loops.size());
#endif

    return std::make_tuple(V_r, F_r, uv_r, Fuv_r, Fn_to_F_r, endpoints_r);
}

Eigen::MatrixXi EdgeSplicer::get_feature_corners() const
{
    return convert_std_to_eigen_matrix(is_feature_corner);
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    Eigen::MatrixXd,
    Eigen::MatrixXi,
    std::vector<int>,
    std::vector<std::pair<int, int>>,
    Eigen::MatrixXi>
stitch_cut_overlay(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv,
    const std::vector<int>& Fn_to_F,
    const std::vector<std::pair<int, int>>& endpoints,
    const Eigen::MatrixXi& is_feature_corner,
    const Eigen::VectorXi& V_map,
    bool use_uniform_bc)
{
    // splice overlay edges together
    EdgeSplicer edge_splicer(V, F, uv, F_uv, Fn_to_F, endpoints, V_map);
    edge_splicer.use_uniform_bc=use_uniform_bc;

    // set the feature corners
    spdlog::debug("Setting feature corners");
    edge_splicer.set_feature_corners(is_feature_corner);

    // do the splicing
    spdlog::debug("Splicing edges");
    edge_splicer.splice_edges();

    // get the stiched mesh and feature corners
    auto [V_r, F_r, uv_r, Fuv_r, Fn_to_F_r, endpoints_r] = edge_splicer.get_overlay_mesh();
    Eigen::MatrixXi is_feature_corner_r = edge_splicer.get_feature_corners();

    return std::make_tuple(V_r, F_r, uv_r, Fuv_r, Fn_to_F_r, endpoints_r, is_feature_corner_r);
}


} // namespace Feature
} // namespace Penner
