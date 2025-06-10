#include "holonomy/field/facet_field.h"
#include <iostream>
#include <fstream>

namespace Penner {
namespace Holonomy {

const char WHITESPACE[] = " \t\r\n";

std::string& rtrim(std::string& s)
{
    size_t pos = s.find_last_not_of(WHITESPACE);
    s.erase(pos == std::string::npos ? 0 : pos + 1);
    return s;
}

////////////////////////////////////////////////////////////

std::istream& readline(std::istream& is, std::string& line)
{
    if (getline(is, line)) {
        rtrim(line);
    }
    return is;
}

////////////////////////////////////////////////////////////

std::istream&
readline_nocomments(std::istream& is, std::string& line, const char* comment_prefix = "#")
{
    const size_t comment_len = strlen(comment_prefix);
    while (getline(is, line)) {
        rtrim(line);
        if (line.size() != 0 && (line.substr(0, comment_len) != comment_prefix)) break;
    }
    return is;
}


// TODO: Matching validity
// if (MIM::matching_initialized(hc->opposite()) &&
//     fmod(MIM::fmatching(hc) + MIM::fmatching(hc->opposite()), 4.0) != 0)
//     Log.warn(
//         "'%s' facet %d: Inconsistent matchings %g and %g across edge.",
//         fname,
//         idx,
//         (double)MIM::fmatching(hc),
//         (double)MIM::fmatching(hc->opposite()));


double cos_angle_rescaled_tri(const Mesh<Scalar>& m, int h)
{
    if (m.opp[h] < 0) return 1.0; // cos(0)

    //
    //     /\.
    //  l0/  \l2
    //   /____\.
    // 01  l1
    // Law of cosines:
    // l2^2 = l0^2 + l1^2 - 2*l0*l1*cos(theta01)
    // => cos(theta01) = (l0^2 + l1^2 - l2^2) / (2*l0*l1);
    double l0 = (double)(m.l[h]);
    double l1 = (double)(m.l[m.n[h]]);
    double l2 = (double)(m.l[m.n[m.n[h]]]);
    double cos01;
    if (l0 == 0 && l1 == 0 && l2 == 0) {
        cos01 = 0.5;
    } else {
        double denom01 = 2 * l0 * l1;
        cos01 = denom01 == 0 ? 1.0 : (l0 * l0 + l1 * l1 - l2 * l2) / denom01;
    }

    // Will (may?) get triangle inequality violations if the following
    // lines are uncommented.
    const double EPS = 1e-10; // FLT_EPSILON;
    if (1.0 <= cos01 && cos01 < 1 + EPS) {
        cos01 = 1.0;
    } else if (-1.0 >= cos01 && cos01 > -1 - EPS) {
        cos01 = -1.0;
    }

    if (-1 > cos01 || cos01 > 1) {
        spdlog::error("Cosine {} is outside [-1, 1].\n", cos01);
    }
    //    assert(-1 <= cos01 && cos01 <= 1);
    return cos01;
}

// Returns a representation of the scaled triangle on the xy plane, with the
// provided halfedge being along the x-axis, starting at the origin.
//                 * p2 = l2(cos(t),sin(t))
//                / \.
//           l2  /   \  l1
//              /t  he\.
//  p0 = (0,0) *-------* p1 = (l0,0)
//                l0
void scaled_triangle(
    const Mesh<Scalar>& m,
    int h,
    Eigen::Vector3d& p0,
    Eigen::Vector3d& p1,
    Eigen::Vector3d& p2)
{
    p0 = Eigen::Vector3d(0, 0, 0);
    p1 = Eigen::Vector3d((double)(m.l[h]), 0, 0);
    int h_prev = m.n[m.n[h]];
    double cos_t = cos_angle_rescaled_tri(m, h_prev);
    double sin_t = sqrt(1.0 - cos_t * cos_t);
    p2 = (double)(m.l[h_prev]) * Eigen::Vector3d(cos_t, sin_t, 0);
}

// reference angles when computed on the scaled triangle (that actually
// lives in R^6)
void reference_angles_rescaled_tri(
    const Mesh<Scalar>& m,
    const Eigen::VectorXi& reference_halfedges,
    int he,
    double& d1,
    double& d2)
{
    // assume two incident faces
    int f1 = m.f[he];
    int f2 = m.f[m.opp[he]];

    // compute a representation of each scaled triangle on the plane
    Eigen::Vector3d p0, p1, p2, q0, q1, q2;
    scaled_triangle(m, he, p0, p1, p2);
    scaled_triangle(m, m.opp[he], q0, q1, q2);

    int ref1 = reference_halfedges[f1];
    int ref2 = reference_halfedges[f2];
    Eigen::Vector3d refv1 = (ref1 == he) ? (p1 - p0) : (ref1 == m.n[he]) ? (p2 - p1) : (p0 - p2);
    Eigen::Vector3d refv2 = (ref2 == m.opp[he])        ? (q1 - q0)
                            : (ref2 == m.n[m.opp[he]]) ? (q2 - q1)
                                                       : (q0 - q2);
    // normals
    const Eigen::Vector3d norm(0, 0, 1);
    // angles from the common edge to ref edges
    d1 = -signed_angle<Eigen::Vector3d>(refv1, p1 - p0, norm);
    d2 = -signed_angle<Eigen::Vector3d>(refv2, q1 - q0, norm);
}

// Assumes that vertex indices are unique
bool has_priority(const Mesh<Scalar>& m, const std::vector<int>& vtx_reindex, int h)
{
    assert(m.to[h] != m.to[m.opp[h]]);
    return (vtx_reindex[m.to[h]] < vtx_reindex[m.to[m.opp[h]]]);
}

// This function computes kij in
//   dtheta = theta_i - theta_j + kij + pij*pi/2
//
// This is the same as the angle difference between adjacent reference
// edges:
//   ref_ang(e) - ref_ang(e->opposite())
//
// *** NOTE ***
// This convention is CONSISTENT WITH the Mixed Integer convention!
// kij goes from facet i (self) to facet j (opposite) since
//   dtheta = 0
//   => theta_j = theta_i + kij + pij*pi/2
//
// Picking a priority edge to always compute against and picking the other
// to be the negative of the first guarantees that the constraint of kappa_ij = -kappa_ji
// won't be a problem. Otherwise pi, -pi would be different for the edge and its opposite.
double diff_reference_angles(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& reference_halfedges,
    const Eigen::VectorXd crossfield_angles,
    int h)
{
    if (!has_priority(m, vtx_reindex, h))
        return -diff_reference_angles(
            m,
            vtx_reindex,
            reference_halfedges,
            crossfield_angles,
            m.opp[h]);

    // angle between reference directions in faces i and j
    double di, dj;
    reference_angles_rescaled_tri(m, reference_halfedges, h, di, dj);
    double kij = di - dj + M_PI;
    // kij += crossfield_angles[m.f[h]] - crossfield_angles[m.f[m.opp[h]]];
    //  to be consistent with mixed integer, ensure that kij \in (-pi, pi]
    kij = pos_fmod(kij, 2 * M_PI);
    if (kij > M_PI) {
        kij -= 2 * M_PI;
    }
    if (!(-M_PI < kij && kij <= M_PI)) {
        spdlog::error("{} out of range (-pi, pi])", kij);
    }
    return kij;
}

// Computes the MI-matchings from the crossfield angles
// * target_field_curvature is from Geometry-Aware. It's 0 by default.
double MI_Matching_from_Crossfield_double(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& reference_halfedges,
    const Eigen::VectorXd& crossfield_angles,
    int h,
    double ang_h,
    double ang_hopp,
    double target_field_curvature = 0)
{
    if (m.opp[h] < 0) return 0;

    // for exact consistency in computation, ensure a strict order of halfedge parameters
    // (order priority set based on pointer address)
    if (!has_priority(m, vtx_reindex, h))
        return -MI_Matching_from_Crossfield_double(
            m,
            vtx_reindex,
            reference_halfedges,
            crossfield_angles,
            m.opp[h],
            ang_hopp,
            ang_h,
            target_field_curvature);

    double kij = diff_reference_angles(m, vtx_reindex, reference_halfedges, crossfield_angles, h);
    double theta_diff = ang_h + kij - ang_hopp;
    double pij_f = (2.0 / M_PI) * (target_field_curvature - theta_diff);
    return pij_f;
}

// Computes the MI-matchings from the crossfield angles
// * target_field_curvature is from Geometry-Aware. It's 0 by default.
double MI_Matching_from_Crossfield_double(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& reference_halfedges,
    const Eigen::VectorXd& crossfield_angles,
    int h,
    double target_field_curvature = 0)
{
    if (m.opp[h] < 0) return 0;

    return MI_Matching_from_Crossfield_double(
        m,
        vtx_reindex,
        reference_halfedges,
        crossfield_angles,
        h,
        crossfield_angles[m.f[h]],
        crossfield_angles[m.f[m.opp[h]]],
        target_field_curvature);
}

// Compute the matching for the whole edge (h and h->opposite()) to be
// consistent with the crossfield.
// This is useful for mapping a 0..3 matching to a -infty..infty matching
// that represents in the difference in the angle representation of the
// field.
//
// This is not well-defined if the matching is uninitialized.
//
// WHY THIS IS NEEDED:
//   Mixed Integer matchings are arbitrary integers indicating
//   (approximately) the difference between the angle representations of
//   adjacent crossfields, divided by pi/2. In particular, they are not
//   restricted to 0..3 and it is not always possible to modify all the
//   angles in a mesh so that the matchings will be 0..3 (this is due to the
//   fact that every triangle has a differently-oriented representative
//   halfedge). In fact, changing all the matchings to 0..3 and then
//   re-running MI crossfield smoothing with fixed matchings will often
//   result in a garbage field.
//
//   Therefore, this function is designed to convert matchings in 0..3 to
//   matchings that are the same mod 4, but which approximate the difference
//   between the angle representations of adjacent crossfield (divided by
//   pi/2).
int MatchingConsistentWithCrossfield(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& reference_halfedges,
    const Eigen::VectorXd& crossfield_angles,
    int h,
    int matching)
{
    const double BAD_MATCHING = 999999;
    if (abs(matching) == BAD_MATCHING) {
        spdlog::error("MatchingConsistentWithCrossfield(): Supplied uninitialized MATCHING! "
                      "Continuing anyway.\n");
    }

    // int match_curr = MIM::matching(h);
    //  Compute the actual matching from the crossfield
    int match_curr = matching;
    double match_actual_f = MI_Matching_from_Crossfield_double(
        m,
        vtx_reindex,
        reference_halfedges,
        crossfield_angles,
        h);
    int match_actual = (int)floor(match_actual_f + 0.5);
    // If the actual matching matches the current matching, we are done.
    if (match_curr == match_actual) return match_curr;

    // If the actual matching (match_actual) differs from the currently
    // assigned matching (match_curr), then find the closest integer to
    // match_actual that is equivalent to match_curr mod 4.
    //
    // This corresponds to adding an integer diff \in -2..2 to match_actual
    // so that (match_actual + diff) % 4 = match_curr %4.
    int diff = (match_curr - match_actual);
    diff = diff < 0 ? (-3 * diff) % 4 : diff % 4; // this makes diff in 0..3
    if (diff > 2) diff -= 4; // normalize diff to -1..2

    // choose between 2 and -2: if rounded down, choose -2
    if (diff == 2 && match_actual_f > match_actual) diff = -2;

    assert(-2 <= diff && diff <= 2);
    match_actual += diff;

    assert((match_actual - match_curr) % 4 == 0);

    return match_actual;
}

// Modifies matchings by multiples of 4 in order to minimize the
// connection (theta_i+k_ij+p_ij*pi/2 - theta_j) on a per-edge basis.
//
// Returns true iff any matchings are changed.
bool MakeMatchingsConsistentWithCrossfield(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXi& reference_halfedges,
    const Eigen::VectorXd& crossfield_angles,
    Eigen::VectorXi& matchings)

{
    bool changed = false;
    int num_halfedges = m.n_halfedges();
    for (int h = 0; h < num_halfedges; ++h) {
        int match_actual = MatchingConsistentWithCrossfield(
            m,
            vtx_reindex,
            reference_halfedges,
            crossfield_angles,
            h,
            matchings[h]);
        if (match_actual != matchings[h] || -match_actual != matchings[m.opp[h]]) {
            matchings[h] = match_actual;
            changed = true;
        }
    }
    return changed;
}

// Loads frame field data from a .ffield file. The file is
// similar to an rscv file except that it stores factorized tensors per
// facet instead of per vertex.
//
std::tuple<
    int, // version
    Eigen::VectorXd, // crossfield_angles
    Eigen::MatrixXi, // F_id
    Eigen::MatrixXi, // F_matching
    Eigen::MatrixXi> // F_sharp
load_facet_field(const std::string& fname)
{
    std::tuple<int, Eigen::VectorXd, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi>
        empty_output;
    std::ifstream fin(fname);
    if (!fin) {
        spdlog::error("{}: Unable to open .ffield file", fname);
        return empty_output;
    }

    int version = 0;

    // read the line that indicates whether the data is curvature or target frame
    std::string line;
    readline(fin, line); // line 1 (optional version string)
    if (line.substr(0, 10) == "# version ") {
        version = atoi(line.substr(10).c_str());
        line.clear();
    }
    spdlog::info("ffield ver.{}: Reading facet field data from {}", version, fname);

    // line 1 (in old format, this is the first line)
    if (line.empty() || line[0] == '#') {
        readline_nocomments(fin, line);
    }
    if (line != "target_frame" && line != "shape_operator") {
        spdlog::error("{}: Expected 'target_frame' or 'shape_operator'. Got {}.", fname, line);
        return empty_output;
    }

    int nframes = 0;
    readline_nocomments(fin, line); // line 2
    std::istringstream is(line);
    is >> nframes;

    // ignore the final header line
    readline_nocomments(fin, line); // line 3

    // LOAD THE FRAMES
    int idx = 0;
    for (int i = 0; i < nframes; ++i) {
        double k1, k2, vk1[3], vk2[3];
        readline_nocomments(fin, line);
        std::istringstream is(line);
        is >> k1 >> k2 >> vk1[0] >> vk1[1] >> vk1[2] >> vk2[0] >> vk2[1] >> vk2[2];
        if (!fin || !is) {
            spdlog::error("{} facet {}: unable to read tensor. File partially loaded", fname, idx);
            return empty_output;
        }
        /*
CVec3T<double> ev1(vk1[0], vk1[1], vk1[2]);
CVec3T<double> ev2(vk2[0], vk2[1], vk2[2]);
if (0) {
    if (!suppress_tangent_frame_check && fabs(1.0 - lenSq(ev1)) > 1.0e-8) {
        Log.warn(
            "'%s' facet %d: Tensor direction 1 is non-unit (mag %lf). Normalizing. "
            "[Will ignore further warnings.]",
            fname,
            idx,
            len(ev1));
        safe_normalize(ev1);
        suppress_tangent_frame_check = true;
    }
    if (!suppress_tangent_frame_check && fabs(1.0 - lenSq(ev2)) > 1.0e-8) {
        Log.warn(
            "'%s' facet %d: Tensor direction 2 is non-unit (mag %lf). Normalizing. "
            "[Will ignore further warnings.]",
            fname,
            idx,
            len(ev2));
        safe_normalize(ev2);
        suppress_tangent_frame_check = true;
    }
}
TangentFrame frame(k1, k2, ev1, ev2);

// load into target frames
if (load_into_frames) {
    if (!target_frame) // loading shape operator => transform
        frame.inplace_perp();
    fi->setFrame(frame);
}
// load into shape operator
else {
    if (target_frame) // loading target frames => transform
        frame.inplace_inverse_perp();
    if (!frame.is_ortho())
        Log.warn("'%s': loading non-orthogonal frame into shape operator!", fname);
    fi->setShapeOperator(frame);
}
        */
    }

    // read the next token
    line.clear();
    if (!readline_nocomments(fin, line) || (line != "fixed" && line != "crossfield_angles")) {
        spdlog::error(
            "{}: Expected 'crossfield_angles' or 'fixed'. Got {}. File partially loaded.",
            fname,
            line.c_str());
        return empty_output;
    }

    // LOAD THE CROSSFIELD ANGLES
    Eigen::VectorXd crossfield_angles(nframes);
    if (line == "crossfield_angles") {
        line.clear();
        if (!readline_nocomments(fin, line)) {
            spdlog::error("{}: Missing list of crossfield angles. File partially loaded.", fname);
            return empty_output;
        }
        std::istringstream is(line);
        for (int i = 0; i < nframes; ++i) {
            is >> crossfield_angles[i];
        }
        if (!is) {
            spdlog::error("{}: Couldn't read crossfield angles. File partially loaded.", fname);
            return empty_output;
        }

        // read in the next token
        line.clear();
        if (!readline_nocomments(fin, line) || line != "fixed") {
            spdlog::error("{}: Expected 'fixed'. Got {}. File partially loaded.", fname, line);
            return empty_output;
        }
    }

    // LOAD THE INDICES OF THE FIXED FACETS
    if (line != "fixed") {
        spdlog::error("{}: Expected 'fixed'. Got {}. File partially loaded.", fname, line);
        return empty_output;
    }
    bool failed = !readline_nocomments(fin, line);
    if (!failed) {
        int nfixed = -1;
        std::istringstream is(line);
        is >> nfixed;
        std::set<int> fixed_facet_indices;
        for (int i = 0; i < nfixed; ++i) {
            int idx = -1;
            is >> idx;
            fixed_facet_indices.insert(idx);
        }
        if (!is)
            failed = true;
        else {
            // int idx = 0;
            // for (Facet_iterator fi = m->facets_begin(); fi != m->facets_end(); ++fi, ++idx) {
            //     bool fixed = (fixed_facet_indices.find(idx) != fixed_facet_indices.end());
            //     fi->fixed_shapeop() = fixed;
            // }
        }
    }
    if (failed) {
        spdlog::error("{}: Couldn't read indices of fixed facets. File partially loaded.", fname);
        return empty_output;
    }

    // LOAD THE INDICES OF THE FIXED VERTICES
    line.clear();
    if (!readline_nocomments(fin, line)) {
        spdlog::error("{}: Couldn't read next line. File partially loaded.", fname);
        return empty_output;
    }
    if (line == "fixed_vertices") { // old versions don't have this
        bool failed = !readline_nocomments(fin, line);
        if (!failed) {
            int nfixed = -1;
            std::istringstream is(line);
            is >> nfixed;
            std::set<int> fixed_vertex_indices;
            for (int i = 0; i < nfixed; ++i) {
                int idx = -1;
                is >> idx;
                fixed_vertex_indices.insert(idx);
            }
            if (!is)
                failed = true;
            else {
                // int idx = 0;
                // for (Vertex_iterator vi = m->vertices_begin(); vi != m->vertices_end();
                //      ++vi, ++idx) {
                //     bool fixed = (fixed_vertex_indices.find(idx) != fixed_vertex_indices.end());
                //     vi->fixed_shapeop() = fixed;
                // }
            }
        }
        if (failed) {
            spdlog::error(
                "{}: Couldn't read indices of fixed vertices. File partially loaded.",
                fname);
            return empty_output;
        }

        line.clear();
        if (!readline_nocomments(fin, line)) {
            spdlog::error("{}: Couldn't read next line. File partially loaded.", fname);
            return empty_output;
        }
    }

    // LOAD THE MATCHINGS
    if ((line != "matchings" && line != "matchings_and_sharp" && line != "MI_matchings" &&
         line != "MI_matchings_and_sharp")) {
        spdlog::error(
            "{}: Expected one of 'matchings', 'matchings_and_sharp', 'MI_matchings',\n"
            "'MI_matchings_and_sharp'. Got {}. File partially loaded.",
            fname,
            line);
        return empty_output;
    }
    bool load_sharp = (line == "matchings_and_sharp" || line == "MI_matchings_and_sharp");
    // old style matching was not equal to mixed integer matching
    bool old_style_matching = (line.substr(0, 3) != "MI_");

    // set vertex indices (to reorient facet)
    // idx = 0;
    // for (Vertex_iterator vi = m->vertices_begin(); vi != m->vertices_end(); ++vi, ++idx)
    //    vi->tag(idx);

    // clear all matchings
    // for (Halfedge_iterator hi = m->halfedges_begin(); hi != m->halfedges_end(); ++hi) {
    //    if (hi->is_border())
    //        MIM::set_matching(hi, 0);
    //    else
    //        MIM::deinitialize_matching(hi);
    //}

    Eigen::MatrixXi F_id(nframes, 3);
    Eigen::MatrixXi F_m(nframes, 3);
    Eigen::MatrixXi F_sh(nframes, 3);
    bool floating_point_matchings = false;
    idx = 0;
    for (int i = 0; i < nframes; ++i) {
        // read in the vertex indices and matchings
        int id[3], sh[3];
        double m[3];
        double ipart;
        readline_nocomments(fin, line);
        std::istringstream is(line);
        is >> id[0] >> id[1] >> id[2] >> m[0] >> m[1] >> m[2];
        if (load_sharp) is >> sh[0] >> sh[1] >> sh[2];
        if (!fin || !is) {
            spdlog::error("{} facet {}: unable to read matchings. File partially loaded.", fname);
            return empty_output;
        }

        // check if any floating-point atchinatchings were loaded
        floating_point_matchings = floating_point_matchings ||
                                   ((modf(m[0], &ipart) != 0.0) || (modf(m[1], &ipart) != 0.0) ||
                                    (modf(m[2], &ipart) != 0.0));

        F_id.row(i) << id[0], id[1], id[2];
        F_sh.row(i) << sh[0], sh[1], sh[2];
        if (old_style_matching) {
            F_m.row(i) << -int(m[0]), -int(m[1]), -int(m[2]);
        } else {
            F_m.row(i) << int(m[0]), int(m[1]), int(m[2]);
        }
    }

    // copy field into crossfield angles as well unless the crossfield
    // angles were loaded
    // if (load_into_frames && !crossfield_angles_loaded) {
    //    spdlog::info("Crossfield angles not loaded. Transferring from target frame.");
    //    XField::FacetCrossfield_from_TargetFrame(m);
    //}

    // compute the cones from matchings on success
    if (floating_point_matchings) {
        spdlog::warn("Some matchings were floating point.");
    } else {
        spdlog::info("All integer matchings.");
    }

    return std::make_tuple(version, crossfield_angles, F_id, F_m, F_sh);
}

// TODO: Extend to open meshes
std::tuple<Eigen::VectorXi, Eigen::VectorXi> generate_halfedge_from_face_matchings(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    const Eigen::VectorXd& crossfield_angles,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_matchings,
    int version)
{
    int num_faces = m.n_faces();
    int num_halfedges = m.n_halfedges();
    assert(num_faces == F.rows());
    assert(num_faces == F_matchings.rows());

    Eigen::VectorXi reference_halfedges(num_faces);
    Eigen::VectorXi matchings(num_halfedges);
    for (int fijk = 0; fijk < num_faces; ++fijk) {
        // locate first halfedge
        int hij = m.h[fijk];
        while (vtx_reindex[m.v_rep[m.to[hij]]] != F(fijk, 1)) {
            hij = m.n[hij];
        }
        int hjk = m.n[hij];
        int hki = m.n[hjk];

        // verify the facet is correct and load in the matchings
        int vi = vtx_reindex[m.v_rep[m.to[hki]]];
        int vj = vtx_reindex[m.v_rep[m.to[hij]]];
        int vk = vtx_reindex[m.v_rep[m.to[hjk]]];
        if ((vi != F(fijk, 0)) || (vj != F(fijk, 1)) || (vk != F(fijk, 2))) {
            spdlog::error(
                "Facet {} vertex indices ({}, {}, {}) don't match mesh ({}, {}, {}).",
                fijk,
                F(fijk, 0),
                F(fijk, 1),
                F(fijk, 2),
                vi,
                vj,
                vk);
        }

        // Copy matchings to halfedge array
        reference_halfedges[fijk] = hki;
        matchings[hij] = F_matchings(fijk, 1);
        matchings[hjk] = F_matchings(fijk, 2);
        matchings[hki] = F_matchings(fijk, 0);
    }

    // Correct matchings
    const int CURR_VERSION = 2;
    if (version < CURR_VERSION)
        if (MakeMatchingsConsistentWithCrossfield(
                m,
                vtx_reindex,
                reference_halfedges,
                crossfield_angles,
                matchings))
            spdlog::warn("Some matchings were corrected.");

    return std::make_tuple(reference_halfedges, matchings);
}


} // namespace Holonomy
} // namespace Penner
