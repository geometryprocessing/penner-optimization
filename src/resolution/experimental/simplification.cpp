#include "resolution/experimental/simplification.h"

#include "util/vector.h"
#include "holonomy/core/viewer.h"
#include "holonomy/holonomy/holonomy.h"
#include "resolution/experimental/holonomy_coordinates.h"
#include "metric/constraint.h"
#include "util/vector.h"
#include <stdexcept>
#include <queue>

namespace Penner {
namespace Resolution {

VectorX shift_halfedge_array(const Mesh<Scalar>& m, const VectorX& v)
{
    int num_halfedges = m.n_halfedges();
    VectorX w(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        w[hij] = v[m.n[hij]];
    }

    return w;
}

VectorX unshift_halfedge_array(const Mesh<Scalar>& m, const VectorX& v)
{
    int num_halfedges = m.n_halfedges();
    VectorX w(num_halfedges);
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        w[m.n[hij]] = v[hij];
    }

    return w;
}

VectorX convert_tip_to_opposite(const Mesh<Scalar>& m, const VectorX& v)
{
    //return v;
    return shift_halfedge_array(m, v);
}

VectorX convert_opposite_to_tip(const Mesh<Scalar>& m, const VectorX& v)
{
    //return v;
    return unshift_halfedge_array(m, v);
}

IntrinsicSimplificationMesh::IntrinsicSimplificationMesh(const Mesh<Scalar>& m)
    : m_mesh(m)
    , m_is_halfedge_deleted(m.n_halfedges(), false)
    , m_is_vertex_deleted(m.n_vertices(), false)
    , m_is_face_deleted(m.n_faces(), false)
    , m_is_edge_deleted(m.n_edges(), false)
    , m_vertex_parent(arange(m.n_vertices()))
    , m_h2e(build_edge_map(m.opp))
{
    m_e2h = invert_map(m_h2e);
    m_rotation_form = VectorX::Zero(m.n_halfedges());
    simplification_factor = 0.;
    prev = invert_map(m.n);

    // Compute the corner angles
    VectorX he2alpha, he2cot;
    corner_angles(m, he2alpha, he2cot);

    // convert from opposite halfedge to vertex tip representation (needed for general polygons)
    m_alpha = convert_opposite_to_tip(m, he2alpha);

    removed_vert_count = 0;
    initial_vertices = m.n_vertices();

}

VectorX IntrinsicSimplificationMesh::get_opposite_angles() const
{
    const Mesh<Scalar>& m = get_mesh();
    return convert_tip_to_opposite(m, get_holonomy_coordinates());
}

void IntrinsicSimplificationMesh::set_rotation_form(const VectorX& rotation_form)
{
    m_rotation_form = rotation_form;
    assert(Field::is_valid_one_form(m_mesh, rotation_form));
    assert(is_rotation_consistent());
}

void IntrinsicSimplificationMesh::set_dual_loop(const std::vector<std::unique_ptr<DualLoop>>& dual_loops)
{
    Mesh<Scalar>& m = get_mesh();
    for (const auto& dual_loop : dual_loops) {
        std::vector<int> dual_loop_faces = dual_loop->generate_face_sequence(m);
        m_dual_loops.push_back(Holonomy::build_dual_path_from_face_sequence(m, dual_loop_faces));
    }

    int num_dual_loops = dual_loops.size();
    m_kappa.resize(num_dual_loops);
    for (int i = 0; i < num_dual_loops; ++i) {
        // Compute field rotation and metric holonomy
        Scalar rotation = Holonomy::compute_dual_loop_rotation(m, m_rotation_form, *dual_loops[i]);
        Scalar holonomy = Holonomy::compute_dual_loop_holonomy(m, get_opposite_angles(), *dual_loops[i]);

        // Constraint is the difference of the holonomy and rotation
        m_kappa[i] = holonomy - rotation;
    }
}


int IntrinsicSimplificationMesh::compute_degree(int vertex_index) const
{
    const Mesh<Scalar>& m = get_mesh();

    // circulate around the vertex to count outgoing edges
    int h_start = m.out[vertex_index];
    int degree = 0;
    int hij = h_start;
    do {
        degree++;
        hij = m.n[m.opp[hij]];
    } while (hij != h_start);
    //} while ((hij != h_start) && (m.opp[hij] != h_start));

    return degree;
}

void IntrinsicSimplificationMesh::simplify_mesh()
{
    Mesh<Scalar>& m = get_mesh();
    int target = lround(initial_vertices * (1. - simplification_factor));
    int num_vertices = m.n_vertices();
    while (true)
    {
        std::vector<int> perm = generate_permutation(num_vertices, use_random_seed);
        bool vertex_deleted = false;

        // try to delete vertices
        for (int i = 0; i < num_vertices; ++i) {
            if (removed_vert_count >= target) return;

            int vi = perm[i];

            // only delete flat vertices that have not been deleted already
            if (is_deleted_vertex(vi)) continue;
            if (!is_flat_vertex(vi)) continue;

            if (remove_vertex(vi))
            {
                vertex_deleted = true;
                ++removed_vert_count;
            }
        }

        // stop if unable to remove any vertices
        if (!vertex_deleted) break;
    }
}

void IntrinsicSimplificationMesh::collapse_edges()
{
    Mesh<Scalar>& m = get_mesh();
    int num_halfedges = m.n_halfedges();
    int target = lround(initial_vertices * (1. - simplification_factor));

    // Initialize the stack of vertices to process with all vertices
    typedef std::pair<int, Scalar> WeightedEdge;
    auto edge_compare = [](const WeightedEdge& left, const WeightedEdge& right) {
        return left.second > right.second;
    };
    std::priority_queue<WeightedEdge, std::vector<WeightedEdge>, decltype(edge_compare)>
        edges_to_process(edge_compare);
    for (int eij = 0; eij < m_e2h.size(); ++eij) {
        int hij = m_e2h[eij];
        edges_to_process.push(std::make_pair(eij, m.l[hij]));
    }

    while (true)
    {
        std::vector<int> perm = generate_permutation(num_halfedges, use_random_seed);
        bool vertex_deleted = false;

        // try to delete vertices
        for (int i = 0; i < num_halfedges; ++i) {
            if (removed_vert_count >= target) return;

            auto [eij, cost] = edges_to_process.top();
            edges_to_process.pop();
            int hij = m_e2h[eij];
            
            if (is_deleted_halfedge(hij)) continue;

            // only collapse edges with flat vertices at base that have not been deleted already
            int vi = m.to[m.opp[hij]];
            //int vj = m.to[hij];
            if (!is_flat_vertex(vi)) continue;
            if (is_degree_1_vertex(vi))
            {
                int hjk = m.n[hij];
                if (is_flippable_edge(hjk)) flip_ccw(hjk);
            }

            if (collapse_edge(hij))
            {
                vertex_deleted = true;
                ++removed_vert_count;
            }
        }

        // stop if unable to remove any vertices
        if (!vertex_deleted) break;
    }
}

bool IntrinsicSimplificationMesh::remove_vertex(int vertex_index)
{
    // TODO boundary is different case (degree 2)

    // attempt to flip vertex to degree 3
    bool success = flip_to_degree(vertex_index, 3);
    if (!success) return false;

    // remove (now) degree 3 vertex
    return remove_degree_3_vertex(vertex_index);
}

bool IntrinsicSimplificationMesh::will_decrease_degree(int hij) const
{
    if (!is_flippable_edge(hij)) return false;

    const Mesh<Scalar>& m = get_mesh();
    int vi = m.to[m.opp[hij]];
    int vj = m.to[hij];
    int vk = m.to[m.n[hij]];
    int vl = m.to[m.n[m.opp[hij]]];

    // flipped edge no longer adjacent to vi
    if ((vi != vk) && (vi != vl)) return true;

    // original edge doubly adjacent to vi; new edge not
    if ((vi == vj) && ((vi != vl) || (vi != vk))) return true;

    return false;
}

bool IntrinsicSimplificationMesh::will_increase_degree(int hij) const
{
    if (!is_flippable_edge(hij)) return false;

    const Mesh<Scalar>& m = get_mesh();
    int vi = m.to[m.opp[hij]];
    int vj = m.to[hij];
    int vk = m.to[m.n[hij]];
    int vl = m.to[m.n[m.opp[hij]]];

    // original edge not adjacent to vk
    if ((vi != vk) && (vj != vk)) return true;

    // new edge doubly adjacent to vi; original edge not
    if ((vk == vl) && ((vi != vk) || (vj != vk))) return true;

    return false;
}

bool IntrinsicSimplificationMesh::decrease_degree(int vertex_index)
{
    Mesh<Scalar>& m = get_mesh();

    // circulate around the vertex to find a flippable edge
    int h_start = m.out[vertex_index];
    int hij = h_start;
    do {
        // check if flip increases degree
        if (will_decrease_degree(hij)) {
            flip_ccw(hij);
            return true;
        }
        hij = m.n[m.opp[hij]];
    } while (hij != h_start);

    // return false if no flip possible
    return false;
}

bool IntrinsicSimplificationMesh::increase_degree(int vertex_index)
{
    Mesh<Scalar>& m = get_mesh();

    // circulate around the vertex to find a flippable edge
    int h_start = m.out[vertex_index];
    int hij = h_start;
    do {
        int hjk = m.n[hij]; // edge opposite vertex i
        if (will_increase_degree(hjk)) {
            flip_ccw(hjk);
            return true;
        }
        hij = m.n[m.opp[hij]];
    } while (hij != h_start);

    // return false if no flip possible
    return false;
}

void set_rotation_form(const Mesh<Scalar>& m, VectorX& rotation_form, int halfedge_index, Scalar kappa)
{
    rotation_form[halfedge_index] = kappa;
    rotation_form[m.opp[halfedge_index]] = -kappa;
}

void align_rotation_form_faces(const Mesh<Scalar>& m, VectorX& rotation_form, int halfedge_index)
{
    // get facet halfedges
    int hij = halfedge_index;
    int hji = m.opp[hij]; 

    // get rotations across edges
    Scalar rij = rotation_form[hij];
    Scalar rji = -rij;
    spdlog::trace("redistributing rotation {} on halfedge {}", rij, hij);

    // set new rotations, adding half the (signed) rotation of hij to the other halfedges
    set_rotation_form(m, rotation_form, hij, 0);
    int hjk = m.n[hij];
    while (hjk != hij)
    {
        Scalar rjk = rotation_form[hjk];
        set_rotation_form(m, rotation_form, hjk, rjk - rij/2);
        hjk = m.n[hjk];
    }

    int hil = m.n[hji];
    while (hil != hji)
    {
        Scalar ril = rotation_form[hil];
        set_rotation_form(m, rotation_form, hil, ril - rji/2);
        hil = m.n[hil];
    }
}

void IntrinsicSimplificationMesh::align_faces(int halfedge_index)
{
    const Mesh<Scalar>& m = get_mesh();
    VectorX& rotation_form = get_rotation_form();
    align_rotation_form_faces(m, rotation_form, halfedge_index);
}


void IntrinsicSimplificationMesh::flip_ccw(int hij)
{
    Mesh<Scalar>& m = get_mesh();
    spdlog::trace("flipping edge hij={} between vi={} and vj={}", hij, m.to[m.opp[hij]], m.to[hij]);
    if (!is_rotation_consistent()) spdlog::error("initial rotation inconsistent");

    // update rotation data
    align_faces(hij);
    if (!is_rotation_consistent()) spdlog::error("alignment of halfedge {} failed", hij);

    VectorX& alpha = get_holonomy_coordinates();
    update_holonomy_coordinates_under_ccw_flip(m, alpha, hij);

    // update tracked loops
    for (auto& dual_loop : m_dual_loops) {
        Holonomy::update_dual_loop_under_ccw_flip(m, hij, dual_loop);
    }

    // perform inrtrinsic flip
    m.flip_ccw(hij);

    // update prev
    int hji = m.opp[hij];
    for (int h : {hij, m.n[hij], m.n[m.n[hij]]}) prev[m.n[h]] = h;
    for (int h : {hji, m.n[hji], m.n[m.n[hji]]}) prev[m.n[h]] = h;

    spdlog::trace("flipped edge now between vi={} and vj={}", m.to[m.opp[hij]], m.to[hij]);
    if (!is_rotation_consistent()) spdlog::error("flip of halfedge {} failed", hij);
    assert(is_rotation_consistent());
}

bool IntrinsicSimplificationMesh::flip_to_degree(int vertex_index, int target_degree)
{
    SPDLOG_INFO("flipping from {} to {}", compute_degree(vertex_index), target_degree);

    // decrease degree if too high
    while (compute_degree(vertex_index) > target_degree) {
        bool success = decrease_degree(vertex_index);
        SPDLOG_INFO("new degree is {}", compute_degree(vertex_index));
        if (!success) return false;
    }

    // increase degree if too low
    while (compute_degree(vertex_index) < target_degree) {
        bool success = increase_degree(vertex_index);
        SPDLOG_INFO("new degree is {}", compute_degree(vertex_index));
        if (!success) return false;
    }

    // degree is correct at end
    assert(compute_degree(vertex_index) == target_degree);
    return true;
}

bool triple_contains(int i, std::array<int, 3> triple)
{
    if (triple[0] == i) return true;
    if (triple[1] == i) return true;
    if (triple[2] == i) return true;
    return false;
}

void prune_caps(std::vector<Holonomy::DualSegment>& dual_loop)
{
    bool cap_found = true;
    while (cap_found)
    {
        cap_found = false;
        int num_faces = dual_loop.size();
        auto loop_index = [&num_faces](int i) { return (i + num_faces) % num_faces; };
        std::vector<Holonomy::DualSegment> new_dual_loop = {};

        // initialize with back
        new_dual_loop.push_back(dual_loop.back());
        if (new_dual_loop[0][0] == new_dual_loop[0][1])
        {
            cap_found = true;

        }
        int cap_index = -1;
        for (int i = 0; i < num_faces-2; ++i) {
            Holonomy::DualSegment segment = dual_loop[loop_index(i)];
            if ((!cap_found) && (segment[0] == segment[1]))
            {
                spdlog::trace("Pruning cap {}=({}, {})", i, segment[0], segment[1]);
                new_dual_loop.back()[1] = dual_loop[loop_index(i + 1)][1];
                cap_index = i;
                ++i;
                cap_found = true;
            }
            else
            {
                new_dual_loop.push_back(segment);
            }
        }
        if (cap_index != num_faces - 3)
        {
            new_dual_loop.push_back(dual_loop[loop_index(-2)]);
        }
        else
        {
            spdlog::warn("Skipping last segment");
        }
        if (new_dual_loop.back()[0] == new_dual_loop.back()[1]) cap_found = true;
        dual_loop = new_dual_loop;
    }
}

bool IntrinsicSimplificationMesh::remove_degree_3_vertex(int vertex_index)
{
    Mesh<Scalar>& m = get_mesh();
    if (!is_degree_3_vertex(vertex_index)) return false;

    // get inner halfedges of triangle by clockwise circulation
    int vi = vertex_index;
    int hij = m.out[vi];
    int hil = m.n[m.opp[hij]];
    int hik = m.n[m.opp[hil]];

    // get outer halfedges of triangle and check for degenerate topology
    int hjk = m.n[hij];
    int hkl = m.n[hik];
    int hlj = m.n[hil];
    if ((hjk == hkl) || (hkl == hlj) || (hlj == hjk)) return false;

    // get vertex indices and check for degenerate topology
    int vj = m.to[hij];
    int vk = m.to[hik];
    int vl = m.to[hil];
    if ((vj == vk) || (vk == vl) || (vl == vj)) return false;
    if ((vj == vi) || (vk == vi) || (vl == vi)) return false;

    // get face indices
    int fijk = m.f[hjk];
    int fikl = m.f[hkl];
    int filj = m.f[hlj];
    std::array<int, 3> faces = {fijk, fikl, filj};
    spdlog::trace("Collapsing faces {}, {}, {} to {}", fijk, fikl, filj, fijk);
    if ((fijk == fikl) || (fikl == filj) || (filj == fijk)) return false;

    // update angles (flattening the vertex by incorporating rotation across the bisecting edge) 
    VectorX& alpha = get_holonomy_coordinates();
    //alpha[hjk] = alpha[hik] + alpha[m.opp[hij]] + get_rotation(hil);
    //alpha[hkl] = alpha[hil] + alpha[m.opp[hik]] + get_rotation(hij);
    //alpha[hlj] = alpha[hij] + alpha[m.opp[hil]] + get_rotation(hik);
    //Scalar area = alpha[hjk] + alpha[hkl] + alpha[hlj];
    alpha[hjk] = alpha[hik] + alpha[m.opp[hij]] + get_rotation(hil);
    alpha[hkl] = alpha[hil] + alpha[m.opp[hik]] + get_rotation(hij);
    alpha[hlj] = alpha[hij] + alpha[m.opp[hil]] + get_rotation(hik);
    Scalar area = alpha[hjk] + alpha[hkl] + alpha[hlj];
    if (!float_equal(area, PI))
    {
        spdlog::warn("vertex removal {} produced triangle area {}", vi, area);
    }

    // mark deleted elements
    delete_edge(m_h2e[hij]);
    delete_edge(m_h2e[hik]);
    delete_edge(m_h2e[hil]);
    delete_vertex(vi);
    delete_face(fikl);
    delete_face(filj);

    // update dual loops
    auto& dual_loops = get_dual_loops();
    for (std::vector<Holonomy::DualSegment>& dual_loop : dual_loops) {
        spdlog::trace("Simplifying loop of length {}", dual_loop.size());
        int num_faces = dual_loop.size();
        auto loop_index = [&num_faces](int i) { return (i + num_faces) % num_faces; };
        std::vector<Holonomy::DualSegment> new_dual_loop = {};
        int offset = 0;
        while (triple_contains(m.f[dual_loop[loop_index(offset)][0]], faces)
        && (triple_contains(m.f[dual_loop[loop_index(offset-1)][0]], faces)))
        {
            offset++;
        }
        for (int i = offset; i < num_faces + offset; ++i) {
            Holonomy::DualSegment segment = dual_loop[loop_index(i)];
            int h_start = segment[0];
            int curr_face = m.f[h_start];

            if (triple_contains(curr_face, faces))
            {
                spdlog::trace("Simplifying segment {}=({}, {}) in face {}", i, segment[0], segment[1],  curr_face);

                segment = dual_loop[loop_index(i + 1)];
                int next_face = m.f[segment[0]];
                while (triple_contains(next_face, faces))
                {
                    spdlog::trace("segment {}=({}, {}) in face {}", i, segment[0], segment[1],  next_face);
                    ++i;
                    segment = dual_loop[loop_index(i + 1)];
                    next_face = m.f[segment[0]];
                }
                spdlog::trace("Following segment {}=({}, {}) in face {}", i, segment[0], segment[1],  next_face);
            }

            int h_end = dual_loop[loop_index(i)][1];
            new_dual_loop.push_back({h_start, h_end});
        }

        // overwrite dual loop
        prune_caps(new_dual_loop);
        dual_loop = new_dual_loop;
    }

    // update halfedge topology
    m.n[hjk] = hkl;
    m.n[hkl] = hlj;
    m.n[hlj] = hjk;

    // update face topology
    m.f[hjk] = fijk;
    m.f[hkl] = fijk;
    m.f[hlj] = fijk;
    m.h[fijk] = hjk;

    // update vertex topology (to still valid)
    m.out[vj] = hjk;
    m.out[vk] = hkl;
    m.out[vl] = hlj;

    for (std::vector<Holonomy::DualSegment>& dual_loop : dual_loops) {
        spdlog::trace("Checking loop of length {}", dual_loop.size());
        if (!Holonomy::is_valid_dual_loop(m, dual_loop))
        {
            throw std::runtime_error("Dual loop simplification failed");
        }
    }

    // attempt to flip to regular vertex
    bool regularize = false;
    if (regularize)
    {
        flip_to_degree(vj, 6);
        flip_to_degree(vk, 6);
        flip_to_degree(vl, 6);
    }

    if (!is_rotation_consistent()) spdlog::error("removal of vertex {} failed", vertex_index);
    assert(is_rotation_consistent());

    return true;
}

// algebraically rotate a given edge counterclockwise
void IntrinsicSimplificationMesh::rotate_edge(int halfedge_index, Scalar theta)
{
    Mesh<Scalar>& m = get_mesh();
    int hij = halfedge_index;
    int hji = m.opp[hij];
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int hil = m.n[hji];
    int hlj = m.n[hil];

    VectorX& alpha = get_holonomy_coordinates();
    alpha[hki] += theta;
    alpha[hij] -= theta;
    alpha[hlj] += theta;
    alpha[hjk] -= theta;
}

void relabel_tip(Mesh<Scalar>& m, int hij, int v)
{
    int hkj = hij;
    do
    {
        m.to[hkj] = v;
        hkj = m.opp[m.n[hkj]];
    } while (hkj != hij);
    m.out[v] = m.opp[hij];
}

void relabel_face(Mesh<Scalar>& m, int hij, int f)
{
    int hjk = hij;
    do
    {
        m.f[hjk] = f;
        hjk = m.n[hjk];
    } while (hjk != hij);
    m.h[f] = hij;
}

int IntrinsicSimplificationMesh::create_face()
{
    Mesh<Scalar>& m = get_mesh();
    int index = m.h.size();
    m.h.push_back(-1);
    m_is_face_deleted.push_back(false);
    return index;
}

int IntrinsicSimplificationMesh::create_vertex()
{
    Mesh<Scalar>& m = get_mesh();
    int index = m.out.size();
    m.out.push_back(-1);
    m.v_rep.push_back(-1);
    m_is_vertex_deleted.push_back(false);
    return index;
}

// preserve tip/face of h if vertex/face deleted
bool IntrinsicSimplificationMesh::splice_halfedges(int h, int g)
{
    Mesh<Scalar>& m = get_mesh();
    int hn = m.n[h];
    int gn = m.n[g];
    m.n[h] = gn;
    m.n[g] = hn;
    prev[hn] = g;
    prev[gn] = h;

    // create new face if splice separates
    if (m.f[h] == m.f[g])
    {
        int new_f = create_face();
        relabel_face(m, g, new_f);
        relabel_face(m, h, m.f[h]); // need to properly set f2he
    }
    // otherwise relabel shared face and delete removed face
    else
    {
        delete_face(m.f[g]);
        relabel_face(m, g, m.f[h]);
    }

    // create new vertex if splice separates
    if (m.to[h] == m.to[g])
    {
        int new_v = create_vertex();
        m.v_rep[new_v] = m.v_rep[m.to[h]];
        relabel_tip(m, g, new_v);
        relabel_tip(m, h, m.to[h]); // need to properly set out
    }
    // otherwise relabel shared vertex and delete removed vertex
    else
    {
        delete_vertex(m.to[g]);
        relabel_tip(m, g, m.to[h]);
    }

    return true;
}


bool IntrinsicSimplificationMesh::join_halfedge_endpoints(int halfedge_index)
{
    Mesh<Scalar>& m = get_mesh();
    int hij = halfedge_index;
    int hji = m.opp[hij];
    int hki = prev[hij];
    int hlj = prev[hji];
    int vi = m.to[hji];
    int vj = m.to[hij];
    spdlog::info("joining edge {}, {}", vi, vj);
    assert(hki != hij);
    assert(hki != hji);
    assert(hlj != hij);
    assert(hlj != hji);

    VectorX& alpha = get_holonomy_coordinates();
    alpha[hki] = alpha[hki] + alpha[hij] - M_PI;
    alpha[hlj] = alpha[hlj] + alpha[hji] - M_PI;

    align_faces(hij);

    splice_halfedges(hlj, hij); // hij removed, so splice second
    splice_halfedges(hki, hji); // hji removed, so splice second
    splice_halfedges(hlj, hki); // want to preserve vj, so splice hlj first

    // delete removed edge
    delete_edge(m_h2e[hij]);
    delete_face(m.f[hij]);
    delete_face(m.f[m.opp[hij]]);
    delete_vertex(m.to[hij]);
    delete_vertex(m.to[m.opp[hij]]);
    assert(m.opp[hij] == m.n[hij]);
    assert(m.n[m.n[hij]] == hij);
    assert(m.n[m.n[m.opp[hij]]] == m.opp[hij]);
    assert(m.f[hij] == m.f[m.opp[hij]]);

    return true;
}

bool IntrinsicSimplificationMesh::join_edge_faces(int edge_index)
{
    Mesh<Scalar>& m = get_mesh();
    int hij = edge_index;
    int hji = m.opp[hij];
    int hki = prev[hij];
    int hlj = prev[hji];
    assert(m.f[hij] != m.f[hji]);
    
    VectorX& alpha = get_holonomy_coordinates();
    alpha[hki] = alpha[hki] + alpha[hji];
    alpha[hlj] = alpha[hlj] + alpha[hij];
    //alpha[hki] = alpha[hki];
    //alpha[hlj] = alpha[hlj];

    align_faces(hij);

    splice_halfedges(prev[hji], hij); // hij removed, so splice second
    splice_halfedges(prev[hij], hji); // hji removed, so splice second
    

    // delete removed edge
    delete_edge(m_h2e[hij]);
    delete_face(m.f[hij]);
    delete_face(m.f[m.opp[hij]]);
    delete_vertex(m.to[hij]);
    delete_vertex(m.to[m.opp[hij]]);
    assert(m.opp[hij] == m.n[hij]);
    assert(m.n[m.n[hij]] == hij);
    assert(m.n[m.n[m.opp[hij]]] == m.opp[hij]);
    assert(m.f[hij] == m.f[m.opp[hij]]);

    return true;
}

int count_adjacency(const Mesh<Scalar>& m, int vi, int v)
{
    int hij = m.out[vi];
    int hkj = hij;
    int count = 0;
    do
    {
        if (m.to[hkj] == v) count++;
        hkj = m.n[m.opp[hkj]];
    } while (hkj != hij);

    return count;
}

bool IntrinsicSimplificationMesh::is_collapse_possible(int edge_index) const
{
    const Mesh<Scalar>& m = get_mesh();

    // get halfedge indices of quad
    int hij = edge_index;
    int hji = m.opp[hij];
    int hjk = m.n[hij];
    int hil = m.n[hji];

    // get oppposite halfedges
    int hkj = m.opp[hjk];
    int hli = m.opp[hil];

    // get vertex indices and check for degenerate topology
    int vi = m.to[hji];
    int vj = m.to[hij];

    // check if possible
    if (is_degree_1_vertex(vi)) return false;
    if (is_degree_1_vertex(vj)) return false;
    if (m.f[hjk] == m.f[hkj]) return false;
    if (m.f[hil] == m.f[hli]) return false;
    if (vi == vj)
    {
        spdlog::warn("trying to collapse edge with identical endpoints");
        //spdlog::info("face halfedges {}, {}, {}", hij, hjk, hli);
        //spdlog::info("opoosite face halfedges {}, {}, {}", hji, hil, hlj);
        return false;
    }

    if (maintain_simplicial)
    {
        // check if either endpoint is multiply adjacent (and collapse will make self loop)
        if (count_adjacency(m, vi, vj) > 1) return false;
        if (count_adjacency(m, vj, vi) > 1) return false;
    }

    return true;
}

bool IntrinsicSimplificationMesh::collapse_edge(int edge_index)
{
    Mesh<Scalar>& m = get_mesh();

    // get halfedge indices of quad
    int hij = edge_index;
    int hji = m.opp[hij];
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int hil = m.n[hji];
    int hlj = m.n[hil];
    assert(m.n[m.n[m.n[hij]]] == hij); // should be triangle
    assert(m.n[m.n[m.n[hji]]] == hji); // should be triangle

    // get oppposite halfedges
    int hkj = m.opp[hjk];
    int hik = m.opp[hki];
    int hli = m.opp[hil];
    int hjl = m.opp[hlj];

    // get vertex indices and check for degenerate topology
    int vi = m.to[hji];
    int vj = m.to[hij];
    int vk = m.to[hjk];
    int vl = m.to[hil];
    spdlog::info("collapsing edge {}, {} opposite {} and {}", vi, vj, vk, vl);

    // get face indices
    int fijk = m.f[hij];
    int fjil = m.f[hji];
    spdlog::info("faces {}, {} to collapse", fijk, fjil);

    // check if possible
    if (!is_collapse_possible(hij)) return false;

    bool modify_directly = false;
    if (modify_directly)
    {
        // rotate edges to eliminate collapsed edge
        VectorX& alpha = get_holonomy_coordinates();
        rotate_edge(hjk, alpha[hjk] / 2.);
        rotate_edge(hki, -alpha[hjk] / 2.);
        rotate_edge(hil, alpha[hil] / 2.);
        rotate_edge(hlj, -alpha[hil] / 2.);

        // align faces across edge
        align_faces(hij);

        // combine rotations across edges (will become equivalent after collapse)
        Scalar rjk = get_rotation(hjk);
        Scalar rki = get_rotation(hki);
        Scalar ril = get_rotation(hil);
        Scalar rlj = get_rotation(hlj);
        set_rotation_form(hjk, rjk - rki);
        set_rotation_form(hki, rki - rjk);
        set_rotation_form(hil, ril - rlj);
        set_rotation_form(hlj, rlj - ril);

        // update vertex topology (face topology still valid)
        relabel_tip(m, hji, vj); // must do before touching halfedge connectivity
        m.out[vj] = hjl;
        m.out[vk] = hkj;
        m.out[vl] = hli;

        // update halfedge topology 
        m.opp[hkj] = hik;
        m.opp[hik] = hkj;
        m.opp[hli] = hjl;
        m.opp[hjl] = hli;
        
        // keep split mesh halfedge topology valid as split torus with removed vertex
        m.opp[hjk] = hil;
        m.opp[hki] = hlj;
        m.opp[hil] = hjk;
        m.opp[hlj] = hki;
        relabel_tip(m, hij, vi); // must do after changing halfedge connectivity
        m.out[vi] = hij;

        // mark deleted elements
        delete_edge(m_h2e[hij]);
        delete_halfedge(hjk);
        delete_halfedge(hki);
        delete_halfedge(hil);
        delete_halfedge(hlj);
        delete_vertex(vi);
        delete_face(fijk);
        delete_face(fjil);
    }
    else
    {
        join_halfedge_endpoints(hij);
        if (!is_deleted_face(m.f[hjk])) join_edge_faces(hjk);
        if (!is_deleted_face(m.f[hil])) join_edge_faces(hil);

        /*
        delete_edge(hij);
        delete_edge(hjk);
        delete_edge(hil);
        delete_vertex(vi);
        delete_face(fijk);
        delete_face(fjil);

        bool fixed_vertex = false;
        if (!is_deleted_edge(hki))
        {
            relabel_tip(m, hki, vj);
            m.out[vj] = m.opp[hki];
            m.out[vk] = hki;
            int f0_new = m.f[m.n[hki]];
            if ((f0_new == fijk) || (f0_new == fjil))
            {
                spdlog::warn("setting to deleted face");
                f0_new = m.f[prev[hki]];
            }
            m.f[hki] = f0_new;
            m.h[f0_new] = hki;
            fixed_vertex = true;
        }
        if (!is_deleted_edge(hlj))
        {
            relabel_tip(m, hlj, vj);
            m.out[vj] = m.opp[hlj];
            m.out[vl] = hlj;
            int f1_new = m.f[m.n[hlj]];
            if ((f1_new == fijk) || (f1_new == fjil))
            {
                spdlog::warn("setting to deleted face");
                f1_new = m.f[prev[hlj]];
            }
            m.f[hlj] = f1_new;
            m.h[f1_new] = hlj;
            fixed_vertex = true;
        }
        if (!fixed_vertex) spdlog::error("vertex index not corrected");
        assert(fixed_vertex);
        */
    }

    //if (!are_polygon_mesh_edges_valid(m.n, prev)) spdlog::error("mesh edges not valid");
    //if (!is_one_sided_inverse(m.opp, m.opp)) spdlog::error("mesh opp map not valid");
    //assert(is_one_sided_inverse(m.opp, m.opp));

    return true;
}

void IntrinsicSimplificationMesh::delete_edge(int edge_index)
{
    const Mesh<Scalar>& m = get_mesh();
    int hij = m_e2h[edge_index];
    int hji = m.opp[hij];
    m_is_edge_deleted[edge_index] = true;
    delete_halfedge(hij);
    delete_halfedge(hji);
}

void IntrinsicSimplificationMesh::delete_halfedge(int halfedge_index)
{
    m_is_halfedge_deleted[halfedge_index] = true;
}

void IntrinsicSimplificationMesh::delete_vertex(int vertex_index)
{
    m_is_vertex_deleted[vertex_index] = true;
}

void IntrinsicSimplificationMesh::delete_face(int face_index)
{
    m_is_face_deleted[face_index] = true;
}

bool IntrinsicSimplificationMesh::is_flat_vertex(int vertex_index) const
{
    const Mesh<Scalar>& m = get_mesh();
    return (float_equal<Scalar>(m.Th_hat[m.v_rep[vertex_index]], 2. * M_PI));
}

bool IntrinsicSimplificationMesh::is_degree_1_vertex(int vertex_index) const
{
    return (compute_degree(vertex_index) == 1);
}

bool IntrinsicSimplificationMesh::is_degree_3_vertex(int vertex_index) const
{
    return (compute_degree(vertex_index) == 3);
}

bool IntrinsicSimplificationMesh::is_flippable_edge(int halfedge_index) const
{
    const Mesh<Scalar>& m = get_mesh();
    int hij = halfedge_index;
    int hji = m.opp[hij];
    int vi = m.to[hji];
    int vj = m.to[hij];
    int vk = m.to[m.n[hij]];
    int vl = m.to[m.n[hji]];

    // optionally prevent flip from or to self edge
    if (maintain_simplicial)
    {
        if (vi == vj) return false;
        if (vk == vl) return false;
    }

    // can't flip edges adjacent to degree one vertex
    if (is_degree_1_vertex(vi)) return false;
    if (is_degree_1_vertex(vj)) return false;

    return true;
}

/*
std::tuple<std::vector<int>, std::vector<int>> index_s(const std::vector<bool>& is_deleted)
{
    // Count sets by counting roots
    int num_elements = is_deleted.size();

    // Make map from roots to sets
    std::vector<int> reduced;
    std::vector<int> set_index(num_elements, -1);
    for (int i = 0; i < num_elements; ++i) {
        if (!is_deleted[i]) {
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
*/

Mesh<Scalar> IntrinsicSimplificationMesh::get_simplified_mesh() const
{
    Mesh<Scalar> m_simp;
    const Mesh<Scalar>& m = get_mesh();

    // enumerate simplified elements
    std::vector<int> f_simp, removed_f, f_map;
    std::vector<int> v_simp, removed_v, v_map;
    std::vector<int> h_simp, removed_h, h_map;
    enumerate_boolean_array(m_is_face_deleted, removed_f, f_simp, f_map);
    enumerate_boolean_array(m_is_vertex_deleted, removed_v, v_simp, v_map);
    enumerate_boolean_array(m_is_halfedge_deleted, removed_h, h_simp, h_map);

    // compute maps with simplification remappings
    m_simp.n = vector_compose(h_map, vector_compose(m.n, h_simp));
    m_simp.to = vector_compose(v_map, vector_compose(m.to, h_simp));
    m_simp.f = vector_compose(f_map, vector_compose(m.f, h_simp));
    m_simp.h = vector_compose(h_map, vector_compose(m.h, f_simp));
    m_simp.out = vector_compose(h_map, vector_compose(m.out, v_simp));
    m_simp.opp = vector_compose(h_map, vector_compose(m.opp, h_simp));
    m_simp.l = vector_reindex(m.l, h_simp);
    m_simp.type = vector_reindex(m.type, h_simp);
    m_simp.type_input = vector_reindex(m.type_input, h_simp);
    m_simp.R = vector_reindex(m.R, h_simp);
    m_simp.v_rep = vector_compose(v_map, vector_compose(m.v_rep, v_simp));

    // TODO Need independent vertex reindexing
    m_simp.Th_hat = vector_reindex(m.Th_hat, v_simp);
    m_simp.fixed_dof = std::vector<bool>(m_simp.n_ind_vertices(), false);
    m_simp.fixed_dof[0] = true;
    int n_v = v_simp.size();
    int n_e = h_simp.size() / 2;
    int n_f = f_simp.size();
    spdlog::warn("Fixing arbitrary simplified vertex degree of freedom");
    spdlog::info("simplified mesh with |V|={}, |E|={}, |F|={}", n_v, n_e, n_f);
    spdlog::info("Euler characteristic is {}", n_v - n_e + n_f);
    if (!Holonomy::is_valid_mesh(m_simp)) spdlog::error("simplification failed");
    assert(Holonomy::is_valid_mesh(m_simp));

    return m_simp;
}

void IntrinsicSimplificationMesh::garbage_collection()
{
    Mesh<Scalar>& m = get_mesh();

    // enumerate simplified elements
    std::vector<int> f_simp, removed_f, f_map;
    std::vector<int> v_simp, removed_v, v_map;
    std::vector<int> h_simp, removed_h, h_map;
    std::vector<int> e_simp, removed_e, e_map;
    enumerate_boolean_array(m_is_face_deleted, removed_f, f_simp, f_map);
    enumerate_boolean_array(m_is_vertex_deleted, removed_v, v_simp, v_map);
    enumerate_boolean_array(m_is_halfedge_deleted, removed_h, h_simp, h_map);
    enumerate_boolean_array(m_is_edge_deleted, removed_e, e_simp, e_map);

    // compute maps with simplification remappings
    m.n = vector_compose(h_map, vector_compose(m.n, h_simp));
    m.to = vector_compose(v_map, vector_compose(m.to, h_simp));
    m.f = vector_compose(f_map, vector_compose(m.f, h_simp));
    m.h = vector_compose(h_map, vector_compose(m.h, f_simp));
    m.out = vector_compose(h_map, vector_compose(m.out, v_simp));
    m.opp = vector_compose(h_map, vector_compose(m.opp, h_simp));
    m.l = vector_reindex(m.l, h_simp);
    m.type = vector_reindex(m.type, h_simp);
    m.type_input = vector_reindex(m.type_input, h_simp);
    m.R = vector_reindex(m.R, h_simp);
    m.v_rep = vector_compose(v_map, vector_compose(m.v_rep, v_simp));

    // TODO Need independent vertex reindexing
    m.Th_hat = vector_reindex(m.Th_hat, v_simp);
    m.fixed_dof = std::vector<bool>(m.n_ind_vertices(), false);
    m.fixed_dof[0] = true;
    int n_v = v_simp.size();
    int n_e = h_simp.size() / 2;
    int n_f = f_simp.size();
    spdlog::warn("Fixing arbitrary simplified vertex degree of freedom");
    spdlog::info("simplified mesh with |V|={}, |E|={}, |F|={}", n_v, n_e, n_f);
    spdlog::info("Euler characteristic is {}", n_v - n_e + n_f);
    if (!Holonomy::is_valid_mesh(m)) spdlog::error("simplification failed");
    assert(Holonomy::is_valid_mesh(m));

    m_is_halfedge_deleted = std::vector<bool>(m.n_halfedges(), false);
    m_is_vertex_deleted = std::vector<bool>(m.n_vertices(), false);
    m_is_face_deleted = std::vector<bool>(m.n_faces(), false);
    m_is_edge_deleted = std::vector<bool>(m.n_edges(), false);
    m_vertex_parent = vector_compose(m_vertex_parent, v_simp);
    m_h2e = vector_compose(e_map, vector_compose(m_h2e, h_simp));
    m_e2h = vector_compose(h_map, vector_compose(m_e2h, e_simp));
    m_rotation_form = vector_compose(m_rotation_form, h_simp);
    m_alpha = vector_compose(m_alpha, h_simp);
    prev = vector_compose(h_map, vector_compose(prev, h_simp));
}

VectorX IntrinsicSimplificationMesh::get_simplified_rotation_form() const
{
    // remap the rotation form to the simplified mesh indexing
    std::vector<int> h_simp, removed_h, h_map;
    enumerate_boolean_array(m_is_halfedge_deleted, removed_h, h_simp, h_map);
    return vector_compose(get_rotation_form(), h_simp);
}

VectorX IntrinsicSimplificationMesh::get_simplified_holonomy_coordinates() const
{
    // remap the rotation form to the simplified mesh indexing
    std::vector<int> h_simp, removed_h, h_map;
    enumerate_boolean_array(m_is_halfedge_deleted, removed_h, h_simp, h_map);
    return vector_compose(get_opposite_angles(), h_simp);
}

MarkedPennerConeMetric IntrinsicSimplificationMesh::get_simplified_marked_metric(
    const MarkedMetricParameters& marked_metric_params
) const
{
    // build the remapped simplified mesh
    Mesh<Scalar> m = get_simplified_mesh();

    // Build initial metric coordinates
    int num_halfedges = m.n_halfedges();
    VectorX metric_coords = VectorX::Zero(num_halfedges);
    m.l = std::vector<Scalar>(num_halfedges, 1.);
    spdlog::warn("Overriding simplified length with uniform");

    // compute basis loops
    std::vector<std::unique_ptr<DualLoop>> basis_loops;
    generate_basis_loops(m, basis_loops, marked_metric_params);

    // remap the rotation form to the simplified mesh indexing
    VectorX rotation_form = get_simplified_rotation_form();
    VectorX alpha = get_simplified_holonomy_coordinates();
    spdlog::info("rotations in range [{}, {}]", rotation_form.maxCoeff(), rotation_form.minCoeff());
    spdlog::info("angles in range [{}, {}]", alpha.maxCoeff(), alpha.minCoeff());

    // compute rotation
    int num_basis_loops = basis_loops.size();
    std::vector<Scalar> kappa(num_basis_loops);
    for (int i = 0; i < num_basis_loops; ++i) {
        // Compute field rotation and metric holonomy
        Scalar rotation = Holonomy::compute_dual_loop_rotation(m, rotation_form, *basis_loops[i]);
        Scalar holonomy = Holonomy::compute_dual_loop_holonomy(m, alpha, *basis_loops[i]);

        // Constraint is the difference of the holonomy and rotation
        kappa[i] = holonomy - rotation;
    } 
    
    int num_ind_vertices = m.n_ind_vertices();
    std::vector<Scalar> Th_hat(num_ind_vertices, 0.);
    for (int h = 0; h < m.n_halfedges(); h++) {
        // Add angle to vertex opposite the halfedge
        Th_hat[m.v_rep[m.to[m.n[h]]]] += alpha[h];

        // Add rotation to the vertex at the tip of the halfedge
        // NOTE: By signing convention, this is the negative of the rotation ccw around
        // the vertex
        Th_hat[m.v_rep[m.to[h]]] += rotation_form[h];
    }

    for (int vi = 0; vi < num_ind_vertices; ++vi)
    {
        if (!float_equal(m.Th_hat[vi], Th_hat[vi]))
        {
            spdlog::error("cone angle {} of {} inconsistent with simplified cone {}", m.Th_hat[vi], vi, Th_hat[vi]);
        }
    }

    m.Th_hat = Th_hat;

    return MarkedPennerConeMetric(m, metric_coords, basis_loops, kappa);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> IntrinsicSimplificationMesh::generate_mesh(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex) const
{
    const Mesh<Scalar>& m = get_mesh();

    // find unremoved vertices
    int num_vertices = m.n_vertices();
    std::vector<int> V_map(num_vertices, -1);
    std::vector<int> simplified_vertices = {};
    for (int vi = 0; vi < num_vertices; ++vi) {
        if (is_deleted_vertex(vi)) continue;
        V_map[vi] = simplified_vertices.size();
        simplified_vertices.push_back(vi);
    }

    // copy unremoved vertices
    int num_simp_vertices = simplified_vertices.size();
    Eigen::MatrixXd V_simp(num_simp_vertices, 3);
    for (int i = 0; i < num_simp_vertices; ++i) {
        int vi = simplified_vertices[i];
        V_simp.row(i) = V.row(vtx_reindex[vertex_parent(vi)]);
    }

    // find unremoved faces
    int num_faces = m.n_faces();
    std::vector<int> simplified_faces = {};
    for (int f = 0; f < num_faces; ++f) {
        if (is_deleted_face(f)) continue;
        simplified_faces.push_back(f);
    }

    // generate doubled faces and halfedge map
    int num_simp_faces = simplified_faces.size();
    Eigen::MatrixXi F(num_simp_faces, 3);
    for (int i = 0; i < num_simp_faces; ++i) {
        int fijk = simplified_faces[i];

        // Get halfedges of face
        int hij = m.h[fijk];
        int hjk = m.n[hij];
        int hki = m.n[hjk];

        // Get vertices of face
        int vj = m.to[hij];
        int vk = m.to[hjk];
        int vi = m.to[hki];

        // Write face with opposite vertex data
        F(i, 0) = V_map[vi];
        F(i, 1) = V_map[vj];
        F(i, 2) = V_map[vk];
    }

    return std::make_tuple(V_simp, F);
}

void IntrinsicSimplificationMesh::view_simplified_mesh(
    const Eigen::MatrixXd& V,
    const std::vector<int>& vtx_reindex) const
{
    auto [V_double, F] = generate_mesh(V, vtx_reindex);
    // spdlog::trace("F: {}", F);

#ifdef ENABLE_VISUALIZATION
    std::string mesh_handle = "simplified mesh";
    polyscope::init();
    polyscope::registerSurfaceMesh(mesh_handle, V_double, F);
    polyscope::show();
#endif
}

void IntrinsicSimplificationMesh::get_dual_loops(
    std::vector<std::unique_ptr<DualLoop>>& dual_loops) const
{
    // enumerate simplified elements
    std::vector<int> h_simp, removed_h, h_map;
    enumerate_boolean_array(m_is_halfedge_deleted, removed_h, h_simp, h_map);

    // Initialize basis list loop and lambda to add loops
    dual_loops.reserve(m_dual_loops.size());
    for (const auto& dual_loop : m_dual_loops)
    {
        int dual_loop_size = dual_loop.size();
        std::vector<Holonomy::DualSegment> simp_dual_loop;
        simp_dual_loop.reserve(dual_loop_size);
        for (int i = 0; i < dual_loop_size; ++i)
        {
            auto [hij, hjk] = dual_loop[i];
            simp_dual_loop.push_back(Holonomy::DualSegment({h_map[hij], h_map[hjk]}));
        }
        dual_loops.push_back(std::make_unique<Holonomy::DualLoopConnectivity>(
            Holonomy::DualLoopConnectivity(simp_dual_loop)));
    }
}

bool IntrinsicSimplificationMesh::is_rotation_consistent() const
{
#ifdef CHECK_VALIDITY
    const Mesh<Scalar>& m = get_mesh();
    const VectorX& alpha = get_opposite_angles();
    const VectorX& rotation_form = get_rotation_form();
    const auto& basis_loops = get_dual_loops();

    // compute cone angles
    int num_ind_vertices = m.n_ind_vertices();
    std::vector<Scalar> Th_hat(num_ind_vertices, 0.);
    for (int h = 0; h < m.n_halfedges(); h++) {
        if (is_deleted_halfedge(h)) continue;

        // Add angle to vertex opposite the halfedge
        Th_hat[m.v_rep[m.to[m.n[h]]]] += alpha[h];

        // Add rotation to the vertex at the tip of the halfedge
        // NOTE: By signing convention, this is the negative of the rotation ccw around
        // the vertex
        Th_hat[m.v_rep[m.to[h]]] += rotation_form[h];
    }

    for (int vi = 0; vi < num_ind_vertices; ++vi)
    {
        if (is_deleted_vertex(vi)) continue;

        if (!float_equal(m.Th_hat[vi], Th_hat[vi], 1e-4))
        {
            spdlog::error("cone angle {} of {} inconsistent with simplified cone {}", m.Th_hat[vi], vi, Th_hat[vi]);
            return false;
        }
    }

    // check dual loop holonomy
    int num_basis_loops = basis_loops.size();
    for (int i = 0; i < num_basis_loops; ++i) {
        // Compute field rotation and metric holonomy
        Holonomy::DualLoopList basis_loop(basis_loops[i]);
        Scalar rotation = Holonomy::compute_dual_loop_rotation(m, rotation_form, basis_loop);
        Scalar holonomy = Holonomy::compute_dual_loop_holonomy(m, alpha, basis_loop);

        // Constraint is the difference of the holonomy and rotation
        if (!float_equal(m_kappa[i], holonomy - rotation, 1e-4))
        {
            spdlog::error("holonomy {} of {} inconsistent with simplified cone {}", m_kappa[i], i, holonomy-rotation);
            return false;
        }
    }

#endif

    return true;
}

} // namespace Resolution 
} // namespace Penner
