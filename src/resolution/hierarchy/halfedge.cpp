#include "resolution/hierarchy/halfedge.h"

#include "util/vector.h"

namespace Penner {
namespace Resolution {

HalfedgeMesh::HalfedgeMesh() {
    m_halfedge_count = 0;
    m_vertex_count = 0;
    m_edge_count = 0;
    m_face_count = 0;
};

HalfedgeMesh::HalfedgeMesh(
    const std::vector<int>& opp_,
    const std::vector<int>& n_,
    const std::vector<int>& to_,
    const std::vector<int>& face_,
    const std::vector<int>& h2e_)
: opp(opp_)
, n(n_)
, to(to_)
, face(face_)
, h2e(h2e_)
{
    // build left inverse maps
    prev = invert_map(n);
    out = invert_map(vector_compose(to, opp));
    f2h = invert_map(face);
    e2h = invert_map(h2e);

    // build deleted element flags
    m_is_halfedge_deleted = std::vector<bool>(prev.size(), false);
    m_is_vertex_deleted = std::vector<bool>(out.size(), false);
    m_is_edge_deleted = std::vector<bool>(e2h.size(), false);
    m_is_face_deleted = std::vector<bool>(f2h.size(), false);

    // get initial counts
    m_halfedge_count = prev.size();
    m_vertex_count = out.size();
    m_edge_count = e2h.size();
    m_face_count = f2h.size();

    // preallocate extra space
    opp.reserve(5 * m_halfedge_count);
    n.reserve(5 * m_halfedge_count);
    prev.reserve(5 * m_halfedge_count);
    to.reserve(5 * m_halfedge_count);
    face.reserve(5 * m_halfedge_count);
    h2e.reserve(5 * m_halfedge_count);
    e2h.reserve(5 * m_edge_count);
    out.reserve(5 * m_vertex_count);
    f2h.reserve(5 * m_face_count);
    m_is_halfedge_deleted.reserve(5 * m_halfedge_count);
    m_is_vertex_deleted.reserve(5 * m_vertex_count);
    m_is_edge_deleted.reserve(5 * m_edge_count);
    m_is_face_deleted.reserve(5 * m_face_count);
}

int HalfedgeMesh::vertex_degree(int vertex_index) const
{
    int hij = out[vertex_index];
    int hik = hij;
    int degree = 0;
    do
    {
        degree++;
        hik = opp[prev[hik]];
    } while (hik != hij);

    return degree;
}

int HalfedgeMesh::face_degree(int face_index) const
{
    int hij = f2h[face_index];
    int hkl = hij;
    int degree = 0;
    do
    {
        degree++;
        hkl = n[hkl];
    } while (hkl != hij);

    return degree;
}

int HalfedgeMesh::halfedge_from_face_vertex(int face_index, int vertex_index) const
{
    // search face for halfedge pointing to vertex
    int hij = f2h[face_index];
    do
    {
        if (to[hij] == vertex_index) return hij;
        hij = n[hij];
    } while (hij != f2h[face_index]);

    // vertex not in face
    return -1;
}

void HalfedgeMesh::relabel_tip(int halfedge_index, int vertex_index)
{
    int hij = halfedge_index;
    m_is_vertex_deleted[to[hij]] = true;

    int hkj = hij;
    do
    {
        to[hkj] = vertex_index;
        hkj = opp[n[hkj]];
    } while (hkj != hij);
    out[vertex_index] = opp[halfedge_index];

    m_is_vertex_deleted[vertex_index] = false;
}

void HalfedgeMesh::relabel_face(int halfedge_index, int face_index)
{
    int hij = halfedge_index;
    m_is_face_deleted[face[hij]] = true;

    int hjk = hij;
    do
    {
        face[hjk] = face_index;
        hjk = n[hjk];
    } while (hjk != hij);
    f2h[face_index] = halfedge_index;

    m_is_face_deleted[face_index] = false;
}

int HalfedgeMesh::create_halfedge()
{
    int index = opp.size();
    opp.push_back(-1);
    n.push_back(-1);
    to.push_back(-1);
    face.push_back(-1);
    h2e.push_back(-1);
    prev.push_back(-1);
    m_is_halfedge_deleted.push_back(false);
    m_halfedge_count++;
    return index;
}

int HalfedgeMesh::create_face()
{
    int index = f2h.size();
    f2h.push_back(-1);
    m_is_face_deleted.push_back(false);
    m_face_count++;
    return index;
}

int HalfedgeMesh::create_edge()
{
    int index = e2h.size();
    e2h.push_back(-1);
    m_is_edge_deleted.push_back(false);
    m_edge_count++;
    return index;
}

int HalfedgeMesh::create_vertex()
{
    int index = out.size();
    out.push_back(-1);
    m_is_vertex_deleted.push_back(false);
    m_vertex_count++;
    return index;
}

int HalfedgeMesh::make_edge()
{
    // create sphere halfedge topology
    int h1 = create_halfedge();
    int h2 = create_halfedge();
    opp[h1] = h2;
    opp[h2] = h1;
    n[h1] = h2;
    n[h2] = h1;
    prev[h1] = h2;
    prev[h2] = h1;

    // add vertex endpoints
    int v1 = create_vertex();
    int v2 = create_vertex();
    to[h1] = v1;
    to[h2] = v2;
    out[v1] = h2;
    out[v2] = h1;

    // add face
    int f = create_face();
    face[h1] = f;
    face[h2] = f;
    f2h[f] = h1;

    // add edge
    int e = create_edge();
    h2e[h1] = e;
    h2e[h2] = e;
    e2h[e] = h1;

    //assert(is_valid());
    return e;
}

void HalfedgeMesh::delete_edge(int edge_index)
{
    int hij = e2h[edge_index];
    int hji = opp[hij];
    delete_halfedge(hij);
    delete_halfedge(hji);
    delete_dangling_edge(edge_index);
}

void HalfedgeMesh::delete_dangling_edge(int edge_index)
{
    m_is_edge_deleted[edge_index] = true;
    m_edge_count--;
}


void HalfedgeMesh::delete_halfedge(int halfedge_index)
{
    m_is_halfedge_deleted[halfedge_index] = true;
    m_halfedge_count--;
}

void HalfedgeMesh::delete_vertex(int vertex_index)
{
    m_is_vertex_deleted[vertex_index] = true;
    m_vertex_count--;
}

void HalfedgeMesh::delete_face(int face_index)
{
    m_is_face_deleted[face_index] = true;
    m_face_count--;
}

void HalfedgeMesh::delete_bubble(int halfedge_index)
{
    int hij = halfedge_index;
    assert(opp[hij] == n[hij]);
    assert(n[n[hij]] == hij);
    assert(n[n[opp[hij]]] == opp[hij]);
    assert(face[hij] == face[opp[hij]]);
    delete_edge(h2e[hij]);
    delete_face(face[hij]);
    delete_vertex(to[hij]);
    delete_vertex(to[opp[hij]]);
}

// preserve tip/face of h if vertex/face deleted
bool HalfedgeMesh::splice_halfedges(int h, int g)
{
    int hn = n[h];
    int gn = n[g];
    n[h] = gn;
    n[g] = hn;
    prev[hn] = g;
    prev[gn] = h;

    // create new face if splice separates
    if (face[h] == face[g])
    {
        int new_f = create_face();
        relabel_face(g, new_f);
        relabel_face(h, face[h]);
    }
    // otherwise relabel shared face and delete removed face
    else
    {
        delete_face(face[g]);
        relabel_face(g, face[h]);
    }

    // create new vertex if splice separates
    if (to[h] == to[g])
    {
        int new_v = create_vertex();
        relabel_tip(g, new_v);
        relabel_tip(h, to[h]);
    }
    // otherwise relabel shared vertex and delete removed vertex
    else
    {
        delete_vertex(to[g]);
        relabel_tip(g, to[h]);
    }

    //assert(is_valid());
    return true;
}


// preserve tip/face of h if vertex/face deleted
bool HalfedgeMesh::splice_halfedges_conservative(int h, int g)
{
    int hn = n[h];
    int gn = n[g];
    n[h] = gn;
    n[g] = hn;
    prev[hn] = g;
    prev[gn] = h;


    // create new face if splice separates
    if (face[h] == face[g])
    {
        delete_face(face[h]);
        int new_fh = create_face();
        int new_fg = create_face();
        relabel_face(h, new_fh);
        relabel_face(g, new_fg);
    }
    // otherwise relabel shared face and delete removed face
    else
    {
        delete_face(face[h]);
        delete_face(face[g]);
        int new_f = create_face();
        relabel_face(h, new_f);
    }


    // create new vertex if splice separates
    if (to[h] == to[g])
    {
        delete_vertex(to[h]);
        int new_vh = create_vertex();
        int new_vg = create_vertex();
        relabel_tip(h, new_vh);
        relabel_tip(g, new_vg);
    }
    // otherwise relabel shared vertex and delete removed vertex
    else
    {
        delete_vertex(to[h]);
        delete_vertex(to[g]);
        int new_v = create_vertex();
        relabel_tip(h, new_v);
    }

    //assert(is_valid());
    return true;
}

bool HalfedgeMesh::is_join_halfedge_endpoints_valid(int halfedge_index) const
{
    int hij = halfedge_index;
    int hji = opp[hij];
    int hki = prev[hij];
    int hlj = prev[hji];
    if (hki == hij) return false;
    if (hki == hji) return false;
    if (hlj == hij) return false;
    if (hlj == hji) return false;

    return true;
}

bool HalfedgeMesh::join_halfedge_endpoints(int halfedge_index)
{
    assert(is_join_halfedge_endpoints_valid(halfedge_index));

    int hij = halfedge_index;
    int hji = opp[hij];
    int hi_prev = prev[hij];
    int hj_prev = prev[hji];
    int hi_n = n[hij];
    int hj_n = n[hji];
    int fi = face[hij];
    int fj = face[hji];
    int vi = to[hji];
    int vj = to[hij];
    
    n[hi_prev] = hi_n;
    prev[hi_n] = hi_prev;
    n[hj_prev] = hj_n;
    prev[hj_n] = hj_prev;
    relabel_tip(hi_prev, vi);
    f2h[fi] = hi_n;
    f2h[fj] = hj_n;
    delete_edge(h2e[hij]);
    delete_vertex(vj);

    spdlog::debug("joining edge {}, {}", vi, vj);

    return true;
}

bool HalfedgeMesh::join_halfedge_endpoints_conservative(int halfedge_index)
{
    assert(is_join_halfedge_endpoints_valid(halfedge_index));
    
    int hij = halfedge_index;
    int hji = opp[hij];
    int hki = prev[hij];
    int hlj = prev[hji];
    int vi = to[hji];
    int vj = to[hij];
    int fijk = face[hij];
    int fjil = face[hji];
    spdlog::debug("joining edge {}, {}", vi, vj);

    splice_halfedges(hlj, hij); // hij removed, so splice second
    splice_halfedges(hki, hji); // hji removed, so splice second
    splice_halfedges(hlj, hki); // want to preserve vj, so splice hlj first

    // delete removed edge
    delete_bubble(hij);

    relabel_tip(hki, vi);
    relabel_face(hki, fijk);
    relabel_face(hlj, fjil);

    return true;
}

int HalfedgeMesh::split_vertex(int h, int g)
{
    assert(h != g);
    assert(to[h] == to[g]);
    int v = to[h];
    spdlog::debug("splitting vertex {}", v);

    int e = make_edge();
    int h1 = e2h[e];
    int h2 = opp[h1];

    splice_halfedges(h, g);
    splice_halfedges(h1, g);
    splice_halfedges(h2, h);

    spdlog::debug("vertex split");

    return h2;
}

int HalfedgeMesh::split_edge(int halfedge_index)
{
    int hij = halfedge_index;
    int hji = opp[hij];
    int hij_n = n[hij];
    int hji_prev = prev[hji];
    int vj = to[hij];
    int f = face[hij];
    int fo = face[hji];
    spdlog::debug("splitting edge between {} and {}", f, fo);

    int h1 = create_halfedge();
    int h2 = create_halfedge();
    int e = create_edge();
    int vn = create_vertex();
    opp[h1] = h2;
    opp[h2] = h1;
    h2e[h1] = e;
    h2e[h2] = e;
    e2h[e] = h1;
    n[hij] = h1;
    n[h1] = hij_n;
    n[hji_prev] = h2;
    n[h2] = hji;
    prev[h1] = hij;
    prev[hij_n] = h1;
    prev[h2] = hji_prev;
    prev[hji] = h2;
    face[h1] = f;
    face[h2] = fo;

    to[h1] = vj;
    to[h2] = vn;
    to[hij] = vn;
    out[vj] = h2;
    out[vn] = h1;

    return hij;
}

int HalfedgeMesh::split_edge_conservative(int halfedge_index)
{
    int hij = halfedge_index;
    int vj = to[hij];
    int f = face[hij];
    int fo = face[opp[hij]];
    spdlog::debug("splitting edge between {} and {}", f, fo);
    int hjk = split_vertex(hij, prev[opp[hij]]);

    // ensure tip of the halfedge remains unchanged
    relabel_tip(opp[hjk], vj);

    // ensure face of the halfedge remains unchanged
    relabel_face(hij, f);
    relabel_face(opp[hij], fo);

    return hij;
}

int HalfedgeMesh::split_triangle_mesh_edge(int halfedge_index)
{
    int hij = halfedge_index;
    int hji = opp[hij];
    int hjk = n[hij];
    int hil = n[hji];
    split_edge(hij);
    //split_vertex(hij, prev[hji]);
    split_face(hjk, hij);
    split_face(hil, prev[hji]);

    return hij;
}

bool HalfedgeMesh::is_join_edge_faces_valid(int halfedge_index) const
{
    int hij = halfedge_index;
    int hji = opp[hij];
    if (face[hij] == face[hji]) return false;
    if (n[hij] == hij) return false;
    if (n[hji] == hji) return false;

    return true;
}

int HalfedgeMesh::join_edge_faces(int halfedge_index)
{
    // ensure not in degree 1 face
    // TODO Check for bubble
    //if (n[halfedge_index] == halfedge_index) return join_edge_faces(opp[halfedge_index]);
    assert(is_join_edge_faces_valid(halfedge_index));

    int hij = halfedge_index;
    int hji = opp[hij];
    int hi_prev = prev[hij];
    int hj_prev = prev[hji];
    int hi_n = n[hji];
    int hj_n = n[hij];
    int fi = face[hij];
    int fj = face[hji];
    int vi = to[hji];
    int vj = to[hij];
    
    n[hi_prev] = hi_n;
    prev[hi_n] = hi_prev;
    n[hj_prev] = hj_n;
    prev[hj_n] = hj_prev;
    relabel_face(hi_prev, fi);
    out[vi] = hi_n;
    out[vj] = hj_n;
    delete_edge(h2e[hij]);
    delete_face(fj);

    return fi;
}

int HalfedgeMesh::join_edge_faces_conservative(int halfedge_index)
{
    // ensure not in degree 1 face
    // TODO Check for bubble
    //if (n[halfedge_index] == halfedge_index) return join_edge_faces(opp[halfedge_index]);
    assert(is_join_edge_faces_valid(halfedge_index));

    int hij = halfedge_index;
    int hji = opp[hij];
    int hki = prev[hij];
    int hlj = prev[hji];
    int vi = to[hki];
    int vj = to[hlj];
    
    splice_halfedges(hlj, hij); // hij removed, so splice second
    splice_halfedges(hki, hji); // hji removed, so splice second

    // delete removed edge
    delete_bubble(hij);

    relabel_tip(hki, vi);
    relabel_tip(hlj, vj);

    return face[hki];
}

int HalfedgeMesh::split_face(int h, int g)
{
    assert(h != g);
    assert(face[h] == face[g]);
    int f = face[h];
    int hn = n[h];
    int gn = n[g];
    int v = to[h];
    int w = to[g];
    spdlog::debug("splitting face {}", f);

    int h1 = create_halfedge();
    int h2 = create_halfedge();
    int e = create_edge();
    int fn = create_face();
    opp[h1] = h2;
    opp[h2] = h1;
    n[h] = h1;
    n[h1] = gn;
    n[g] = h2;
    n[h2] = hn;
    prev[h1] = h;
    prev[gn] = h1;
    prev[h2] = g;
    prev[hn] = h2;
    face[h1] = f;
    face[h2] = fn;
    to[h1] = w;
    to[h2] = v;
    h2e[h1] = e;
    h2e[h2] = e;
    e2h[e] = h1;

    f2h[f] = h1;
    relabel_face(h2, fn);
    spdlog::debug("face split");

    //assert(is_valid());
    return h1;
}

int HalfedgeMesh::split_face_conservative(int h, int g)
{
    assert(h != g);
    assert(face[h] == face[g]);
    int f = face[h];
    int v = to[h];
    int w = to[g];
    spdlog::debug("splitting face {}", f);

    int e = make_edge();
    int h1 = e2h[e];
    int h2 = opp[h1];

    splice_halfedges(g, h1);
    splice_halfedges(h, h2);

    spdlog::debug("face split");

    relabel_tip(h, v);
    relabel_tip(g, w);

    //assert(is_valid());
    return h1;
}

int HalfedgeMesh::remove_degree_one_vertex(int vertex_index)
{
    int vj = vertex_index;
    int hji = out[vj];
    int vi = to[hji];
    int hij = opp[hji];
    int f = face[hij];
    int hki = prev[hij];
    int hil = n[hji];
    n[hki] = hil;
    prev[hil] = hki;

    // delete removed edge
    delete_edge(h2e[hij]);
    delete_vertex(vj);

    f2h[f] = hki;
    out[vi] = hil;

    return vi;
}

int HalfedgeMesh::remove_degree_one_vertex_conservative(int vertex_index)
{
    int vj = vertex_index;
    int hji = out[vj];
    int vi = to[hji];
    int hij = opp[hji];
    int f = face[hij];
    int hki = prev[hij];
    splice_halfedges(hki, hji);

    // delete removed edge
    delete_bubble(hij);

    relabel_tip(hki, vi);
    relabel_face(hki, f);

    return vi;
}

int HalfedgeMesh::add_center_to_face(int face_index)
{
    int hij = f2h[face_index];
    int vj = to[hij];
    int num_edges = face_degree(face_index);

    // make first edge
    int e = make_edge();
    int h1 = e2h[e];
    int h2 = opp[h1];

    // splice edge into face
    splice_halfedges(hij, h1);
    assert(n[hij] == h2);

    // ensure tip of the halfedge remains unchanged
    relabel_tip(hij, vj);

    // connect vertex to other face corners
    int hjk = n[opp[n[hij]]];
    for (int i = 1; i < num_edges; ++i)
    {
        split_face(hjk, h2);
        hjk = n[opp[n[hjk]]];
    }

    return to[h2];
}

bool HalfedgeMesh::is_collapse_edge_valid(int halfedge_index) const
{
    // cannot collapse degree 1 vertices
    int hij = halfedge_index;
    int hji = opp[hij];
    if (n[hij] == hji) return false;
    if (n[hji] == hij) return false;

    // conditions for joining faces to work
    int hjk = n[hij];
    int hlj = prev[hji];
    if (face[hjk] == face[opp[hjk]]) return false;
    if (face[hlj] == face[opp[hlj]]) return false;

    // not in halfedge formulation: needed for the two face joins to be distinct
    //if (opp[hjk] == hlj) return false;

    return true;
}


int HalfedgeMesh::collapse_edge(int halfedge_index)
{
    int hij = halfedge_index;
    int hji = opp[hij];
    assert(!is_halfedge_deleted(halfedge_index));
    assert(face_degree(face[hij]) == 3);
    assert(face_degree(face[hji]) == 3);

    // halfedges to remove
    int hjk = n[hij];
    int hki = n[hjk];
    int hlj = prev[hji];
    int hil = prev[hlj];

    // faces to remove
    int fijk = face[hij];
    int fjil = face[hji];

    // two edges and vertex to remove
    int ejk = h2e[hjk];
    int elj = h2e[hlj];
    int vj = to[hij];

    // halfedges to join
    int hkj = opp[hjk];
    int hik = opp[hki];
    int hjl = opp[hlj];
    int hli = opp[hil];

    // two edges and vertex to keep
    int eik = h2e[hik]; 
    int eli = h2e[hli];
    int vi = to[hji];
    int vk = to[hjk];
    int vl = to[hil];

    // h2e, opp: link e_ik = (h_ik, h_kj); e_li = ( h_li,h_jl)
    h2e[hkj] = eik;
    h2e[hjl] = eli;
    opp[hik] = hkj; opp[hkj] = hik;
    opp[hli] = hjl; opp[hjl] = hli;

    //e2h: if a halfedge to be deleted is the halfedge assigned to an edge
    // reassign the one that we keep
    if (e2h[eik] == hki) e2h[eik] = hik;
    if (e2h[eli] == hil) e2h[eli] = hli;

    // delete edges and halfedges
    delete_edge(h2e[hij]);
    for (int h : {hjk, hki, hlj, hil})
    {
        delete_halfedge(h);
    }

    // only delete edge label, not associated halfedges
    delete_dangling_edge(ejk);
    delete_dangling_edge(elj);

    // delete faces and vertices
    delete_vertex(vj);
    delete_face(fijk);
    delete_face(fjil);

    // relabel collapsed vertex
    relabel_tip(hli, vi);
    out[vk] = hkj;
    out[vl] = hli;

    assert(face_degree(face[hkj]) == 3);
    assert(face_degree(face[hik]) == 3);
    assert(face_degree(face[hjl]) == 3);
    assert(face_degree(face[hli]) == 3);

    return hli;
}

int HalfedgeMesh::collapse_edge_conservative(int halfedge_index)
{
    int hij = halfedge_index;
    int hji = opp[hij];
    assert(face_degree(face[hij]) == 3);
    assert(face_degree(face[hji]) == 3);
    int hjk = n[hij];
    int hlj = prev[hji];
    int hki = n[hjk];
    int hil = prev[hlj];
    int vi = to[hji];
    join_halfedge_endpoints(hij);
    join_edge_faces(hjk);
    if (opp[hlj] != hjk)
    {
        join_edge_faces(hlj);
        assert(face_degree(face[hil]) == 3);
        assert(face_degree(face[opp[hil]]) == 3);
    }
    else
    {
        join_edge_faces(hil);
    }

    assert(face_degree(face[hki]) == 3);
    assert(face_degree(face[opp[hki]]) == 3);
    return hki;
}


// get link without vertices of triangles to be collapsed
std::set<int> HalfedgeMesh::get_collapse_link(int halfedge_index) const
{
    std::set<int> link;
    int hij = halfedge_index;
    //hij = n[opp[n[opp[hij]]]]; // iterate twice to skip triangle vertex
    //int h_end = opp[prev[hij]];
    int h_end = halfedge_index;
    do
    {
        link.insert(to[hij]);
        hij = n[opp[hij]];
    } while (hij != h_end);

    return link;
}

bool HalfedgeMesh::is_collapse_link_valid(int halfedge_index) const
{
    // cannot collapse degree 1 vertices
    int hij = halfedge_index;
    int hji = opp[hij];
    int hjk = n[hij];
    int hil = n[hji];
    int vk = to[hjk];
    int vl = to[hil];
    std::set<int> vi_link = get_collapse_link(hij);
    std::set<int> vj_link = get_collapse_link(hji);
    std::vector<int> intersection;
    std::set_intersection(
        vi_link.begin(), vi_link.end(),
        vj_link.begin(), vj_link.end(),
        std::back_inserter(intersection)
    );

    if (intersection.size() != 2) return false;
    if ((intersection[0] != vk) && (intersection[1] != vk)) return false;
    if ((intersection[0] != vl) && (intersection[1] != vl)) return false;
    return true;
}

int HalfedgeMesh::flip_edge(int halfedge_index)
{
    // local halfedges
    int hij = halfedge_index;
    int hji = opp[hij];
    int hjk = n[hij];
    int hki = n[hjk];
    int hil = n[hji];
    int hlj = n[hil];

    // local vertices
    int vi = to[hji];
    int vj = to[hij];
    int vk = to[hjk];
    int vl = to[hil];

    // local faces
    int fijk = face[hij];
    int fjil = face[hji];

    // update connectivity
    int hlk = hij; // rename for sanity
    int hkl = hji; // rename for sanity
    set_next(hlk, hki);
    set_next(hki, hil);
    set_next(hil, hlk);
    set_next(hkl, hlj);
    set_next(hlj, hjk);
    set_next(hjk, hkl);

    // face connectivity
    relabel_face(hlk, fijk);
    relabel_face(hkl, fjil);

    // vertex connectivity
    to[hlk] = vk;
    to[hkl] = vl;
    out[vi] = hil;
    out[vj] = hjk;
    
    return hlk;
}

bool HalfedgeMesh::is_vertex_degree_one(int vertex_index) const
{
    int hji = out[vertex_index];
    return (n[opp[hji]] == hji);
}

bool HalfedgeMesh::is_vertex_degree_two(int vertex_index) const
{
    int hji = out[vertex_index];
    int hjk = n[opp[hji]];
    if (hji == hjk) return false;
    return (n[opp[hjk]] == hji);
}

void HalfedgeMesh::triangulate()
{
    int num_faces = f2h.size();
    for (int f = 0; f < num_faces; ++f)
    {
        if (m_is_face_deleted[f]) continue;
        bool success = triangulate_face(f);
    }
}

bool HalfedgeMesh::triangulate_face(int face_index)
{
    int hij = f2h[face_index];
    int degree = face_degree(face_index);
    if (degree < 3)
    {
        spdlog::warn("cannot triangulate degree {} face", degree);
        return false;
    }

    for (int i = 3; i < degree; ++i)
    {
        // TODO make this a proper nondegenerate triangulation
        spdlog::debug("{} split of degree {} face", i - 2, degree);
        split_face(hij, n[n[hij]]);
    }

    return true;
}

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> HalfedgeMesh::generate_halfedge_faces() const
{
    auto [opp_r, n_r, prev_r] = reduce_halfedge_maps();
    auto [face_r, f2h_r] = reduce_face_maps();
    auto [to_r, out_r] = reduce_vertex_maps();
    return generate_connectivity_faces(n_r, f2h_r, to_r);
}

void HalfedgeMesh::reindex_vertices(const std::vector<int>& vtx_reindex)
{
    std::vector<int> vtx_reindex_inv = invert_map(vtx_reindex);
    to = vector_compose(vtx_reindex, to);
    out = vector_compose(out, vtx_reindex_inv);
}

Eigen::SparseMatrix<int> HalfedgeMesh::generate_VV_to_halfedge_map() const
{
    auto [he_new2old, he_old2new] = get_halfedge_reductions();
    auto [v_new2old, v_old2new] = get_vertex_reductions();

    int num_vertices = v_new2old.size();
    Eigen::SparseMatrix<int> vv2he(num_vertices, num_vertices);

    // build sparse matrix
    typedef Eigen::Triplet<int> Trip;
    std::vector<Trip> trips;
    int num_halfedges = he_new2old.size();
    trips.reserve(num_halfedges);
    for (int i = 0; i < num_halfedges; ++i)
    {
        int hij = he_new2old[i];
        int vj = v_old2new[to[hij]];
        int vi = v_old2new[to[opp[hij]]];
        trips.push_back(Trip(vi, vj, hij + 1));
    }
    vv2he.setFromTriplets(trips.begin(), trips.end());

    return vv2he;
}


std::tuple<std::vector<int>, std::vector<int>> HalfedgeMesh::get_halfedge_reductions() const
{
    std::vector<int> he_removed, he_new2old, he_old2new;
    enumerate_boolean_array(m_is_halfedge_deleted, he_removed, he_new2old, he_old2new);
    return std::make_tuple(he_new2old, he_old2new);
}

std::tuple<std::vector<int>, std::vector<int>> HalfedgeMesh::get_face_reductions() const
{
    std::vector<int> f_removed, f_new2old, f_old2new;
    enumerate_boolean_array(m_is_face_deleted, f_removed, f_new2old, f_old2new);
    return std::make_tuple(f_new2old, f_old2new);
}

std::tuple<std::vector<int>, std::vector<int>> HalfedgeMesh::get_vertex_reductions() const
{
    std::vector<int> v_removed, v_new2old, v_old2new;
    enumerate_boolean_array(m_is_vertex_deleted, v_removed, v_new2old, v_old2new);
    return std::make_tuple(v_new2old, v_old2new);
}

std::tuple<std::vector<int>, std::vector<int>> HalfedgeMesh::get_edge_reductions() const
{
    std::vector<int> e_removed, e_new2old, e_old2new;
    enumerate_boolean_array(m_is_edge_deleted, e_removed, e_new2old, e_old2new);
    return std::make_tuple(e_new2old, e_old2new);
}


std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> HalfedgeMesh::reduce_halfedge_maps() const
{
    auto [he_new2old, he_old2new] = get_halfedge_reductions();

    std::vector<int> opp_new = vector_compose(he_old2new, vector_compose(opp, he_new2old));
    std::vector<int> n_new = vector_compose(he_old2new, vector_compose(n, he_new2old));
    std::vector<int> prev_new = vector_compose(he_old2new, vector_compose(prev, he_new2old));

    return std::make_tuple(opp_new, n_new, prev_new);
}

std::tuple<std::vector<int>, std::vector<int>> HalfedgeMesh::reduce_face_maps() const
{
    auto [he_new2old, he_old2new] = get_halfedge_reductions();
    auto [f_new2old, f_old2new] = get_face_reductions();

    std::vector<int> face_new = vector_compose(f_old2new, vector_compose(face, he_new2old));
    std::vector<int> f2h_new = vector_compose(he_old2new, vector_compose(f2h, f_new2old));

    return std::make_tuple(face_new, f2h_new);
}

std::tuple<std::vector<int>, std::vector<int>> HalfedgeMesh::reduce_edge_maps() const
{
    auto [he_new2old, he_old2new] = get_halfedge_reductions();
    auto [e_new2old, e_old2new] = get_edge_reductions();

    std::vector<int> h2e_new = vector_compose(e_old2new, vector_compose(h2e, he_new2old));
    std::vector<int> e2h_new = vector_compose(he_old2new, vector_compose(e2h, e_new2old));

    return std::make_tuple(h2e_new, e2h_new);
}

std::tuple<std::vector<int>, std::vector<int>> HalfedgeMesh::reduce_vertex_maps() const
{
    auto [he_new2old, he_old2new] = get_halfedge_reductions();
    auto [v_new2old, v_old2new] = get_vertex_reductions();

    std::vector<int> to_new = vector_compose(v_old2new, vector_compose(to, he_new2old));
    std::vector<int> out_new = vector_compose(he_old2new, vector_compose(out, v_new2old));

    return std::make_tuple(to_new, out_new);
}

bool HalfedgeMesh::is_simplicial() const
{
    int num_halfedges = count_halfedges();
    for (int hij = 0; hij < num_halfedges; ++hij)
    {
        if (is_halfedge_deleted(hij)) continue;
        if (to[hij] == opp[to[hij]]) return false;
        if (face[hij] == face[opp[hij]]) return false;
    }

    return true;
}

bool HalfedgeMesh::is_valid() const
{
    int num_halfedges = count_halfedges();
    auto [opp_r, n_r, prev_r] = reduce_halfedge_maps();
    if (opp_r.size() != num_halfedges) return errmess("opp wrong size");
    if (n_r.size() != num_halfedges)  return errmess("next wrong size");
    if (prev_r.size() != num_halfedges)  return errmess("prev wrong size");

    int num_faces = count_faces();
    auto [face_r, f2h_r] = reduce_face_maps();
    if (face_r.size() != num_halfedges) return errmess("face wrong size");
    if (f2h_r.size() != num_faces) return errmess("f2h wrong size");

    int num_edges = count_edges();
    auto [h2e_r, e2h_r] = reduce_edge_maps();
    if (h2e_r.size() != num_halfedges) return errmess("h2e wrong size");
    if (e2h_r.size() != num_edges) return errmess("e2h wrong size");

    int num_vertices = count_vertices();
    auto [to_r, out_r] = reduce_vertex_maps();
    if (to_r.size() != num_halfedges) return errmess("to wrong size");
    if (out_r.size() != num_vertices) return errmess("out wrong size");

    if (!are_polygon_mesh_halfedges_valid(n_r, prev_r)) return errmess("mesh halfedges not valid");
    if (!are_polygon_mesh_edges_valid(opp_r, h2e_r, e2h_r)) return errmess("mesh edges not valid");
    if (!are_polygon_mesh_vertices_valid(opp_r, prev_r, to_r, out_r)) return errmess("mesh vertices not valid");
    if (!are_polygon_mesh_faces_valid(n_r, face_r, f2h_r)) return errmess("mesh faces not valid");

    return true;
}

void HalfedgeMesh::collect_garbage()
{
    if (skip_garbage_collection) return;

    auto [he_new2old, he_old2new] = get_halfedge_reductions();
    auto [e_new2old, e_old2new] = get_edge_reductions();
    auto [v_new2old, v_old2new] = get_vertex_reductions();
    auto [f_new2old, f_old2new] = get_face_reductions();
    collect_garbage(
        he_new2old,
        e_new2old,
        v_new2old,
        f_new2old,
        he_old2new,
        e_old2new,
        v_old2new,
        f_old2new);
}

void HalfedgeMesh::collect_garbage(
    const std::vector<int>& he_new2old,
    const std::vector<int>& e_new2old,
    const std::vector<int>& v_new2old,
    const std::vector<int>& f_new2old,
    const std::vector<int>& he_old2new,
    const std::vector<int>& e_old2new,
    const std::vector<int>& v_old2new,
    const std::vector<int>& f_old2new
)
{
    if (skip_garbage_collection) return;
    m_halfedge_count = he_new2old.size();
    m_vertex_count = v_new2old.size();
    m_edge_count = e_new2old.size();
    m_face_count = f_new2old.size();
    opp.resize(m_halfedge_count);
    n.resize(m_halfedge_count);
    prev.resize(m_halfedge_count);
    h2e.resize(m_halfedge_count);
    face.resize(m_halfedge_count);
    to.resize(m_halfedge_count);
    for (int hij = 0; hij < m_halfedge_count; ++hij)
    {
        opp[hij] = he_old2new[opp[he_new2old[hij]]];
        n[hij] = he_old2new[n[he_new2old[hij]]];
        prev[hij] = he_old2new[prev[he_new2old[hij]]];
        h2e[hij] = e_old2new[h2e[he_new2old[hij]]];
        face[hij] = f_old2new[face[he_new2old[hij]]];
        to[hij] = v_old2new[to[he_new2old[hij]]];
    }

    e2h.resize(m_edge_count);
    for (int eij = 0; eij < m_edge_count; ++eij)
    {
        e2h[eij] = he_old2new[e2h[e_new2old[eij]]];
    }

    out.resize(m_vertex_count);
    for (int vi = 0; vi < m_vertex_count; ++vi)
    {
        out[vi] = he_old2new[out[v_new2old[vi]]];
    }

    f2h.resize(m_face_count);
    for (int f = 0; f < m_face_count; ++f)
    {
        f2h[f] = he_old2new[f2h[f_new2old[f]]];
    }
    
    // set all elements to undeleted
    m_is_halfedge_deleted = std::vector<bool>(n.size(), false);
    m_is_vertex_deleted = std::vector<bool>(out.size(), false);
    m_is_edge_deleted = std::vector<bool>(e2h.size(), false);
    m_is_face_deleted = std::vector<bool>(f2h.size(), false);
}

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> generate_connectivity_faces(
    const std::vector<int>& next,
    const std::vector<int>& f2h,
    const std::vector<int>& to)
{
    int num_faces = f2h.size();
    Eigen::MatrixXi F(num_faces, 3);
    Eigen::MatrixXi F_halfedge(num_faces, 3);

    for (int fijk = 0; fijk < num_faces; ++fijk) {
        // Get halfedges of face
        int hij = f2h[fijk];
        int hjk = next[hij];
        int hki = next[hjk];

        // Get vertices of face
        int vj = to[hij];
        int vk = to[hjk];
        int vi = to[hki];

        // Write face with halfedge and opposite vertex data
        F_halfedge(fijk, 0) = hij;
        F_halfedge(fijk, 1) = hjk;
        F_halfedge(fijk, 2) = hki;
        F(fijk, 0) = vj;
        F(fijk, 1) = vk;
        F(fijk, 2) = vi;
    }

    return std::make_tuple(F, F_halfedge);
}

std::tuple<HalfedgeMesh, std::vector<int>, std::vector<std::pair<int, int>>> generate_halfedge(const Eigen::MatrixXi& F)
{
    std::vector<int> next_he;
    std::vector<int> opp;
    std::vector<int> bnd_loops;
    std::vector<std::vector<int>> corner_to_he;
    std::vector<std::pair<int, int>> he_to_corner;
    std::vector<int> vtx_reindex;
    Connectivity C;
    FV_to_NOB(F, next_he, opp, bnd_loops, vtx_reindex, corner_to_he, he_to_corner);
    NOB_to_connectivity(next_he, opp, bnd_loops, C);

    std::vector<int> h2e, e2h;
    build_edge_maps(C.opp, h2e, e2h);

    HalfedgeMesh halfedge(C.opp, C.n, C.to, C.f, h2e);

    return std::make_tuple(halfedge, vtx_reindex, he_to_corner);
}

} // namespace Resolution 
} // namespace Penner
