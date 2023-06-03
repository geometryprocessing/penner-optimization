#include "transitions.hh"

#include "embedding.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {

Scalar
log_length_regular(Scalar lle, Scalar lla, Scalar llb, Scalar llc, Scalar lld)
{
  Scalar a = (lla + llc - lle) / 2.0;
  Scalar b = (llb + lld - lle) / 2.0;
  Scalar c = (a + b) / 2.0;
  return 2.0 * (c + log(exp(a - c) + exp(b - c)));
}

Scalar
Delaunay_ind_T(Scalar ld, Scalar la, Scalar lb)
{
  return (la / lb + lb / la - (ld / la) * (ld / lb));
}

Scalar
metric_halfedge_Delaunay_ind(const Mesh<Scalar>& m,
                             const VectorX& he_metric_coords,
                             int h)
{
  // FIXME Boundary loops
  int hd = h;
  int hb = m.n[h];
  int ha = m.n[hb];

  // Compute the edge lengths of the triangle for h, scaled so that h has length
  // 1
  Scalar scale = he_metric_coords[hd] / 2.0;
  Scalar ld = exp(he_metric_coords[hd] / 2.0 - scale);
  Scalar la = exp(he_metric_coords[ha] / 2.0 - scale);
  Scalar lb = exp(he_metric_coords[hb] / 2.0 - scale);

  return Delaunay_ind_T(ld, la, lb);
}

Scalar
metric_edge_Delaunay_ind(const Mesh<Scalar>& m,
                         const VectorX& he_metric_coords,
                         int h)
{
  Scalar ind1 = metric_halfedge_Delaunay_ind(m, he_metric_coords, h);
  Scalar ind2 = metric_halfedge_Delaunay_ind(m, he_metric_coords, m.opp[h]);
  return ind1 + ind2;
}

void
compute_metric_Delaunay_inds(const Mesh<Scalar>& m,
                             const VectorX& he_metric_coords,
                             VectorX& inds)
{
  inds.resize(he_metric_coords.size());
  for (int h = 0; h < he_metric_coords.size(); ++h) {
    inds[h] = metric_edge_Delaunay_ind(m, he_metric_coords, h);
  }
}

bool
flip_metric_ccw(Mesh<Scalar>& m, VectorX& he_metric_coords, int h)
{
  Mesh<Scalar>& mc = m.cmesh();

  // Get halfedges of the edge
  int ha = h;
  int hb = mc.opp[h];

  // Get faces adjacent to the edge and return if there is only one adjacent
  // face
  int f0 = mc.f[ha];
  int f1 = mc.f[hb];
  if (f0 == f1)
    return false;

  // Get the halfedges on the boundary of the edge flap around e
  int h2 = mc.n[ha];
  int h3 = mc.n[h2];
  int h4 = mc.n[hb];
  int h5 = mc.n[h4];

  // Compute the log length of the edge after the flip
  he_metric_coords[ha] = he_metric_coords[hb] =
    log_length_regular(he_metric_coords[ha],
                       he_metric_coords[h2],
                       he_metric_coords[h3],
                       he_metric_coords[h4],
                       he_metric_coords[h5]);

  return m.flip_ccw(h);
}

void
update_jacobian_del(const Mesh<Scalar>& m,
                    const VectorX& he_metric_coords,
                    int h,
                    const std::vector<int>& he2e,
                    std::vector<std::map<int, Scalar>>& J_del_lol)
{
  // Get local mesh information near hd
  int hd = h;
  int hb = m.n[hd];
  int ha = m.n[hb];
  int hdo = m.opp[hd];
  int hao = m.n[hdo];
  int hbo = m.n[hao];

  // Get edges corresponding to halfedges
  int ed = he2e[hd];
  int eb = he2e[hb];
  int ea = he2e[ha];
  int eao = he2e[hao];
  int ebo = he2e[hbo];

  // Compute the shear for the edge ed
  Scalar lla = he_metric_coords[ha];
  Scalar llb = he_metric_coords[hb];
  Scalar llao = he_metric_coords[hao];
  Scalar llbo = he_metric_coords[hbo];
  Scalar x = exp((lla + llbo - llb - llao) / 2.0);

  // The matrix Pd corresponding to flipping edge ed is the identity except for
  // the row corresponding to edge ed, which has entries defined by Pd_scalars
  // in the column corresponding to the edge with the same index in Pd_edges
  std::vector<int> Pd_edges = { ed, ea, ebo, eao, eb };
  std::vector<Scalar> Pd_scalars = {
    -1.0, x / (1.0 + x), x / (1.0 + x), 1.0 / (1.0 + x), 1.0 / (1.0 + x)
  };

  // Compute the new row of J_del corresponding to edge ed, which is the only
  // edge that changes
  std::map<int, Scalar> J_del_d_new;
  for (int i = 0; i < 5; ++i) {
    int ei = Pd_edges[i];
    Scalar Di = Pd_scalars[i];
    for (auto it : J_del_lol[ei]) {
      J_del_d_new[it.first] += Di * it.second;

      // Delete the updated entry if it is near 0
      if (abs(J_del_d_new[it.first]) < 1e-15)
        J_del_d_new.erase(it.first);
    }
  }

  J_del_lol[ed] = J_del_d_new;
}

bool
edge_flip(Mesh<Scalar>& m,
          VectorX& he_metric_coords,
          int h,
          int tag,
          std::set<int>& q,
          std::vector<int>& flip_seq,
          const std::vector<int>& he2e,
          std::vector<std::map<int, Scalar>>& J_del_lol,
          bool need_jacobian)
{
  Mesh<Scalar>& mc = m.cmesh();

  // Get halfedges of the edge flap about h
  int hij = mc.h0(h);
  int hjk = mc.n[hij];
  int hki = mc.n[hjk];
  int hji = mc.h1(h);
  int him = mc.n[hji];
  int hmj = mc.n[him];

  std::vector<char>& type = mc.type;

  // Determine edges that need to be flipped to maintain symmetry, and update R
  // and type
  // TODO Move this to a separate function to clearly separate the duplicated
  // code
  std::vector<int> to_flip;
  if (type[hij] > 0) // skip in non-symmetric mode for efficiency
  {
    int types;
    bool reverse = true;
    if (type[hki] <= type[hmj]) {
      types = type[hki] * 100000 + type[hjk] * 10000 + type[hij] * 1000 +
              type[hji] * 100 + type[him] * 10 + type[hmj];
      reverse = false;
    } else
      types = type[hmj] * 100000 + type[him] * 10000 + type[hji] * 1000 +
              type[hij] * 100 + type[hjk] * 10 + type[hki];

    if (types == 231123 || types == 231132 || types == 321123)
      return false; // t1t irrelevant
    if (types == 132213 || types == 132231 || types == 312213)
      return false; // t2t irrelevant
    if (types == 341143)
      return false; // q1q irrelevant
    if (types == 342243)
      return false; // q2q irrelevant

    switch (types) {
      case 111222: // (1|2)
        type[hij] = type[hji] = 3;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 123312: // (t,_,t)
        type[hij] = type[hki];
        type[hji] = type[hmj];
        mc.R[hij] = hji;
        mc.R[hji] = hij;
        break;
      case 111123: // (1,1,t)
        type[hij] = type[hji] = 4;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 111132: // (1,1,t) mirrored
        type[hij] = type[hji] = 4;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 222214: // (2,2,t) following (1,1,t) mirrored
        type[hij] = type[hji] = 3;
        to_flip.push_back(
          6); // to make sure all fake diagonals are top left to bottom right
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 142222: // (2,2,t) following (1,1,t)
        type[hij] = type[hji] = 3;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 213324: // (t,_,q)
        type[hij] = type[hji] = 2;
        to_flip.push_back(6);
        break;
      case 134412: // (t,_,q) 2nd
        type[hij] = type[hji] = 1;
        if (!reverse) {
          mc.R[hji] = hmj;
          mc.R[hmj] = hji;
          mc.R[mc.opp[hji]] = mc.opp[hmj];
          mc.R[mc.opp[hmj]] = mc.opp[hji];
        } else {
          mc.R[hij] = hki;
          mc.R[hki] = hij;
          mc.R[mc.opp[hij]] = mc.opp[hki];
          mc.R[mc.opp[hki]] = mc.opp[hij];
        }
        break;
      case 123314: // (q,_,t)
        type[hij] = type[hji] = 1;
        to_flip.push_back(6);
        break;
      case 124432: // (q,_,t) 2nd
        type[hij] = type[hji] = 2;
        if (!reverse) {
          mc.R[hki] = hij;
          mc.R[hij] = hki;
          mc.R[mc.opp[hki]] = mc.opp[hij];
          mc.R[mc.opp[hij]] = mc.opp[hki];
        } else {
          mc.R[hmj] = hji;
          mc.R[hji] = hmj;
          mc.R[mc.opp[hmj]] = mc.opp[hji];
          mc.R[mc.opp[hji]] = mc.opp[hmj];
        }
        break;
      case 111143: // (1,1,q)
        type[hij] = type[hji] = 4;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 222243: // (2,2,q) following (1,1,q)
        type[hij] = type[hji] = 4;
        to_flip.push_back(5);
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 144442: // (1,1,q)+(2,2,q) 3rd
        type[hij] = type[hji] = 3;
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 413324: // (q,_,q)
        type[hij] = type[hji] = 4;
        to_flip.push_back(6);
        to_flip.push_back(1);
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 423314: // (q,_,q) opp
        type[hij] = type[hji] = 4;
        to_flip.push_back(1);
        to_flip.push_back(6);
        mc.R[hij] = hij;
        mc.R[hji] = hji;
        break;
      case 134414: // (q,_,q) 2nd
        type[hij] = type[hji] = 1;
        break;
      case 234424: // (q,_,q) 3rd
        type[hij] = type[hji] = 2;
        if (!reverse) {
          mc.R[hji] =
            mc.n[mc.n[mc.opp[mc.n[mc.n[hji]]]]]; // attention: hji is not yet
                                                 // flipped here, hence twice
                                                 // .n[]
          mc.R[mc.n[mc.n[mc.opp[mc.n[mc.n[hji]]]]]] = hji;
          mc.R[mc.opp[hji]] = mc.opp[mc.R[hji]];
          mc.R[mc.opp[mc.R[hji]]] = mc.opp[hji];
        } else {
          mc.R[hij] = mc.n[mc.n[mc.opp[mc.n[mc.n[hij]]]]];
          mc.R[mc.n[mc.n[mc.opp[mc.n[mc.n[hij]]]]]] = hij;
          mc.R[mc.opp[hij]] = mc.opp[mc.R[hij]];
          mc.R[mc.opp[mc.R[hij]]] = mc.opp[hij];
        }
        break;
      case 314423: // fake diag switch following (2,2,t) following (1,1,t)
                   // mirrored
        break;
      case 324413: // fake diag switch (opp) following (2,2,t) following (1,1,t)
                   // mirrored
        break;
      case 111111:
        break;
      case 222222:
        break;
      case 000000:
        type[hij] = type[hji] = 0; // for non-symmetric mode
        break;
      default:
        spdlog::error(" (attempted to flip edge that should never be "
                      "non-Delaunay (type{})).",
                      types);
        return false;
    }

    if (reverse) {
      for (size_t i = 0; i < to_flip.size(); i++)
        to_flip[i] = 7 - to_flip[i];
    }
  }

  // Flip edge and update Jacobian if needed
  flip_metric_ccw(m, he_metric_coords, hij);
  flip_seq.push_back(hij);
  if (need_jacobian)
    update_jacobian_del(m, he_metric_coords, hij, he2e, J_del_lol);
  if (tag == 1) {
    flip_metric_ccw(m, he_metric_coords, hij);
    flip_seq.push_back(hij);
    if (need_jacobian)
      update_jacobian_del(m, he_metric_coords, hij, he2e, J_del_lol);

    flip_metric_ccw(m, he_metric_coords, hij);
    flip_seq.push_back(hij);
    if (need_jacobian)
      update_jacobian_del(m, he_metric_coords, hij, he2e, J_del_lol);
  } // to make it cw on side 2

  // Add edges of the boundary of the triangle flap to the set q
  q.insert(mc.h0(hjk));
  q.insert(mc.h0(hki));
  q.insert(mc.h0(him));
  q.insert(mc.h0(hmj));

  // Recursively flip other edges to maintain symmetry
  for (size_t i = 0; i < to_flip.size(); i++) {
    if (to_flip[i] == 1)
      edge_flip(m,
                he_metric_coords,
                mc.e(hki),
                2,
                q,
                flip_seq,
                he2e,
                J_del_lol,
                need_jacobian);
    if (to_flip[i] == 2)
      edge_flip(m,
                he_metric_coords,
                mc.e(hjk),
                2,
                q,
                flip_seq,
                he2e,
                J_del_lol,
                need_jacobian);
    if (to_flip[i] == 5)
      edge_flip(m,
                he_metric_coords,
                mc.e(him),
                2,
                q,
                flip_seq,
                he2e,
                J_del_lol,
                need_jacobian);
    if (to_flip[i] == 6)
      edge_flip(m,
                he_metric_coords,
                mc.e(hmj),
                2,
                q,
                flip_seq,
                he2e,
                J_del_lol,
                need_jacobian);
  }

  return true;
}

void
make_delaunay_with_jacobian_in_place(Mesh<Scalar>& m_del,
                                     const VectorX& metric_coords,
                                     VectorX& metric_coords_del,
                                     MatrixX& J_del,
                                     std::vector<int>& flip_seq,
                                     bool need_jacobian)
{

  Mesh<Scalar>& mc = m_del.cmesh();

  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(mc, he2e, e2he);

  // Get halfedge lambdas array from edge array
  VectorX he_metric_coords_del;
  expand_edge_func(mc, metric_coords, he_metric_coords_del);

  // Initialize the Jacobian to the identity. We use a list of maps
  // representation for very quick row access and updates of this sparse matrix
  std::vector<std::map<int, Scalar>> J_del_lol(metric_coords.size(),
                                               std::map<int, Scalar>());
  if (need_jacobian) {
    for (int e = 0; e < metric_coords.size(); ++e) {
      J_del_lol[e][e] = 1.0;
    }
  }

  // Get unique halfedge representative for a full set of independent edges
  // (i.e. no edge is a reflection of another edge) to initialize local make
  // Delaunay algorithm
  int flips = 0;
  std::set<int> q;
  for (int h = 0; h < mc.n_halfedges(); h++) {
    // Only consider halfedges with lower index to prevent duplication of
    // halfedges
    if (mc.opp[h] < h)
      continue;

    // type 22 edges are flipped below; type 44 edges (virtual diagonals) are
    // never flipped.
    int type0 = mc.type[mc.h0(h)];
    int type1 = mc.type[mc.h1(h)];
    if (type0 == 0 || type0 == 1 || type1 == 1 || type0 == 3)
      q.insert(h);
  }

  // Continue until there are no more potentially non-Delaunay edges
  while (!q.empty()) {
    // Get halfedge representative h and types of the associated halfedges for
    // edge e
    int h = *(q.begin());
    q.erase(q.begin());
    int type0 = mc.type[mc.h0(h)];
    int type1 = mc.type[mc.h1(h)];

    // Get edge flap halfedges for e
    int hd = mc.h0(h);
    int hb = mc.n[hd];
    int ha = mc.n[hb];
    int hdo = mc.opp[hd];

    bool is_del;
    Scalar ind;
    // Self adjacent triangles are always Delaunay
    if ((type0 == 0) && ((hdo == ha) || (hdo == hb))) {
      is_del = true;
    }
    // Other edges are delaunay iff the Delaunay index is positive
    else {
      ind = metric_edge_Delaunay_ind(mc, he_metric_coords_del, hd);
      is_del = (ind > 0);
    }

    // Only flip nondiagonal edges that are not type 2 and not Delaunay
    if (!(type0 == 2 && type1 == 2) && (type0 != 4) && !is_del) {
      // Get a halfedge representative for the reflected edge of e
      int Rhd = -1;
      if (type0 == 1 && type1 == 1)
        Rhd = mc.h0(mc.R[mc.h0(hd)]);

      // Flip the edge, or continue to the next edge if no flip was performed,
      // and add potentially invalidated halfedges to the set q
      if (!edge_flip(m_del,
                     he_metric_coords_del,
                     hd,
                     0,
                     q,
                     flip_seq,
                     he2e,
                     J_del_lol,
                     need_jacobian))
        continue;
      flips++;

      // flip mirror edge on sheet 2 and add invalidated halfedges to q
      if (type0 == 1 && type1 == 1) {
        int hd = Rhd;
        if (!edge_flip(m_del,
                       he_metric_coords_del,
                       hd,
                       1,
                       q,
                       flip_seq,
                       he2e,
                       J_del_lol,
                       need_jacobian))
          continue;
        flips++;
      }
    }
  }

  // FIXME Remove
  std::cout << "Flips: " << flips << std::endl;

  // Create Jacobian from list of maps if needed
  if (need_jacobian) {
    // Get triplet (i, j, v) for each value v = J_del_lol[i][j] in the list of
    // maps
    int num_edges = mc.n.size() / 2;
    std::vector<T> tripletList;
    tripletList.reserve(5 * num_edges);
    for (int i = 0; i < num_edges; ++i) {
      for (auto it : J_del_lol[i]) {
        tripletList.push_back(T(i, it.first, it.second));
      }
    }

    // Create the matrix from the triplets
    J_del.resize(metric_coords.size(), metric_coords.size());
    J_del.reserve(tripletList.size());
    J_del.setFromTriplets(tripletList.begin(), tripletList.end());
  }

  // Get edge based Delaunay log edge lengths from halfedge array
  restrict_he_func(mc, he_metric_coords_del, metric_coords_del);
}

void
update_jacobian_del(const Mesh<Scalar>& m,
                    int h,
                    const std::vector<int>& he2e,
                    std::vector<std::map<int, Scalar>>& J_del_lol)
{
  // Get local mesh information near hd
  int hd = h;
  int hb = m.n[hd];
  int ha = m.n[hb];
  int hdo = m.opp[hd];
  int hao = m.n[hdo];
  int hbo = m.n[hao];

  // Get edges corresponding to halfedges
  int ed = he2e[hd];
  int eb = he2e[hb];
  int ea = he2e[ha];
  int eao = he2e[hao];
  int ebo = he2e[hbo];

  // Compute the shear for the edge ed
  Scalar la = m.l[ha];
  Scalar lb = m.l[hb];
  Scalar lao = m.l[hao];
  Scalar lbo = m.l[hbo];
  Scalar x = (la * lbo) / (lb * lao);

  // The matrix Pd corresponding to flipping edge ed is the identity except for
  // the row corresponding to edge ed, which has entries defined by Pd_scalars
  // in the column corresponding to the edge with the same index in Pd_edges
  std::vector<int> Pd_edges = { ed, ea, ebo, eao, eb };
  std::vector<Scalar> Pd_scalars = {
    -1.0, x / (1.0 + x), x / (1.0 + x), 1.0 / (1.0 + x), 1.0 / (1.0 + x)
  };

  // Compute the new row of J_del corresponding to edge ed, which is the only
  // edge that changes
  std::map<int, Scalar> J_del_d_new;
  for (int i = 0; i < 5; ++i) {
    int ei = Pd_edges[i];
    Scalar Di = Pd_scalars[i];
    for (auto it : J_del_lol[ei]) {
      J_del_d_new[it.first] += Di * it.second;

      // Delete the updated entry if it is near 0
      if (abs(J_del_d_new[it.first]) < 1e-15)
        J_del_d_new.erase(it.first);
    }
  }

  J_del_lol[ed] = J_del_d_new;
}

void
make_delaunay_with_jacobian(const Mesh<Scalar>& m,
                            const VectorX& metric_coords,
                            Mesh<Scalar>& m_del,
                            VectorX& metric_coords_del,
                            MatrixX& J_del,
                            std::vector<int>& flip_seq,
                            bool need_jacobian)
{
  // Initialize Delauany mesh with m
  //m_del = m;
  
  //make_delaunay_with_jacobian_in_place(
  //  m_del, metric_coords, metric_coords_del, J_del, flip_seq, need_jacobian);
  
  //return;

  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Build refl projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Convert embedded mesh log edge lengths to a halfedge length array l for m
  int num_edges = e2he.size();
  int num_halfedges = he2e.size();
  m_del = m;
  for (int h = 0; h < num_halfedges; ++h) {
    m_del.l[h] = exp(metric_coords[he2e[h]] / 2.0);
  }
  VectorX u;
  u.setZero(m_del.n_ind_vertices());
  DelaunayStats del_stats;
  SolveStats<Scalar> solve_stats;
  bool use_ptolemy = true;
  ConformalIdealDelaunay<Scalar>::MakeDelaunay(m_del, u, del_stats, solve_stats, use_ptolemy);
	flip_seq = del_stats.flip_seq;

  // If needed, redo the flips to build the Jacobian
  if (need_jacobian)
  {
    // Build a copy of the input mesh for length flip tracking
    Mesh<Scalar> m_jac = m;
    for (int h = 0; h < num_halfedges; ++h) {
      m_jac.l[h] = exp(metric_coords[he2e[h]] / 2.0);
    }

    // Initialize the Jacobian to the identity. We use a list of maps
    // representation for very quick row access and updates of this sparse matrix
    std::vector<std::map<int, Scalar>> J_del_lol(metric_coords.size(),
                                                std::map<int, Scalar>());
    for (int e = 0; e < metric_coords.size(); ++e) {
      J_del_lol[e][e] = 1.0;
    }

    // Duplicate all flips while building the jacobian
    int num_flips = flip_seq.size();
    for (int i = 0; i < num_flips; ++i)
    {
      int hij = flip_seq[i];
      m_jac.flip_ccw(hij);
      update_jacobian_del(m_jac, hij, he2e, J_del_lol);
    }

    // Get triplet (i, j, v) for each value v = J_del_lol[i][j] in the list of
    // maps
    std::vector<T> tripletList;
    tripletList.reserve(5 * num_edges);
    for (int i = 0; i < num_edges; ++i) {
      for (auto it : J_del_lol[i]) {
        tripletList.push_back(T(i, it.first, it.second));
      }
    }

    // Create the matrix from the triplets
    J_del.resize(metric_coords.size(), metric_coords.size());
    J_del.reserve(tripletList.size());
    J_del.setFromTriplets(tripletList.begin(), tripletList.end());

    // Check flipped meshes are consistent
    for (int h = 0; h < num_halfedges; ++h)
    {
      if (m_del.l[h] != m_jac.l[h])
      {
        spdlog::error("Inconsistent lengths");
      }
    }
  }

  // Convert Delaunay mesh halfedge lengths to log edge lengths
  metric_coords_del.resize(num_edges);
  for (int e = 0; e < num_edges; ++e) {
    metric_coords_del[e] = 2.0 * log(m_del.l[e2he[e]]);
  }
}

void
flip_edges(const Mesh<Scalar>& m,
           const VectorX& metric_coords,
           const std::vector<int>& flip_seq,
           Mesh<Scalar>& m_flip,
           VectorX& metric_coords_flip)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Initialize m_flip with m and lambdas_full
  int num_halfedges = he2e.size();
  m_flip = m;
  VectorX he_metric_coords_flip(num_halfedges);
  for (int h = 0; h < num_halfedges; ++h) {
    he_metric_coords_flip[h] = metric_coords[he2e[h]];
  }

  // Follow flip sequence with only Ptolemy flips
  for (size_t i = 0; i < flip_seq.size(); ++i) {
    int h = flip_seq[i];
    if (h < 0) {
      flip_metric_ccw(m_flip, he_metric_coords_flip, -h - 1);
    } else {
      flip_metric_ccw(m_flip, he_metric_coords_flip, h);
    }
  }

  // Restrict flip values to edges
  int num_edges = e2he.size();
  metric_coords_flip.resize(num_edges);
  for (int e = 0; e < num_edges; ++e) {
    metric_coords_flip[e] = he_metric_coords_flip[e2he[e]];
  }
}

#ifdef PYBIND
std::tuple<Mesh<Scalar>,
           VectorX,
           Eigen::SparseMatrix<Scalar, Eigen::RowMajor>,
           std::vector<int>>
make_delaunay_with_jacobian_pybind(const Mesh<Scalar>& C,
                                   const VectorX& lambdas,
                                   bool need_jacobian)
{
  Mesh<Scalar> C_del(C);
  VectorX lambdas_del;
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> J_del;
  std::vector<int> flip_seq;
  make_delaunay_with_jacobian(
    C, lambdas, C_del, lambdas_del, J_del, flip_seq, need_jacobian);

  return std::make_tuple(C_del, lambdas_del, J_del, flip_seq);
}

std::tuple<OverlayMesh<Scalar>,
           VectorX,
           Eigen::SparseMatrix<Scalar, Eigen::RowMajor>,
           std::vector<int>>
make_delaunay_with_jacobian_overlay(const OverlayMesh<Scalar>& C,
                                    const VectorX& lambdas,
                                    bool need_jacobian)
{
  OverlayMesh<Scalar> C_del(C);
  VectorX lambdas_del;
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> J_del;
  std::vector<int> flip_seq;
  make_delaunay_with_jacobian(
    C, lambdas, C_del, lambdas_del, J_del, flip_seq, need_jacobian);

  return std::make_tuple(C_del, lambdas_del, J_del, flip_seq);
}

std::tuple<Mesh<Scalar>, VectorX>
flip_edges_pybind(const Mesh<Scalar>& m,
                  const VectorX& metric_coords,
                  const std::vector<int>& flip_seq)
{
  Mesh<Scalar> m_flip;
  VectorX metric_coords_flip;
  flip_edges(m, metric_coords, flip_seq, m_flip, metric_coords_flip);

  return std::make_tuple(m_flip, metric_coords_flip);
}
#endif
}


