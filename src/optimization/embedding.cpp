#include "embedding.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {

void
build_edge_maps(const Mesh<Scalar>& m,
                std::vector<int>& he2e,
                std::vector<int>& e2he)
{
  int num_halfedges = m.opp.size();
  int num_edges = num_halfedges / 2;
  he2e.resize(num_halfedges);
  e2he.clear();
  e2he.reserve(num_edges);

  // First build map from edges to the lower index halfedges, which is a
  // bijection
  for (int h = 0; h < num_halfedges; ++h) {
    if (h < m.opp[h]) {
      e2he.push_back(h);
    }
  }

  // Construct map from halfedges to edges
  for (int e = 0; e < num_edges; ++e) {
    int h = e2he[e];
    he2e[h] = e;
    he2e[m.opp[h]] = e;
  }
}

void
build_refl_proj(const Mesh<Scalar>& m,
                std::vector<int>& proj,
                std::vector<int>& embed)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Resize arrays
  proj.resize(e2he.size());
  embed.clear();
  embed.reserve(e2he.size());

  // Build injective map from edges that are not type 2 to double mesh
  int num_edges = e2he.size();
  for (int e = 0; e < num_edges; ++e) {
    int h0 = m.h0(e2he[e]);
    int h1 = m.h1(e2he[e]);
    if ((m.type[h0] != 2) || (m.type[h1] != 2)) {
      embed.push_back(e);
    }
  }

  // Construct map from double mesh to the embedded mesh
  // Map reflection of edges in the image of the embedding to the original edge
  int num_embedded_edges = embed.size();
  for (int E = 0; E < num_embedded_edges; ++E) {
    int e = embed[E];
    int Re = he2e[m.R[e2he[e]]];
    proj[Re] = E;
  }

  // Map embedded edge to itself. Note that if E is identified with e = embed[E]
  // then this implies proj[E] = E and the map is a projection.
  for (int E = 0; E < num_embedded_edges; ++E) {
    int e = embed[E];
    proj[e] = E;
  }
}

void
build_refl_he_proj(const Mesh<Scalar>& m,
                   std::vector<int>& he_proj,
                   std::vector<int>& he_embed)
{
  // Get edge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Resize arrays
  int num_halfedges = he2e.size();
  int num_edges = e2he.size();
  he_proj.resize(num_halfedges);
  he_embed.clear();
  he_embed.reserve(num_edges);

  // Build injective map from halfedges that are not type 2 to double mesh
  for (int h = 0; h < num_halfedges; ++h) {
    if (m.type[h] != 2) {
      he_embed.push_back(h);
    }
  }

  // Construct map from double mesh to the embedded mesh
  // Map reflection of halfedges in the image of the embedding to the original
  // halfedge
  int num_embedded_halfedges = he_embed.size();
  for (int H = 0; H < num_embedded_halfedges; ++H) {
    int h = he_embed[H];
    int Rh = m.R[h];
    he_proj[Rh] = H;
  }
  // Map embedded halfedge to itself. Note that if H is identified with h =
  // embed[H] then this implies proj[H] = H and the map is a projection.
  for (int H = 0; H < num_embedded_halfedges; ++H) {
    int h = he_embed[H];
    he_proj[h] = H;
  }
}

void
build_refl_matrix(const Mesh<Scalar>& m, MatrixX& projection)
{
  // Get reflection projection and embedding
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  // Creatae projection matrix
  int num_edges = proj.size();
  int num_embedded_edges = embed.size();
  std::vector<T> tripletList;
  tripletList.reserve(num_edges);
  for (int e = 0; e < num_edges; ++e) {
    tripletList.push_back(T(e, proj[e], 1.0));
  }
  projection.resize(num_edges, num_embedded_edges);
  projection.reserve(tripletList.size());
  projection.setFromTriplets(tripletList.begin(), tripletList.end());
}

bool
is_embedded_edge(
  const std::vector<int>& proj,
  const std::vector<int>& embed,
  int e
) {
  return (embed[proj[e]] == e);
}

void
reduce_symmetric_function(const std::vector<int>& embed,
                          const VectorX& symmetric_function,
                          VectorX& reduced_function)
{
  // Define f[E] by the value of f_full at e = embed[E]
  int num_embedded_edges = embed.size();
  reduced_function.resize(num_embedded_edges);
  for (int E = 0; E < num_embedded_edges; ++E) {
    int e = embed[E];
    reduced_function[E] = symmetric_function[e];
  }
}

void
expand_reduced_function(const std::vector<int>& proj,
                        const VectorX& reduced_function,
                        VectorX& symmetric_function)
{
  // Define f_full[e] by the value of f at E = proj[e]. This implies that
  // f_full[embed[e]] = f[proj[embed[e]] = f[e]
  int num_edges = proj.size();
  symmetric_function.resize(num_edges);
  for (int e = 0; e < num_edges; ++e) {
    int E = proj[e];
    symmetric_function[e] = reduced_function[E];
  }
}

void
restrict_he_func(const Mesh<Scalar>& m, const VectorX& f_he, VectorX& f_e)
{
  // Build edge to halfedge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Restrict f_he to edges
  int n_edges = e2he.size();
  f_e.resize(n_edges);
  for (int e = 0; e < n_edges; ++e) {
    int h = e2he[e];
    f_e[e] = f_he[h];
  }
}

void
expand_edge_func(const Mesh<Scalar>& m, const VectorX& f_e, VectorX& f_he)
{
  // Build edge to halfedge maps
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  // Expand f_e to halfedges
  int n_halfedges = he2e.size();
  f_he.resize(n_halfedges);
  for (int h = 0; h < n_halfedges; ++h) {
    int e = he2e[h];
    f_he[h] = f_e[e];
  }
}

bool
is_valid_halfedge(const Mesh<Scalar>& m)
{
  // Build prev array
  int num_halfedges = m.n.size();
  std::vector<int> prev(num_halfedges);
  for (int h = 0; h < num_halfedges; ++h) {
    prev[m.n[h]] = h;
  }

  // Check halfedge conditions
  for (int h = 0; h < num_halfedges; ++h) {
    // opp^2 is the identity
    if (m.opp[m.opp[h]] != h)
      return false;
    // opp has no fixed points
    if (m.opp[h] == h)
      return false;
    // Triangle mesh
    if (m.n[m.n[m.n[h]]] != h)
      return false;
  }

  // Check vertex conditions
  int num_vertices = m.out.size();
  for (int v = 0; v < num_vertices; ++v) {
    // opp reverses the tail and tip of out
    if (m.to[m.opp[m.out[v]]] != v)
      return false;
  }

  return true;
}

bool
is_valid_symmetry(const Mesh<Scalar>& m)
{
  bool error_found = true;

  // Check mesh is valid
  if (!is_valid_halfedge(m)) {
    std::cerr << "Invalid halfedge" << std::endl;
    error_found = false;
  }

  // Check halfedge conditions
  int num_halfedges = m.n.size();
  for (int h = 0; h < num_halfedges; ++h) {
    // Reversing n
    if (m.n[m.n[m.R[h]]] != m.R[m.n[h]]) {
      std::cerr << "R does not reverse n at " << h << std::endl;
      error_found = false;
    }

    // Commuting with opp
    if (m.opp[m.R[h]] != m.R[m.opp[h]]) {
      std::cerr << "R does not commute with opp at " << h << std::endl;
      error_found = false;
    }

    // Halfedges partitioned
    if ((m.type[h] == 1) && (m.type[m.R[h]] != 2)) {
      std::cerr << "Type 1 halfedge does not map to a type 1 halfedge at " << h
                << std::endl;
      error_found = false;
    }
    if ((m.type[h] == 2) && (m.type[m.R[h]] != 1)) {
      std::cerr << "Type 1 halfedge does not map to a type 1 halfedge at " << h
                << std::endl;
      error_found = false;
    }
    if ((m.type[h] == 3) && (m.R[h] != h)) {
      std::cerr << "Type 3 halfedge does not map to itself at " << h
                << std::endl;
      error_found = false;
    }
  }

  // Check face conditions
  int num_faces = m.h.size();
  for (int f = 0; f < num_faces; ++f) {
    // Face maps to itself
    if (m.f[m.R[m.h[f]]] == f) {
      continue;
    }
    int h = m.h[f];
    if (h == 1) {
      if ((m.n[h] != 1) || (m.n[m.n[h]] != 1)) {
        std::cerr << "Interior face not labelled consistently at " << f
                  << std::endl;
        error_found = false;
      }
      if ((m.R[h] != 2) || (m.n[m.R[h]] != 2) || (m.n[m.n[m.R[h]]] != 2)) {
        std::cerr << "Interior face not labelled consistently at " << f
                  << std::endl;
        error_found = false;
      }
    }
    if (h == 2) {
      if ((m.n[h] != 2) || (m.n[m.n[h]] != 2)) {
        std::cerr << "Interior face not labelled consistently at " << f
                  << std::endl;
        error_found = false;
      }
      if ((m.R[h] != 1) || (m.n[m.R[h]] != 1) || (m.n[m.n[m.R[h]]] != 1)) {
        std::cerr << "Interior face not labelled consistently at " << f
                  << std::endl;
        error_found = false;
      }
    }
  }

  return error_found;
}

#ifdef PYBIND
// Pybind definitions
std::tuple<std::vector<int>, std::vector<int>>
build_edge_maps_pybind(const Mesh<Scalar>& m)
{
  std::vector<int> he2e;
  std::vector<int> e2he;
  build_edge_maps(m, he2e, e2he);

  return std::make_tuple(he2e, e2he);
}

std::tuple<std::vector<int>, std::vector<int>>
build_refl_proj_pybind(const Mesh<Scalar>& m)
{
  std::vector<int> proj;
  std::vector<int> embed;
  build_refl_proj(m, proj, embed);

  return std::make_tuple(proj, embed);
}

std::tuple<std::vector<int>, std::vector<int>>
build_refl_he_proj_pybind(const Mesh<Scalar>& m)
{
  std::vector<int> he_proj;
  std::vector<int> he_embed;
  build_refl_he_proj(m, he_proj, he_embed);

  return std::make_tuple(he_proj, he_embed);
}

MatrixX
build_refl_matrix_pybind(const Mesh<Scalar>& m)
{
  MatrixX projection;
  build_refl_matrix(m, projection);
  return projection;
}

#endif

}
