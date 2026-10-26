#include "resolution/experimental/representatives.h"
#include "util/union_find.h"

namespace Penner
{
namespace Resolution
{

std::vector<std::vector<int>> generate_face_union(int num_faces)
{
    std::vector<std::vector<int>> F_he(num_faces, std::vector<int>(3, -1));
    int hij = 0;
    for (int i = 0; i < num_faces; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            F_he[i][j] = hij;
            hij++;
        }
    }
    return F_he;
}

std::vector<std::vector<int>> glue_face_union(
    const std::vector<std::vector<int>>& F_he,
    const std::vector<int>& gluing_map)
{
    // glue together faces into edges
    int num_faces = F_he.size();
    std::vector<std::vector<int>> F(num_faces, std::vector<int>(3, -1));
    for (int i = 0; i < num_faces; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            int hij = F_he[i][j];
            F[i][j] = gluing_map[hij];
        }
    }

    return F;
}

std::vector<std::vector<int>> generate_radial_surface(int genus)
{
    // initialize list of faces with halfedges
    int num_polygon_edges = 4 * genus;
    int num_faces = num_polygon_edges;
    std::vector<std::vector<int>> F_he = generate_face_union(num_faces);
    
    // build identification map for polygon
    int num_halfedges = 3 * num_faces;
    UnionFind edge_gluing(num_halfedges);

    // glue triangles into polygon and list other edges in order 
    std::vector<int> top_halfedges, bottom_halfedges;
    top_halfedges.reserve(2 * genus);
    bottom_halfedges.reserve(2 * genus);
    top_halfedges.push_back(F_he[0][0]);
    bottom_halfedges.push_back(F_he[0][1]);

    // glue triangles into a single polygon
    for (int i = 0; i < num_faces; ++i)
    {
        int j = (i + 1) % num_faces;
        edge_gluing.union_sets(F_he[i][2], F_he[j][1]);
    }

    // glue together top and bottom
    for (int i = 0; i < 2 * genus; ++i)
    {
        int j = 2 * genus + i;
        edge_gluing.union_sets(F_he[i][0], F_he[j][0]);
    }

    // get gluing map
    std::vector<int> gluing_map = edge_gluing.index_sets();

    return glue_face_union(F_he, gluing_map);
}

std::vector<std::vector<int>> generate_representative_surface(int genus)
{
    // initialize list of faces with halfedges
    int num_polygon_edges = 4 * genus;
    int num_faces = num_polygon_edges - 2;
    std::vector<std::vector<int>> F_he = generate_face_union(num_faces);
    
    // build identification map for polygon
    int num_halfedges = 3 * num_faces;
    UnionFind edge_gluing(num_halfedges);

    // glue triangles into polygon and list other edges in order 
    std::vector<int> top_halfedges, bottom_halfedges;
    top_halfedges.reserve(2 * genus);
    bottom_halfedges.reserve(2 * genus);
    top_halfedges.push_back(F_he[0][0]);
    bottom_halfedges.push_back(F_he[0][1]);
    int he_prev = F_he[0][2];
    for (int i = 1; i < num_faces - 1; ++i)
    {
        if (i % 2)
        {
            top_halfedges.push_back(F_he[i][0]);
            edge_gluing.union_sets(he_prev, F_he[i][1]);
            he_prev = F_he[i][2];
        }
        else
        {
            edge_gluing.union_sets(he_prev, F_he[i][0]);
            bottom_halfedges.push_back(F_he[i][1]);
            he_prev = F_he[i][2];
        }
    }
    top_halfedges.push_back(F_he[num_faces - 1][0]);
    edge_gluing.union_sets(he_prev, F_he[num_faces - 1][1]);
    bottom_halfedges.push_back(F_he[num_faces - 1][2]);

    // glue together top and bottom
    for (int i = 0; i < 2 * genus; ++i)
    {
        int top_he = top_halfedges[i];
        int bottom_he = bottom_halfedges[2 * genus - 1 - i];
        edge_gluing.union_sets(top_he, bottom_he);
    }

    // get gluing map
    std::vector<int> gluing_map = edge_gluing.index_sets();

    return glue_face_union(F_he, gluing_map);
}

}
}