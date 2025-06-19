
/// Build a VF mesh for the embedded mesh in the doubled mesh and also extract
/// the mapping from VF mesh corners to opposite halfedge index
///
/// @param[in] m: input mesh
/// @param[in] vtx_reindex: reindexing for the vertices wrt the mesh indexing
/// @param[out] F: mesh faces
/// @param[out] corner_to_halfedge: mesh corner to opposite halfedge indexing
void extract_embedded_mesh(
    const Mesh<Scalar>& m,
    const std::vector<int>& vtx_reindex,
    Eigen::MatrixXi& F,
    Eigen::MatrixXi& corner_to_halfedge)
{
    // Get number of vertices and faces for the embedded mesh
    int num_faces = m.n_faces();
    int num_embedded_faces = 0;
    for (int f = 0; f < num_faces; ++f) {
        // Skip face if it is in the doubled mesh
        int hij = m.h[f];
        // TODO Hack for closed overlay meshes. A better solution would be to have
        // a separate method for overlay meshes
        if ((!m.type.empty()) && (m.type[hij] == 2)) continue;

        // Count embedded faces
        num_embedded_faces++;
    }

    // Build faces and halfedge lengths
    std::vector<std::array<int, 3>> F_vec, corner_to_halfedge_vec;
    F_vec.reserve(num_embedded_faces);
    corner_to_halfedge_vec.reserve(num_embedded_faces);
    for (int f = 0; f < num_faces; ++f) {
        // Get halfedges of face
        int hli = m.h[f];
        int hij = m.n[hli];
        int hjk = m.n[hij];

        // TODO Hack for closed overlay meshes. A better solution would be to have
        // a separate method for overlay meshes
        if ((!m.type.empty()) && (m.type[hij] == 2)) continue;

        // Get vertices of the fist face
        int vi = m.to[hli];
        int vj = m.to[hij];
        int vk = m.to[hjk];

        // Triangle case (l == k)
        if (m.n[hjk] == hli) {
            // Build vertex embedding for the face
            F_vec.push_back({vtx_reindex[vi], vtx_reindex[vj], vtx_reindex[vk]});

            // Build halfedge index map for the face
            corner_to_halfedge_vec.push_back({hjk, hli, hij});
        }
        // Polygon case
        else {
            // Build vertex embedding for the first face
            F_vec.push_back({vtx_reindex[vi], vtx_reindex[vj], vtx_reindex[vk]});

            // Build halfedge index map for the first face
            corner_to_halfedge_vec.push_back({hjk, -1, hij});

            int hkp = m.n[hjk];
            while (m.n[hkp] != hli) {
                // Get vertices of the interior face
                int vk = m.to[hjk];
                int vp = m.to[hkp];

                // Build vertex embedding for the interior face for halfedge hkp
                F_vec.push_back({vtx_reindex[vi], vtx_reindex[vk], vtx_reindex[vp]});

                // Build halfedge index map for the interior face for halfedge hkp
                corner_to_halfedge_vec.push_back({hkp, -1, -1});

                // Increment halfedges
                hkp = m.n[hkp];
                hjk = m.n[hjk];
            }

            // Get vertices of the final face (p == l)
            int vk = m.to[hjk];
            int vp = m.to[hkp];

            // Build vertex embedding for the final face
            F_vec.push_back({vtx_reindex[vi], vtx_reindex[vk], vtx_reindex[vp]});

            // Build halfedge index map for the final face
            corner_to_halfedge_vec.push_back({hkp, hli, -1});
        }
    }

    // Copy lists of lists to matrices
    int num_triangles = F_vec.size();
    F.resize(num_triangles, 3);
    corner_to_halfedge.resize(num_triangles, 3);
    for (int fi = 0; fi < num_triangles; ++fi) {
        for (int j = 0; j < 3; ++j) {
            F(fi, j) = F_vec[fi][j];
            corner_to_halfedge(fi, j) = corner_to_halfedge_vec[fi][j];
        }
    }
}