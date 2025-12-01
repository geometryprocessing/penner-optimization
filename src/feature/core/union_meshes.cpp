
#include "feature/core/union_meshes.h"

namespace Penner {
namespace Feature {

int count_total_vertices(std::vector<Eigen::MatrixXd>& mesh_vertex_components)
{
    int num_vertices = 0;
    for (const auto& mesh_vertex_component : mesh_vertex_components) {
        num_vertices += mesh_vertex_component.rows();
    }

    return num_vertices;
}

int count_total_faces(std::vector<Eigen::MatrixXi>& mesh_face_components)
{
    int num_faces = 0;
    for (const auto& mesh_face_component : mesh_face_components) {
        num_faces += mesh_face_component.rows();
    }

    return num_faces;
}

// compute the sum of vector sizes for a vector of vectors
template <typename VectorType>
int compute_total_domain_size(const std::vector<VectorType>& maps)
{
    int count = 0;
    for (const auto& map : maps) {
        count += map.size();
    }

    return count;
}

// concatenate a vector of vectors into a single vector
template <typename T>
std::vector<T> union_attributes(const std::vector<std::vector<T>>& attributes)
{
    // Precompute the total attribute domain size
    int total_domain_size = compute_total_domain_size(attributes);
    std::vector<T> total_attribute = {};
    total_attribute.reserve(total_domain_size);

    for (const auto& attribute : attributes) {
        // Add attributes
        for (size_t j = 0; j < attribute.size(); ++j) {
            total_attribute.push_back(attribute[j]);
        }
    }

    return total_attribute;
}

Mesh<Scalar> union_meshes(const std::vector<Mesh<Scalar>>& meshes)
{
    Mesh<Scalar> total_mesh;

    // build lists of attribute counts per component
    std::vector<int> n_h, n_f, n_v, n_ind_v;
    for (const auto& mesh : meshes) {
        n_h.push_back(mesh.n_halfedges());
        n_f.push_back(mesh.n_faces());
        n_v.push_back(mesh.n_vertices());
        n_ind_v.push_back(mesh.n_ind_vertices());
    }

    // build list of lists of element maps per component
    std::vector<std::vector<int>> n, to, f, h, out, opp, R, v_rep;
    for (const auto& mesh : meshes) {
        n.push_back(mesh.n);
        to.push_back(mesh.to);
        f.push_back(mesh.f);
        h.push_back(mesh.h);
        out.push_back(mesh.out);
        opp.push_back(mesh.opp);
        R.push_back(mesh.R);
        v_rep.push_back(mesh.v_rep);
    }

    // build combined element maps by unioning the maps
    total_mesh.n = union_maps(n, n_h);
    total_mesh.to = union_maps(to, n_v);
    total_mesh.f = union_maps(f, n_f);
    total_mesh.h = union_maps(h, n_h);
    total_mesh.out = union_maps(out, n_h);
    total_mesh.opp = union_maps(opp, n_h);
    total_mesh.R = union_maps(R, n_h);
    total_mesh.v_rep = union_maps(v_rep, n_ind_v);

    // build list of lists of element attributes per component
    std::vector<std::vector<char>> type, type_input;
    std::vector<std::vector<Scalar>> l, Th_hat;
    std::vector<std::vector<bool>> fixed_dof;
    for (const auto& mesh : meshes) {
        type.push_back(mesh.type);
        type_input.push_back(mesh.type_input);
        l.push_back(mesh.l);
        Th_hat.push_back(mesh.Th_hat);
        fixed_dof.push_back(mesh.fixed_dof);
    }

    // build combined attributes by unioning the attributes
    total_mesh.type = union_attributes(type);
    total_mesh.type_input = union_attributes(type_input);
    total_mesh.l = union_attributes(l);
    total_mesh.Th_hat = union_attributes(Th_hat);
    total_mesh.fixed_dof = union_attributes(fixed_dof);

    return total_mesh;
}

MarkedPennerConeMetric union_marked_metrics(
    const std::vector<MarkedPennerConeMetric>& marked_metrics)
{
    // union meshes, metric coordinates, and holonomy prescriptions
    std::vector<Mesh<Scalar>> meshes;
    std::vector<VectorX> metric_coords;
    std::vector<std::vector<Scalar>> kappa;
    for (const auto& marked_metric : marked_metrics) {
        meshes.push_back(marked_metric);
        metric_coords.push_back(marked_metric.get_metric_coordinates());
        kappa.push_back(marked_metric.kappa_hat);
    }
    Mesh<Scalar> total_mesh = union_meshes(meshes);
    VectorX total_metric_coords = union_vectors(metric_coords);
    std::vector<Scalar> total_kappa = union_attributes(kappa);

    // reindex and union homology basis loops
    int offset = 0; // global face index offset
    std::vector<std::unique_ptr<DualLoop>> homology_basis_loops;
    for (const auto& marked_metric : marked_metrics) {
        // get basis loops for component
        const auto& basis_loops = marked_metric.get_homology_basis_loops();
        std::vector<std::unique_ptr<DualLoop>> component_homology_basis_loops;

        // reconstruct basis loops on global mesh
        for (const auto& basis_loop : basis_loops) {
            // convert basis loop to a face sequence
            std::vector<int> face_loop = basis_loop->generate_face_sequence(marked_metric);

            // reindex face indices from local to global mesh
            int face_loop_size = face_loop.size();
            for (int i = 0; i < face_loop_size; ++i) {
                face_loop[i] += offset;
            }

            // convert face sequence back to homology loops
            homology_basis_loops.push_back(
                std::make_unique<Holonomy::DualLoopConnectivity>(Holonomy::DualLoopConnectivity(
                    Holonomy::build_dual_path_from_face_sequence(total_mesh, face_loop))));
        }

        // incremement face offset count
        offset += marked_metric.n_faces();
    }

    return MarkedPennerConeMetric(
        total_mesh,
        total_metric_coords,
        homology_basis_loops,
        total_kappa);
}

} // namespace Feature
} // namespace Penner