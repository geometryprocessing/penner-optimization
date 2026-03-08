#include "feature/core/io.h"

#include "feature/core/vf_corners.h"

#include <igl/remove_unreferenced.h>

namespace Penner {
namespace Feature {

void write_feature_edges(
    const std::string& fe_filename,
    const std::vector<VertexEdge>& feature_edges) {
    std::ofstream output_file(fe_filename, std::ios::out | std::ios::trunc);
    int n = feature_edges.size();
    for (int i = 0; i < n; ++i) {
        output_file << feature_edges[i][0] << " " << feature_edges[i][1] << std::endl;
    }
    output_file.close();
}

std::vector<VertexEdge> load_feature_edges(const std::string& fe_filename) {
    // try to open mesh
	std::ifstream inf(fe_filename);
	if (!inf) {
		spdlog::error("Failed to load feature edges file\n");
		exit(EXIT_FAILURE);
	}
	
    // get count
	std::vector<VertexEdge> feature_edges = {};
	std::string line{};

    // get all edges
	while (std::getline(inf, line)) {
		std::istringstream iss(line);
		int v1, v2;
		iss >> v1 >> v2;
		feature_edges.push_back({v1, v2});
	}

	return feature_edges;
}

std::vector<VertexEdge> load_mesh_edges(const std::string& fe_filename) {
    // try to open mesh
	std::ifstream inf(fe_filename);
	if (!inf) {
		spdlog::error("Failed to load feature edges file\n");
		exit(EXIT_FAILURE);
	}

    // get all edges
	std::vector<VertexEdge> feature_edges = {};
	std::string line{};
	std::getline(inf, line);
	while (std::getline(inf, line)) {
		char label;
		std::istringstream iss(line);

        // make sure edge line
		iss >> label;
		if (label != 'l') 
			continue;

        // get edge vertices
		int v1;
		int v2;
		iss >> v1 >> v2;

        // convert to 0 indexing
		feature_edges.push_back({v1 - 1, v2 - 1});
	}

	return feature_edges;
}

void write_mesh_edges(
    const std::string& fe_filename,
    const std::vector<VertexEdge>& feature_edges) {
    std::ofstream output_file(fe_filename, std::ios::out | std::ios::app);
    int n = feature_edges.size();
    for (int i = 0; i < n; ++i) {
        output_file << "l " << feature_edges[i][0] + 1;
        output_file << " " << feature_edges[i][1] + 1 << std::endl;
    }
    output_file.close();
}


// write edge geometry to file
void write_edges(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& E
) {
    // write all feature edge vertices
    std::ofstream output_file(filename, std::ios::out | std::ios::trunc);
    for (int vi = 0; vi < V.rows(); ++vi)
    {
        output_file << "v ";
        for (int i = 0; i < 3; ++i)
        {
            output_file << std::fixed << std::setprecision(17) << V(vi, i) << " ";
        }
        output_file << std::endl;
    }
    for (int eij = 0; eij < E.rows(); ++eij)
    {
        output_file << "l " << E(eij, 0) + 1 << " " << E(eij, 1) + 1 << std::endl;
    }
    output_file.close();
}


void write_seams(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv,
    const Eigen::MatrixXi& F_is_feature)
{
    Eigen::MatrixXi F_is_seam = mask_difference(find_seams(F, F_uv), F_is_feature);
    auto [VN, EN] = generate_edges(V, F, F_is_seam);
    write_edges(filename, VN, EN);
}


void write_features(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_is_feature)
{
    auto [VN, EN] = generate_edges(V, F, F_is_feature);
    write_edges(filename, VN, EN);
}

void write_boundary(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& F_uv)
{
    Eigen::MatrixXi F_is_seam = find_seams(F, F_uv);
    auto [VN, EN] = generate_edges(V, F, F_is_seam);
    write_edges(filename, VN, EN);

}


} // namespace Feature
} // namespace Penner
