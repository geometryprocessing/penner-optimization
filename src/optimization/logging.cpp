#include "logging.hh"
#include "energies.hh"
#include "embedding.hh"

/// FIXME Do cleaning pass

namespace CurvatureMetric {

void
create_log(const std::filesystem::path& log_dir, const std::string& log_name)
{
	// If the log directory is trivial, use standard output
  if (log_dir.empty()) {
    auto ostream_sink =
      std::make_shared<spdlog::sinks::ostream_sink_mt>(std::cout);
    auto logger =
      std::make_shared<spdlog::logger>(log_name, ostream_sink);
    spdlog::register_logger(logger);
	// Make a file logger if an output path is specified
  } else {
    std::filesystem::path log_dir_path(log_dir);
    std::filesystem::create_directories(log_dir_path);
    std::filesystem::path log_path =
      join_path(log_dir_path, log_name + ".log");
		spdlog::basic_logger_mt(log_name, log_path);
  }
}

void
log_mesh_information(const Mesh<Scalar>& m, const std::string& log_name)
{
  ReductionMaps reduction_maps(m);
  std::vector<int> cone_vertices;
  compute_cone_vertices(m, reduction_maps, cone_vertices);
  spdlog::get(log_name)->trace(
    "Fixed vertices are {}",
    formatted_vector(reduction_maps.fixed_v)
  );
  spdlog::get(log_name)->trace(
    "Cone vertices are {}",
    formatted_vector(cone_vertices)
  );
  spdlog::get(log_name)->trace(
    "Mesh next map: {}",
    formatted_vector(m.n)
  );
  spdlog::get(log_name)->trace(
    "Mesh face to halfedge map: {}",
    formatted_vector(m.h)
  );
  spdlog::get(log_name)->trace(
    "Mesh halfedge to face map: {}",
    formatted_vector(m.f)
  );
}

void write_vector(
	const VectorX &vec,
	const std::string &filename,
	int precision
) {
	std::ofstream output_file(filename, std::ios::out | std::ios::trunc);
  for (Eigen::Index i = 0; i < vec.size(); ++i)
  {
    output_file << std::setprecision(precision) << vec[i] << std::endl;
  }
  output_file.close();
}

}
