// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "util/io.h"

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/ostream_sink.h"
#include <H5Cpp.h>
#include <igl/writeOBJ.h>

#include "util/vector.h"

namespace Penner {

namespace {

H5::DSetCreatPropList create_matrix_props(hsize_t rows, hsize_t cols)
{
    H5::DSetCreatPropList props;
    hsize_t chunk_dims[2] = {
        std::max<hsize_t>(1, std::min<hsize_t>(rows, 1024)),
        std::max<hsize_t>(1, cols)
    };
    props.setChunk(2, chunk_dims);
    props.setDeflate(6);
    return props;
}

}

void create_log(const std::filesystem::path& log_dir, const std::string& log_name)
{
    // If the log directory is trivial, use standard output
    if (log_dir.empty()) {
        auto ostream_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(std::cout);
        auto logger = std::make_shared<spdlog::logger>(log_name, ostream_sink);
        spdlog::register_logger(logger);
        // Make a file logger if an output path is specified
    } else {
        std::filesystem::path log_dir_path(log_dir);
        std::filesystem::create_directories(log_dir_path);
        std::filesystem::path log_path = join_path(log_dir_path, log_name + ".log");
        spdlog::basic_logger_mt(log_name, log_path);
    }
}

void log_mesh_information(const Mesh<Scalar>& m, const std::string& log_name)
{
    ReductionMaps reduction_maps(m);
    std::vector<int> cone_vertices;
    // compute_cone_vertices(m, reduction_maps, cone_vertices);
    spdlog::get(log_name)->trace("Fixed vertices are {}", formatted_vector(reduction_maps.fixed_v));
    // spdlog::get(log_name)->trace(
    //   "Cone vertices are {}",
    //   formatted_vector(cone_vertices)
    //);
    spdlog::get(log_name)->trace("Mesh next map: {}", formatted_vector(m.n));
    spdlog::get(log_name)->trace("Mesh face to halfedge map: {}", formatted_vector(m.h));
    spdlog::get(log_name)->trace("Mesh halfedge to face map: {}", formatted_vector(m.f));
}

void write_matrix(const Eigen::MatrixXd& matrix, const std::string& filename, std::string separator)
{
    if (matrix.cols() == 0) {
        return;
    }

    // Open file
    std::ofstream output_file;
    output_file.open(filename);

    // Iterate over rows
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        // Iterate over columns of row i
        Scalar v = matrix(i, 0);
        output_file << std::fixed << std::setprecision(17) << v;
        for (Eigen::Index j = 1; j < matrix.cols(); ++j) {
            Scalar v = matrix(i, j);
            output_file << std::fixed << std::setprecision(17) << separator << v;
        }

        // Add newline to end of row
        output_file << std::endl;
    }

    // Close file
    output_file.close();
}

void write_integer_matrix(const Eigen::MatrixXi& matrix, const std::string& filename, std::string separator)
{
    if (matrix.cols() == 0) {
        return;
    }

    // Open file
    std::ofstream output_file;
    output_file.open(filename);

    // Iterate over rows
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        // Iterate over columns of row i
        Scalar v = matrix(i, 0);
        output_file << v;
        for (Eigen::Index j = 1; j < matrix.cols(); ++j) {
            Scalar v = matrix(i, j);
            output_file << separator << v;
        }

        // Add newline to end of row
        output_file << std::endl;
    }

    // Close file
    output_file.close();
}


 Eigen::MatrixXd read_matrix(const std::string& filename)
{
    Eigen::MatrixXd matrix;

    // Open file
    std::ifstream input_file(filename);
    if (!input_file) return {};

    // Read file
    std::vector<std::array<double, 3>> matrix_vec = {};
    std::string line;
    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        double v1, v2, v3;
        iss >> v1 >> v2 >> v3;
        matrix_vec.push_back({v1, v2, v3});
    }

    // Close file
    input_file.close();

    return convert_std_to_eigen_matrix(matrix_vec);
}

void write_obj_with_uv(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    Eigen::MatrixXd N;
    Eigen::MatrixXi FN;
    igl::writeOBJ(filename, V, F, N, FN, uv, F_uv);
}

void write_hdf5_mesh(const std::string& path,
                const Eigen::MatrixXd& V,   // Nx3
                const Eigen::MatrixXi& F)   // Mx3
{
    H5::H5File file(path, H5F_ACC_TRUNC);

    // Vertices
    hsize_t v_dims[2] = {(hsize_t)V.rows(), 3};
    H5::DataSpace v_space(2, v_dims);
    H5::DataSet v_set = file.createDataSet(
        "vertices", H5::PredType::NATIVE_DOUBLE, v_space, create_matrix_props(v_dims[0], v_dims[1]));
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V_row_major = V;
    v_set.write(V_row_major.data(), H5::PredType::NATIVE_DOUBLE);

    // Faces
    hsize_t f_dims[2] = {(hsize_t)F.rows(), 3};
    H5::DataSpace f_space(2, f_dims);
    H5::DataSet f_set = file.createDataSet(
        "faces", H5::PredType::NATIVE_INT, f_space, create_matrix_props(f_dims[0], f_dims[1]));
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F_row_major = F;
    f_set.write(F_row_major.data(), H5::PredType::NATIVE_INT);
}

void write_hdf5_mesh_with_uv(const std::string& path,
                const Eigen::MatrixXd& V,
                const Eigen::MatrixXi& F,
                const Eigen::MatrixXd& uv,
                const Eigen::MatrixXi& FT)
{
    H5::H5File file(path, H5F_ACC_TRUNC);

    // Vertices
    hsize_t v_dims[2] = {(hsize_t)V.rows(), 3};
    H5::DataSpace v_space(2, v_dims);
    H5::DataSet v_set = file.createDataSet(
        "vertices", H5::PredType::NATIVE_DOUBLE, v_space, create_matrix_props(v_dims[0], v_dims[1]));
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V_row_major = V;
    v_set.write(V_row_major.data(), H5::PredType::NATIVE_DOUBLE);

    // Faces
    hsize_t f_dims[2] = {(hsize_t)F.rows(), 3};
    H5::DataSpace f_space(2, f_dims);
    H5::DataSet f_set = file.createDataSet(
        "faces", H5::PredType::NATIVE_INT, f_space, create_matrix_props(f_dims[0], f_dims[1]));
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F_row_major = F;
    f_set.write(F_row_major.data(), H5::PredType::NATIVE_INT);

    // uv Vertices
    hsize_t uv_dims[2] = {(hsize_t)uv.rows(), 2};
    H5::DataSpace uv_space(2, uv_dims);
    H5::DataSet uv_set = file.createDataSet(
        "uv_vertices", H5::PredType::NATIVE_DOUBLE, uv_space, create_matrix_props(uv_dims[0], uv_dims[1]));
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> uv_row_major = uv;
    uv_set.write(uv_row_major.data(), H5::PredType::NATIVE_DOUBLE);

    // Faces
    hsize_t ft_dims[2] = {(hsize_t)FT.rows(), 3};
    H5::DataSpace ft_space(2, ft_dims);
    H5::DataSet ft_set = file.createDataSet(
        "uv_faces", H5::PredType::NATIVE_INT, ft_space, create_matrix_props(ft_dims[0], ft_dims[1]));
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FT_row_major = FT;
    ft_set.write(FT_row_major.data(), H5::PredType::NATIVE_INT);
}

void write_hdf5_scalar_field(const std::string& path,
                        const std::string& name,
                        const Eigen::VectorXd& scalar_field)
{
    // Open existing file in read/write mode (no truncation)
    H5::H5File file(path, H5F_ACC_RDWR);

    // Open or create attributes group
    H5::Group attrs;
    if (file.exists("attributes"))
        attrs = file.openGroup("attributes");
    else
        attrs = file.createGroup("attributes");

    hsize_t dims[1] = {(hsize_t)scalar_field.size()};
    H5::DataSet dset = attrs.createDataSet(
        name, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(1, dims));
    dset.write(scalar_field.data(), H5::PredType::NATIVE_DOUBLE);
}

void write_hdf5_vector_field(const std::string& path,
                        const std::string& name,
                        const Eigen::MatrixXd& vector_field)
{
    // Open existing file in read/write mode (no truncation)
    H5::H5File file(path, H5F_ACC_RDWR);

    // Open or create attributes group
    H5::Group attrs;
    if (file.exists("attributes"))
        attrs = file.openGroup("attributes");
    else
        attrs = file.createGroup("attributes");

    hsize_t dims[2] = {(hsize_t)vector_field.rows(), (hsize_t)vector_field.cols()};
    H5::DataSpace space(2, dims);
    H5::DataSet dset = attrs.createDataSet(
        name, H5::PredType::NATIVE_DOUBLE, space, create_matrix_props(dims[0], dims[1]));
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vector_field_rm = vector_field;
    dset.write(vector_field_rm.data(), H5::PredType::NATIVE_DOUBLE);
}

void write_hdf5_integer_matrix(const std::string& path,
                        const std::string& name,
                        const Eigen::MatrixXi& vector_field)
{
    // Open existing file in read/write mode (no truncation)
    H5::H5File file(path, H5F_ACC_RDWR);

    // Open or create attributes group
    H5::Group attrs;
    if (file.exists("attributes"))
        attrs = file.openGroup("attributes");
    else
        attrs = file.createGroup("attributes");

    hsize_t dims[2] = {(hsize_t)vector_field.rows(), (hsize_t)vector_field.cols()};
    H5::DataSpace space(2, dims);
    H5::DataSet dset = attrs.createDataSet(
        name, H5::PredType::NATIVE_INT, space, create_matrix_props(dims[0], dims[1]));
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vector_field_rm = vector_field;
    dset.write(vector_field_rm.data(), H5::PredType::NATIVE_INT);
}

Eigen::MatrixXd read_hdf5_vector_field(const std::string& path,
                                   const std::string& name)
{
    H5::H5File file(path, H5F_ACC_RDONLY);

    H5::DataSet dset = file.openDataSet("attributes/" + name);
    hsize_t dims[2];
    dset.getSpace().getSimpleExtentDims(dims);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vector_field_rm(dims[0], dims[1]);
    dset.read(vector_field_rm.data(), H5::PredType::NATIVE_DOUBLE);

    return vector_field_rm;
}

Eigen::MatrixXi read_hdf5_integer_matrix(const std::string& path,
                                   const std::string& name)
{
    H5::H5File file(path, H5F_ACC_RDONLY);

    H5::DataSet dset = file.openDataSet("attributes/" + name);
    hsize_t dims[2];
    dset.getSpace().getSimpleExtentDims(dims);

    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vector_field_rm(dims[0], dims[1]);
    dset.read(vector_field_rm.data(), H5::PredType::NATIVE_INT);

    return vector_field_rm;
}


void read_hdf5_mesh(const std::string& path,
               Eigen::MatrixXd& V,
               Eigen::MatrixXi& F)
{
    H5::H5File file(path, H5F_ACC_RDONLY);

    // Vertices
    H5::DataSet v_set = file.openDataSet("vertices");
    H5::DataSpace v_space = v_set.getSpace();
    hsize_t v_dims[2];
    v_space.getSimpleExtentDims(v_dims);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V_row_major(v_dims[0], v_dims[1]);
    v_set.read(V_row_major.data(), H5::PredType::NATIVE_DOUBLE);
    V = V_row_major;

    // Faces
    H5::DataSet f_set = file.openDataSet("faces");
    H5::DataSpace f_space = f_set.getSpace();
    hsize_t f_dims[2];
    f_space.getSimpleExtentDims(f_dims);

    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F_row_major(f_dims[0], f_dims[1]);
    f_set.read(F_row_major.data(), H5::PredType::NATIVE_INT);
    F = F_row_major;
}

void read_hdf5_mesh_with_uv(const std::string& path,
               Eigen::MatrixXd& V,
               Eigen::MatrixXi& F,
               Eigen::MatrixXd& uv,
               Eigen::MatrixXi& FT)
{
    H5::H5File file(path, H5F_ACC_RDONLY);

    // Vertices
    H5::DataSet v_set = file.openDataSet("vertices");
    H5::DataSpace v_space = v_set.getSpace();
    hsize_t v_dims[2];
    v_space.getSimpleExtentDims(v_dims);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V_row_major(v_dims[0], v_dims[1]);
    v_set.read(V_row_major.data(), H5::PredType::NATIVE_DOUBLE);
    V = V_row_major;

    // Faces
    H5::DataSet f_set = file.openDataSet("faces");
    H5::DataSpace f_space = f_set.getSpace();
    hsize_t f_dims[2];
    f_space.getSimpleExtentDims(f_dims);

    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F_row_major(f_dims[0], f_dims[1]);
    f_set.read(F_row_major.data(), H5::PredType::NATIVE_INT);
    F = F_row_major;

    // UV vertices
    H5::DataSet uv_set = file.openDataSet("uv_vertices");
    H5::DataSpace uv_space = uv_set.getSpace();
    hsize_t uv_dims[2];
    uv_space.getSimpleExtentDims(uv_dims);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> uv_row_major(uv_dims[0], uv_dims[1]);
    uv_set.read(uv_row_major.data(), H5::PredType::NATIVE_DOUBLE);
    uv = uv_row_major;

    // UV faces
    H5::DataSet ft_set = file.openDataSet("uv_faces");
    H5::DataSpace ft_space = ft_set.getSpace();
    hsize_t ft_dims[2];
    ft_space.getSimpleExtentDims(ft_dims);

    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FT_row_major(ft_dims[0], ft_dims[1]);
    ft_set.read(FT_row_major.data(), H5::PredType::NATIVE_INT);
    FT = FT_row_major;
}

void write_sparse_matrix(const MatrixX& matrix, const std::string& filename, std::string format)
{
    /// Open file
    std::ofstream output_file;
    output_file.open(filename);

    // Iterate over nonzero entries
    for (Eigen::Index k = 0; k < matrix.outerSize(); ++k) {
        for (MatrixX::InnerIterator it(matrix, k); it; ++it) {
            int i = it.row();
            int j = it.col();
            Scalar v = it.value();

            // CSV format has comma separated 0-indexed values
            if (format == "csv") {
                output_file << std::fixed << std::setprecision(17) << i << "," << j << "," << v
                            << std::endl;
            }
            // MATLAB uses space separated 1-indexed values
            else if (format == "matlab") {
                output_file << std::fixed << std::setprecision(17) << (i + 1) << "  " << (j + 1)
                            << "  " << v << std::endl;
            }
        }
    }

    // Close file
    output_file.close();
}

} // namespace Penner