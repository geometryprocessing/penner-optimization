#include "holonomy/field/cross_field.h"

#include <directional/TriMesh.h>
#include <directional/IntrinsicFaceTangentBundle.h>
#include <directional/CartesianField.h>
#include <directional/polycurl_reduction.h>
#include <directional/combing.h>
#include <directional/curl_matching.h>
#include <directional/directional_viewer.h>

#include <igl/rotate_vectors.h>

namespace Penner {
namespace Holonomy {

std::array<Eigen::MatrixXd, 4> load_rawfield(const std::string& filename)
{
    // Open file
    std::ifstream input_file(filename);
    if (!input_file) return {};

    // get number of faces
    std::string line;
    std::getline(input_file, line);
    std::istringstream iss(line);
    int n, num_faces;
    iss >> n >> num_faces;
    if (n != 4)
    {
        spdlog::error("Can only read cross fields");
    }

    // initialize vectors
    std::array<Eigen::MatrixXd, 4> cross_field;
    for (int i = 0; i < 4; ++i)
    {
        cross_field[i].resize(num_faces, 3);
    }

    // Read file one face at a time
    int f = 0;
    while ((f < num_faces) && (std::getline(input_file, line))) {
        std::istringstream iss(line);
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                iss >> cross_field[i](f, j);
            }
        }

        // increment face
        ++f;
    }

    // check all faces written
    if (num_faces != f)
    {
        spdlog::error("Number of faces inconsistent with number of lines");
    }

    // Close file
    input_file.close();

    return cross_field;
}

void write_cross_field(
    const std::string& output_filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta)
{
    std::array<Eigen::MatrixXd, 4> cross_field = generate_cross_field(V, F, reference_field, theta);

    std::ofstream field_file(output_filename, std::ios::out | std::ios::trunc);
    field_file << "4 " << F.rows() << std::endl;
    for (int f = 0; f < F.rows(); ++f)
    {
        // write all directions
        for (int i : {0 , 1, 2, 3})
        {
            for (int j : {0 , 1, 2})
            {
                field_file << std::fixed << std::setprecision(17) << cross_field[i](f, j) << " ";
            }
        }

        field_file << std::endl;
    }

    field_file.close();
}

std::array<Eigen::MatrixXd, 4> generate_cross_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& reference_field,
    const Eigen::VectorXd& theta)
{
    Eigen::MatrixXd B1, B2, B3;
    igl::local_basis(V, F, B1, B2, B3);
    Eigen::VectorXd right_angle = Eigen::VectorXd::Constant(F.rows(), M_PI / 2.);

    std::array<Eigen::MatrixXd, 4> cross_field;
    cross_field[0] = igl::rotate_vectors(reference_field, theta, B1, B2);
    for (int i : {1, 2, 3}) {
        cross_field[i] = igl::rotate_vectors(cross_field[i-1], right_angle, B1, B2);
    }

    return cross_field;
}

std::array<Eigen::MatrixXd, 4> reduce_curl(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::array<Eigen::MatrixXd, 4>& cross_field,
    const std::vector<int>& fixed_faces)
{
    spdlog::debug("Initializing mesh with face tangent bundle");
    directional::TriMesh mesh;
    directional::IntrinsicFaceTangentBundle tb;
    mesh.set_mesh(V,F);
    tb.init(mesh);

    int N = 4;
    int num_faces = F.rows();
    Eigen::MatrixXd extField(num_faces, 3*N);
    spdlog::debug("Copying cross field to raw field with {} faces", num_faces);
    for (int f = 0; f < num_faces; f++)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                extField(f, 3 * i + j) = cross_field[i](f, j);
            }
        }
        spdlog::trace("row {}: {}", f, extField.row(f));
    }

    assert(tb.sources.rows()==extField.rows());
    assert(tb.hasEmbedding() && "This tangent bundle doesn't admit an extrinsic embedding");
    directional::CartesianField rawFieldInit, rawField, combedFieldCF;
    rawFieldInit.init(tb, directional::fieldTypeEnum::RAW_FIELD, N);
    rawFieldInit.set_extrinsic_field(extField);
    rawField = rawFieldInit;

    // get initial curl
    Eigen::VectorXd curlOrig, curlCF; // norm of curl per edge
    directional::curl_matching(rawField,curlOrig);
    double curlMax = curlOrig.maxCoeff();
    spdlog::info("curl norm original: {} ", curlOrig.norm());
    spdlog::info("curl max original: {} ", curlMax);

    directional::PolyCurlReductionSolverData pcrdata;
    directional::polycurl_reduction_parameters params;
    int num_fixed_faces = fixed_faces.size();
    //num_fixed_faces = 1;
    Eigen::VectorXi b(num_fixed_faces);
    Eigen::MatrixXd bc(num_fixed_faces, 6);
    spdlog::debug("Initializing {} fixed faces", num_fixed_faces);
    for (int f = 0; f < num_fixed_faces; ++f)
    {
        b[f] = fixed_faces[f];
        spdlog::trace("face {} fixed", b[f]);
        bc.row(f) = rawField.extField.row(fixed_faces[f]).head(6);
    }
    Eigen::VectorXi blevel = Eigen::VectorXi::Constant(num_fixed_faces, 2);
    directional::polycurl_reduction_precompute(mesh, b, bc, blevel, rawField , pcrdata);
    Eigen::VectorXi constFaces = b;

    directional::DirectionalViewer viewer;
    bool view = false;
    if (view)
    {
        viewer.set_mesh(mesh,0);
        viewer.set_mesh(mesh, 1);
        viewer.set_field(rawField,Eigen::MatrixXd(), 1,0.9, 0, 10.0);

        viewer.toggle_mesh(false,0);
        viewer.toggle_field(true,0);
        viewer.toggle_field(true,1);
        viewer.set_selected_faces(constFaces,1);
        viewer.launch();
    }

    spdlog::debug("Optimizing curl");
    int iter = 0;
    params.numIter = 5;
    //params.wCurl = 1e2;
    //params.wQuotCurl = 1e2;
    //params.wBarrier = 0;
    //params.sBarrier = 0;
    //params.wSmooth = 0;
    //params.wCloseUnconstrained = 0;
    for (int bi = 0; bi < 3; ++bi)
    {
        
        spdlog::debug("Batch {}", iter);
        directional::polycurl_reduction_solve(pcrdata, params, rawField, iter ==0);
        iter++;
        params.wSmooth *= params.redFactor_wsmooth;

        directional::combing(rawField, combedFieldCF);
        directional::curl_matching(combedFieldCF,curlCF);
        curlMax= curlCF.maxCoeff();
        std:: cout<<"curl norm optimized: "<<curlCF.norm()<<std::endl;
        std:: cout<<"curlMax for batch: "<<curlMax<<std::endl;
    }

    // set back hard constraints
    for (int f = 0; f < num_fixed_faces; ++f)
    {
        rawField.extField.row(fixed_faces[f]) = rawFieldInit.extField.row(fixed_faces[f]);
    }
    
    Eigen::VectorXi prinIndices;
    directional::curl_matching(rawField, curlCF);
    curlMax= curlCF.maxCoeff();
    std:: cout<<"curlMax optimized before combing: "<<curlMax<<std::endl;

    directional::combing(rawField, combedFieldCF);

    std::array<Eigen::MatrixXd, 4> opt_cross_field;
    for (int i = 0; i < 4; i++)
    {
        opt_cross_field[i].resize(num_faces, 3);
    }
    for (int f = 0; f < num_faces; f++)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                opt_cross_field[i](f, j) = combedFieldCF.extField(f, 3 * i + j);
            }
        }
    }

    return opt_cross_field;
}

} // namespace Holonomy
} // namespace Penner