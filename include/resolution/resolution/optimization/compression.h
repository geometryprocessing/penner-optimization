#include "resolution/core/common.h"

namespace Penner {
namespace Resolution {

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> build_grid_mesh(
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z);

std::tuple<Eigen::MatrixXd, std::vector<int>, std::vector<int>, std::vector<int>>
build_grid_halfedge(
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z);

std::tuple<GeometryData, GeometryData, GeometryData> read_geometry_image(
  const std::string& image_path,
  std::string filetype="png");


void write_geometry_image(
  const std::string& image_path,
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z,
  bool lossless = true,
  float quality = 90.0f,
  std::string filetype="png");

#if USE_JXL
std::tuple<GeometryData, GeometryData, GeometryData> read_geometry_image_jxl(
  const std::string& image_path);

// Write a geometry image as JPEG XL. By default the image is encoded
// losslessly; pass lossless=false to use lossy compression at the given
// quality (0..100, higher is better quality). If geometry_safe is true
// (the default), the encoder avoids color/perceptual assumptions that only
// hold for photographic RGB data (XYB/YCoCg color transforms, Gaborish,
// EPF, dots, and patches), since the channels here store independent x/y/z
// coordinates rather than correlated image color channels.
void write_geometry_image_jxl(
  const std::string& image_path,
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z,
  bool lossless = true,
  float quality = 90.0f,
  bool geometry_safe = true);
#endif

#if USE_EXR
std::tuple<GeometryData, GeometryData, GeometryData> read_geometry_image_exr(
  const std::string& image_path);

// Write a geometry image as OpenEXR. By default the image is encoded
// losslessly (ZIP compression); pass lossless=false to use lossy DWAA
// compression at the given quality (0..100, higher is better quality).
void write_geometry_image_exr(
  const std::string& image_path,
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z,
  bool lossless = true,
  float quality = 90.0f);
#endif

}
}
