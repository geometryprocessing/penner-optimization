#include "resolution/optimization/compression.h"


#define IGL_STB_IMAGE_IMPLEMENTATION
#include <igl/png/readPNG.h>
#include <igl/png/writePNG.h>

#include <algorithm>
#include <fstream>
#include <vector>

#if USE_JXL
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/thread_parallel_runner_cxx.h>
#endif

#if USE_EXR
#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#endif

namespace Penner {
namespace Resolution {

#if USE_JXL
// Encode a buffer of interleaved 32-bit float RGB pixels as a JPEG XL file.
// If lossless is false, the image is compressed lossily at the given quality
// (0..100, higher is better quality, matching JxlEncoderDistanceFromQuality).
// If geometry_safe is true, the encoder avoids color/perceptual assumptions
// that are only valid for photographic RGB data (the XYB/YCoCg color
// transforms and psychovisual heuristics such as Gaborish, EPF, dots, and
// patches), since the R/G/B channels here actually store independent x/y/z
// geometry coordinates rather than correlated image color channels.
bool encode_jxl_rgb(
  const std::string& image_path,
  uint32_t width,
  uint32_t height,
  const std::vector<float>& pixels,
  bool lossless,
  float quality,
  bool geometry_safe)
{
  auto encoder = JxlEncoderMake(nullptr);
  auto runner = JxlThreadParallelRunnerMake(
    nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  if (JxlEncoderSetParallelRunner(
        encoder.get(), JxlThreadParallelRunner, runner.get()) != JXL_ENC_SUCCESS)
  {
    spdlog::error("Failed to set JPEG XL parallel runner");
    return false;
  }

  JxlBasicInfo basic_info;
  JxlEncoderInitBasicInfo(&basic_info);
  basic_info.xsize = width;
  basic_info.ysize = height;
  basic_info.bits_per_sample = 32;
  basic_info.exponent_bits_per_sample = 8;
  basic_info.num_color_channels = 3;
  basic_info.uses_original_profile = JXL_TRUE;
  if (JxlEncoderSetBasicInfo(encoder.get(), &basic_info) != JXL_ENC_SUCCESS)
  {
    spdlog::error("Failed to set JPEG XL basic info");
    return false;
  }

  JxlColorEncoding color_encoding = {};
  JxlColorEncodingSetToSRGB(&color_encoding, JXL_FALSE);
  color_encoding.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
  if (JxlEncoderSetColorEncoding(encoder.get(), &color_encoding) != JXL_ENC_SUCCESS)
  {
    spdlog::error("Failed to set JPEG XL color encoding");
    return false;
  }

  JxlEncoderFrameSettings* frame_settings =
    JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);
  JxlEncoderSetFrameLossless(frame_settings, lossless ? JXL_TRUE : JXL_FALSE);
  if (!lossless)
  {
    JxlEncoderSetFrameDistance(
      frame_settings, JxlEncoderDistanceFromQuality(quality));
  }
  if (geometry_safe)
  {
    // Force modular mode so VarDCT's blockwise DCT and XYB color transform
    // are never used, and disable the reversible color transform modular
    // mode would otherwise try per group; both assume correlated RGB color
    // channels, which does not hold for x/y/z coordinates.
    JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_MODULAR, 1);
    JxlEncoderFrameSettingsSetOption(
      frame_settings, JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE, 0);
    JxlEncoderFrameSettingsSetOption(
      frame_settings, JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, 1);
    // Disable psychovisual heuristics and image-specific filters that trade
    // numerical accuracy for perceived photographic quality.
    JxlEncoderFrameSettingsSetOption(
      frame_settings, JXL_ENC_FRAME_SETTING_DISABLE_PERCEPTUAL_HEURISTICS, 1);
    JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_EPF, 0);
    JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_GABORISH, 0);
    JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_DOTS, 0);
    JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_PATCHES, 0);
  }

  JxlPixelFormat pixel_format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  if (JxlEncoderAddImageFrame(
        frame_settings,
        &pixel_format,
        pixels.data(),
        pixels.size() * sizeof(float)) != JXL_ENC_SUCCESS)
  {
    spdlog::error("Failed to add JPEG XL image frame");
    return false;
  }
  JxlEncoderCloseInput(encoder.get());

  std::vector<uint8_t> compressed(1 << 20);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size();
  JxlEncoderStatus status = JXL_ENC_NEED_MORE_OUTPUT;
  while (status == JXL_ENC_NEED_MORE_OUTPUT)
  {
    status = JxlEncoderProcessOutput(encoder.get(), &next_out, &avail_out);
    if (status == JXL_ENC_NEED_MORE_OUTPUT)
    {
      size_t offset = next_out - compressed.data();
      compressed.resize(compressed.size() * 2);
      next_out = compressed.data() + offset;
      avail_out = compressed.size() - offset;
    }
  }
  compressed.resize(next_out - compressed.data());
  if (status != JXL_ENC_SUCCESS)
  {
    spdlog::error("Failed to encode JPEG XL image");
    return false;
  }

  std::ofstream out_file(image_path, std::ios::binary);
  if (!out_file)
  {
    spdlog::error("Failed to open {} for writing", image_path);
    return false;
  }
  out_file.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());

  return true;
}

// Decode a lossless JPEG XL file into a buffer of interleaved 32-bit float RGB pixels.
bool decode_jxl_rgb(
  const std::string& image_path,
  uint32_t& width,
  uint32_t& height,
  std::vector<float>& pixels)
{
  std::ifstream in_file(image_path, std::ios::binary);
  if (!in_file)
  {
    spdlog::error("Failed to open {} for reading", image_path);
    return false;
  }
  std::vector<uint8_t> compressed(
    (std::istreambuf_iterator<char>(in_file)), std::istreambuf_iterator<char>());

  auto decoder = JxlDecoderMake(nullptr);
  auto runner = JxlThreadParallelRunnerMake(
    nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  if (JxlDecoderSetParallelRunner(
        decoder.get(), JxlThreadParallelRunner, runner.get()) != JXL_DEC_SUCCESS)
  {
    spdlog::error("Failed to set JPEG XL parallel runner");
    return false;
  }

  if (JxlDecoderSubscribeEvents(
        decoder.get(), JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) != JXL_DEC_SUCCESS)
  {
    spdlog::error("Failed to subscribe to JPEG XL decoder events");
    return false;
  }

  if (JxlDecoderSetInput(decoder.get(), compressed.data(), compressed.size()) !=
      JXL_DEC_SUCCESS)
  {
    spdlog::error("Failed to set JPEG XL decoder input");
    return false;
  }
  JxlDecoderCloseInput(decoder.get());

  JxlPixelFormat pixel_format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  for (;;)
  {
    JxlDecoderStatus status = JxlDecoderProcessInput(decoder.get());
    if (status == JXL_DEC_ERROR)
    {
      spdlog::error("Failed to decode JPEG XL image");
      return false;
    }
    else if (status == JXL_DEC_BASIC_INFO)
    {
      JxlBasicInfo basic_info;
      if (JxlDecoderGetBasicInfo(decoder.get(), &basic_info) != JXL_DEC_SUCCESS)
      {
        spdlog::error("Failed to get JPEG XL basic info");
        return false;
      }
      width = basic_info.xsize;
      height = basic_info.ysize;
      pixels.resize(static_cast<size_t>(width) * height * 3);
    }
    else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER)
    {
      size_t buffer_size;
      if (JxlDecoderImageOutBufferSize(
            decoder.get(), &pixel_format, &buffer_size) != JXL_DEC_SUCCESS)
      {
        spdlog::error("Failed to get JPEG XL output buffer size");
        return false;
      }
      if (buffer_size != pixels.size() * sizeof(float))
      {
        spdlog::error("Unexpected JPEG XL output buffer size");
        return false;
      }
      if (JxlDecoderSetImageOutBuffer(
            decoder.get(), &pixel_format, pixels.data(), buffer_size) != JXL_DEC_SUCCESS)
      {
        spdlog::error("Failed to set JPEG XL output buffer");
        return false;
      }
    }
    else if (status == JXL_DEC_FULL_IMAGE)
    {
      // Pixels buffer is fully populated; keep looping until JXL_DEC_SUCCESS.
    }
    else if (status == JXL_DEC_SUCCESS)
    {
      break;
    }
  }

  return true;
}

void write_geometry_image_jxl(
  const std::string& image_path,
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z,
  bool lossless,
  float quality,
  bool geometry_safe)
{
    int width = x.cols();
    int height = x.rows();
    std::vector<float> pixels(static_cast<size_t>(width) * height * 3);
    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
        size_t p = (static_cast<size_t>(i) * width + j) * 3;
        pixels[p + 0] = static_cast<float>(x(i, j));
        pixels[p + 1] = static_cast<float>(y(i, j));
        pixels[p + 2] = static_cast<float>(z(i, j));
      }
    }

    encode_jxl_rgb(image_path, width, height, pixels, lossless, quality, geometry_safe);
}

std::tuple<GeometryData, GeometryData, GeometryData> read_geometry_image_jxl(
  const std::string& image_path)
{
    uint32_t width, height;
    std::vector<float> pixels;
    if (!decode_jxl_rgb(image_path, width, height, pixels))
    {
      return std::make_tuple(GeometryData(), GeometryData(), GeometryData());
    }

    GeometryData X(height, width), Y(height, width), Z(height, width);
    for (uint32_t i = 0; i < height; ++i)
    {
      for (uint32_t j = 0; j < width; ++j)
      {
        size_t p = (static_cast<size_t>(i) * width + j) * 3;
        X(i, j) = pixels[p + 0];
        Y(i, j) = pixels[p + 1];
        Z(i, j) = pixels[p + 2];
      }
    }

    return std::make_tuple(X, Y, Z);
}
#endif

#if USE_EXR
// Encode a buffer of interleaved 32-bit float RGB pixels as an OpenEXR file.
// If lossless is false, the image is compressed lossily with DWAA at the
// given quality (0..100, higher is better quality).
bool encode_exr_rgb(
  const std::string& image_path,
  uint32_t width,
  uint32_t height,
  const std::vector<float>& pixels,
  bool lossless,
  float quality)
{
  Imf::Header header(
    static_cast<int>(width),
    static_cast<int>(height),
    1.0f,
    Imath::V2f(0, 0),
    1.0f,
    Imf::INCREASING_Y,
    lossless ? Imf::ZIP_COMPRESSION : Imf::DWAA_COMPRESSION);
  if (!lossless)
  {
    // Higher quality should mean lower DWA compression level (less loss).
    header.dwaCompressionLevel() = (100.0f - std::clamp(quality, 0.0f, 100.0f));
  }
  header.channels().insert("R", Imf::Channel(Imf::FLOAT));
  header.channels().insert("G", Imf::Channel(Imf::FLOAT));
  header.channels().insert("B", Imf::Channel(Imf::FLOAT));

  try
  {
    Imf::OutputFile file(image_path.c_str(), header);

    size_t x_stride = 3 * sizeof(float);
    size_t y_stride = x_stride * width;
    Imf::FrameBuffer frame_buffer;
    frame_buffer.insert(
      "R",
      Imf::Slice(
        Imf::FLOAT,
        reinterpret_cast<char*>(const_cast<float*>(pixels.data()) + 0),
        x_stride,
        y_stride));
    frame_buffer.insert(
      "G",
      Imf::Slice(
        Imf::FLOAT,
        reinterpret_cast<char*>(const_cast<float*>(pixels.data()) + 1),
        x_stride,
        y_stride));
    frame_buffer.insert(
      "B",
      Imf::Slice(
        Imf::FLOAT,
        reinterpret_cast<char*>(const_cast<float*>(pixels.data()) + 2),
        x_stride,
        y_stride));
    file.setFrameBuffer(frame_buffer);
    file.writePixels(static_cast<int>(height));
  }
  catch (const std::exception& e)
  {
    spdlog::error("Failed to encode OpenEXR image: {}", e.what());
    return false;
  }

  return true;
}

// Decode an OpenEXR file into a buffer of interleaved 32-bit float RGB pixels.
bool decode_exr_rgb(
  const std::string& image_path,
  uint32_t& width,
  uint32_t& height,
  std::vector<float>& pixels)
{
  try
  {
    Imf::InputFile file(image_path.c_str());
    Imath::Box2i data_window = file.header().dataWindow();
    width = static_cast<uint32_t>(data_window.max.x - data_window.min.x + 1);
    height = static_cast<uint32_t>(data_window.max.y - data_window.min.y + 1);
    pixels.assign(static_cast<size_t>(width) * height * 3, 0.0f);

    size_t x_stride = 3 * sizeof(float);
    size_t y_stride = x_stride * width;
    char* base = reinterpret_cast<char*>(pixels.data()) -
      (data_window.min.x * static_cast<long>(x_stride)) -
      (data_window.min.y * static_cast<long>(y_stride));

    Imf::FrameBuffer frame_buffer;
    frame_buffer.insert(
      "R", Imf::Slice(Imf::FLOAT, base + 0 * sizeof(float), x_stride, y_stride));
    frame_buffer.insert(
      "G", Imf::Slice(Imf::FLOAT, base + 1 * sizeof(float), x_stride, y_stride));
    frame_buffer.insert(
      "B", Imf::Slice(Imf::FLOAT, base + 2 * sizeof(float), x_stride, y_stride));
    file.setFrameBuffer(frame_buffer);
    file.readPixels(data_window.min.y, data_window.max.y);
  }
  catch (const std::exception& e)
  {
    spdlog::error("Failed to decode OpenEXR image: {}", e.what());
    return false;
  }

  return true;
}

void write_geometry_image_exr(
  const std::string& image_path,
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z,
  bool lossless,
  float quality)
{
    int width = x.cols();
    int height = x.rows();
    std::vector<float> pixels(static_cast<size_t>(width) * height * 3);
    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
        size_t p = (static_cast<size_t>(i) * width + j) * 3;
        pixels[p + 0] = static_cast<float>(x(i, j));
        pixels[p + 1] = static_cast<float>(y(i, j));
        pixels[p + 2] = static_cast<float>(z(i, j));
      }
    }

    encode_exr_rgb(image_path, width, height, pixels, lossless, quality);
}

std::tuple<GeometryData, GeometryData, GeometryData> read_geometry_image_exr(
  const std::string& image_path)
{
    uint32_t width, height;
    std::vector<float> pixels;
    if (!decode_exr_rgb(image_path, width, height, pixels))
    {
      return std::make_tuple(GeometryData(), GeometryData(), GeometryData());
    }

    GeometryData X(height, width), Y(height, width), Z(height, width);
    for (uint32_t i = 0; i < height; ++i)
    {
      for (uint32_t j = 0; j < width; ++j)
      {
        size_t p = (static_cast<size_t>(i) * width + j) * 3;
        X(i, j) = pixels[p + 0];
        Y(i, j) = pixels[p + 1];
        Z(i, j) = pixels[p + 2];
      }
    }

    return std::make_tuple(X, Y, Z);
}
#endif

std::tuple<GeometryData, GeometryData, GeometryData> read_geometry_image_txt(
  const std::string& image_path)
{
    // Open file
    std::ifstream input_file(image_path);
    if (!input_file) return {};

    // Read file
    std::vector<std::vector<double>> matrix_vec = {};
    std::string line;
    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        matrix_vec.push_back({});
        double v;
        while (iss >> v)
        {
          matrix_vec.back().push_back(v);
        }
    }

    // Close file
    input_file.close();

    int height = matrix_vec.size() / 3;
    int width = matrix_vec[0].size();
    GeometryData X(height, width);
    GeometryData Y(height, width);
    GeometryData Z(height, width);
    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
        X(i, j) = matrix_vec[i][j];
        Y(i, j) = matrix_vec[height + i][j];
        Z(i, j) = matrix_vec[2 * height + i][j];
      }
    }

    return std::make_tuple(X, Y, Z);
}

void write_geometry_channel_txt(
  std::ofstream& output_file,
  const GeometryData& x)
{
    int width = x.cols();
    int height = x.rows();

    // Iterate over rows
    for (Eigen::Index i = 0; i < height; ++i) {
        // Iterate over columns of row i
        double v = x(i, 0);
        output_file << std::fixed << std::setprecision(17) << v;
        for (Eigen::Index j = 1; j < width; ++j) {
            double v = x(i, j);
            output_file << std::fixed << std::setprecision(17) << " " << v;
        }

        // Add newline to end of row
        output_file << std::endl;
    }
}

void write_geometry_image_txt(
  const std::string& image_path,
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z)
{
    // Open file
    std::ofstream output_file;
    output_file.open(image_path);
    write_geometry_channel_txt(output_file, x);
    write_geometry_channel_txt(output_file, y);
    write_geometry_channel_txt(output_file, z);
    output_file.close();
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> build_grid_mesh(
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z)
{
  int width = x.cols();
  int height = x.rows();
  int num_vertices = width * height;
  int num_quads = (width - 1) * (height - 1);

  // build flattened vertex data from xyz data
  Eigen::MatrixXd V(num_vertices, 3);
  for (int i = 0; i < height; ++i)
  {
    for (int j = 0; j < width; ++j)
    {
      int v = (i * width) + j;
      V(v, 0) = x(i, j);
      V(v, 1) = y(i, j);
      V(v, 2) = z(i, j);
    }
  }

  // build faces of grid
  Eigen::MatrixXi F(num_quads, 4);
  for (int i = 0; i < height - 1; ++i)
  {
    for (int j = 0; j < width - 1; ++j)
    {
      int q = (i * (width - 1)) + j;
      int ip = i + 1;
      int jp = j + 1;
      int v00 = (i * width) + j;
      int v01 = (i * width) + jp;
      int v10 = (ip * width) + j;
      int v11 = (ip * width) + jp;
      F.row(q) << v00, v01, v11, v10;
    }
  }

  return std::make_tuple(V, F);
}

std::tuple<Eigen::MatrixXd, std::vector<int>, std::vector<int>, std::vector<int>>
build_grid_halfedge(
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z)
{
  int width = x.cols();
  int height = x.rows();
  int num_vertices = width * height;
  int num_quads = (width - 1) * (height - 1);
  int num_halfedges = 4 * num_quads;

  // build flattened vertex data from xyz data
  Eigen::MatrixXd V(num_vertices, 3);
  for (int i = 0; i < height; ++i)
  {
    for (int j = 0; j < width; ++j)
    {
      int v = (i * width) + j;
      V(v, 0) = x(i, j);
      V(v, 1) = y(i, j);
      V(v, 2) = z(i, j);
    }
  }

  // build faces of grid
  std::vector<int> n(num_halfedges, -1);
  std::vector<int> to(num_halfedges, -1);
  for (int i = 0; i < height - 1; ++i)
  {
    for (int j = 0; j < width - 1; ++j)
    {
      int q = (i * (width - 1)) + j;
      int h = 4 * q;
      int ip = i + 1;
      int jp = j + 1;

      // build next adjacency
      for (int k = 0; k < 4; ++k)
      {
        int kp = (k + 1) % 4;
        n[h + k] = h + kp;
      }

      // set halfedge tip vertices
      to[h + 0] = (i * width) + jp; // bottom
      to[h + 1] = (ip * width) + jp; // right
      to[h + 2] = (ip * width) + j; // top
      to[h + 3] = (i * width) + j; // left
    }
  }

  // set horizontal opposites
  std::vector<int> opp(num_halfedges, -1);
  for (int i = 0; i < height - 1; ++i)
  {
    for (int j = 1; j < width - 1; ++j)
    {
      int q = (i * (width - 1)) + j;
      int qp = (i * (width - 1)) + (j - 1);
      int h = 4 * q + 3;
      int ho = 4 * qp + 1;
      opp[h] = ho;
      opp[ho] = h;
    }
  }

  // set vertical opposites
  for (int i = 1; i < height - 1; ++i)
  {
    for (int j = 0; j < width - 1; ++j)
    {
      int q = (i * (width - 1)) + j;
      int qp = ((i - 1) * (width - 1)) + j;
      int h = 4 * q + 0;
      int ho = 4 * qp + 2;
      opp[h] = ho;
      opp[ho] = h;
    }
  }

  return std::make_tuple(V, n, opp, to);
}

ImageData convert_geometry_to_image_data(const GeometryData& X)
{
  return (256 * X).cast<unsigned char>();
}

GeometryData convert_image_to_geometry_data(const ImageData& X)
{
  return (X / 256).cast<Scalar>();
}

std::tuple<GeometryData, GeometryData, GeometryData> read_geometry_image_png(
  const std::string& image_path)
{
    ImageData x, y, z, alpha;
    igl::png::readPNG(image_path, x, y, z, alpha);

    GeometryData X, Y, Z;
    X = convert_image_to_geometry_data(x);
    Y = convert_image_to_geometry_data(y);
    Z = convert_image_to_geometry_data(z);

    return std::make_tuple(X, Y, Z);
}

void write_geometry_image_png(
  const std::string& image_path,
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z)
{
    ImageData X, Y, Z;
    X = convert_geometry_to_image_data(x);
    Y = convert_geometry_to_image_data(y);
    Z = convert_geometry_to_image_data(z);
    ImageData A = ImageData::Constant(X.rows(), X.cols(), 255);
    igl::png::writePNG(X, Y, Z, A, image_path);
}

std::tuple<GeometryData, GeometryData, GeometryData> read_geometry_image(
  const std::string& image_path,
  std::string filetype)
{
  if (filetype == "txt")
  {
    return read_geometry_image_txt(image_path);
  }
#if USE_JXL
  else if (filetype == "jxl")
  {
    return read_geometry_image_jxl(image_path);
  }
#endif
#if USE_EXR
  else if (filetype == "exr")
  {
    return read_geometry_image_exr(image_path);
  }
#endif
  else
  {
    return read_geometry_image_png(image_path);
  }
}

void write_geometry_image(
  const std::string& image_path,
  const GeometryData& x,
  const GeometryData& y,
  const GeometryData& z,
  bool lossless,
  float quality,
  std::string filetype)
{
  if (filetype == "txt")
  {
    write_geometry_image_txt(image_path, x, y, z);
  }
#if USE_JXL
  else if (filetype == "jxl")
  {
    write_geometry_image_jxl(image_path, x, y, z, lossless, quality, true);
  }
#endif
#if USE_EXR
  else if (filetype == "exr")
  {
    write_geometry_image_exr(image_path, x, y, z, lossless, quality);
  }
#endif
  else
  {
    write_geometry_image_png(image_path, x, y, z);
  }
}


}
}
