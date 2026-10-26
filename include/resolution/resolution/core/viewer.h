
#pragma once
#include "resolution/core/common.h"

namespace Penner {
namespace Resolution {

/**
 * @brief View the uv coordinate layout of a parametrization induced by a metric.
 * 
 * @param marked_metric: metric that induces parametrization
 * @param show: if true, show the layout
 */
void view_layout(const MarkedPennerConeMetric& marked_metric, bool show=true);

/**
 * @brief View a combed frame field on a surface
 * 
 * @param V: mesh vertices
 * @param F: mesh faces
 * @param uv: uv coordinates
 * @param FT: uv face connectivity
 * @param PD1: first combed vector field (u gradients)
 * @param PD2: second combed vector field (v gradients)
 * @param mesh_handle: name for mesh in viewer
 */
void view_combed_field(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& FT,
    const Eigen::MatrixXd& PD1,
    const Eigen::MatrixXd& PD2,
    std::string mesh_handle="");

}
}