// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#if USE_UV_OPTIMIZATION 

#include "ExtremeOpt.h"
#include "MeshCutter.h"
#include "main_helper.h"


/// @file Methods to optimize a parametrization in terms of uv coordinates, reducing distortion
/// and also aligning the gradients with a guiding cross-field, while preserving seamless constraints
/// 
/// Also supports preservation of feature alignment constraints and a term to improve alignment of
/// misaligned feature edges.

namespace Penner {
namespace Optimization {

SymDir::Parameters load_parameters(const std::string& filepath);

SymDir::Parameters read_parameters(const nlohmann::json& config);

Eigen::MatrixXd optimize_seamless_parameterization(
    const Eigen::MatrixXd& V_init,
    const Eigen::MatrixXi& F_init,
    const Eigen::MatrixXd& uv_init,
    const Eigen::MatrixXi& FT_init,
    const Eigen::MatrixXd& Du,
    const Eigen::MatrixXd& Dv,
    SymDir::Parameters param=SymDir::Parameters());

Eigen::MatrixXd optimize_aligned_parameterization(
    const Eigen::MatrixXd& V_init,
    const Eigen::MatrixXi& F_init,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& FE_init,
    const Eigen::MatrixXi& ME,
    const Eigen::MatrixXd& Du,
    const Eigen::MatrixXd& Dv,
    SymDir::Parameters param=SymDir::Parameters());

} // namespace Optimization
} // namespace Penner

#endif