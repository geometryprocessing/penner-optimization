// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "util/common.h"

namespace Penner {

// Parameters to pass to the conformal method for projecting to the constraint.
// More detail on these parameters can be found in the documentation for that
// method.
struct ProjectionParameters
{
    int max_itr = 100; // maximum number of iterations
    Scalar bound_norm_thres = 1e-10; // line search threshold for dropping the gradient norm bound
#ifdef MULTIPRECISION
    Scalar error_eps = 1e-24; // minimum error termination condition
#else
    Scalar error_eps = 1e-8; // minimum error termination condition
#endif
    bool do_reduction =
        true; // reduce the initial line step if the range of coordinate values is large
    bool initial_ptolemy = true; // initial_ptolemy: use ptolemy flips for the initial make_delaunay
    bool use_edge_flips = true; // use intrinsic edge flips
    std::string output_dir = "";
};

} // namespace Penner