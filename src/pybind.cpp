// This file is part of penner-optimization, a constrained parametrization library.
// 
// Copyright (C) 2026 Ryan Capouellez <rjcapouellez@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "optimization/pybind.h"
#include "holonomy/pybind.h"
#include "feature/pybind.h"

using namespace Penner;
using namespace Optimization;
using namespace Holonomy;
using namespace Feature;

#ifdef PYBIND
#ifndef MULTIPRECISION

// wrap as Python module
PYBIND11_MODULE(penner, m)
{
    m.doc() = "pybindings for penner methods";
    init_optimization_pybind(m);
    init_holonomy_pybind(m);
    init_feature_pybind(m);
}

#endif
#endif