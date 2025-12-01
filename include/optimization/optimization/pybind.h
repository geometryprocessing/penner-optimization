#pragma once
#ifdef PYBIND
#ifndef MULTIPRECISION

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Penner {
namespace Optimization {

void init_optimization_pybind(pybind11::module& m);

} // namespace Optimization
} // namespace Penner

#endif
#endif
