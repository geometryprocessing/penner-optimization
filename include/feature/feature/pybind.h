#pragma once
#ifdef PYBIND
#ifndef MULTIPRECISION

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Penner {
namespace Feature {

void init_feature_pybind(pybind11::module& m);


} // namespace Holonomy
} // namespace Penner

#endif
#endif