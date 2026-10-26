#pragma once
#ifdef PYBIND
#ifndef MULTIPRECISION

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Penner {
namespace Resolution {

void init_resolution_pybind(pybind11::module& m);


} // namespace on
} // namespace Penner

#endif
#endif