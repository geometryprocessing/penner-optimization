#include "resolution/pybind.h"
#include "resolution/core/quad_quality.h"
#include "resolution/optimization/improvement.h"

namespace Penner {
namespace Resolution {

#ifdef PYBIND
#ifndef MULTIPRECISION

void init_resolution_pybind(pybind11::module& m)
{
    spdlog::set_level(spdlog::level::info);
    pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>
        default_call_guard;

    m.def("compute_quad_lengths", &compute_quad_lengths, default_call_guard);
    m.def("compute_quad_angles", &compute_quad_angles, default_call_guard);
    m.def("compute_scaled_jacobian", &compute_scaled_jacobian, default_call_guard);

#ifdef USE_UV_OPTIMIZATION
    m.def("symmetric_dirichlet", &symmetric_dirichlet, default_call_guard);
#endif
}


#endif
#endif

} // namespace Resolution
} // namespace Penner
