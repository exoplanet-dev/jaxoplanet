// cppimport
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ellip.h"

namespace py = pybind11;

PYBIND11_MODULE(limbdark, m) {
  m.def("cel_f64", py::vectorize(exoplanet::internal::ellip::cel<double>));
  m.def("cel_f32", py::vectorize(exoplanet::internal::ellip::cel<float>));
}

/*
<%
setup_pybind11(cfg)
%>
*/
