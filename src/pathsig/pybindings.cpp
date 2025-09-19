// pybindings.cpp
#include <torch/extension.h>
#include "compute_sig_launch.cuh"
#include "sig_backprop_launch.cuh"
#include "SigDecomposition.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<pathsig::SigDecomposition>(m, "SigDecomposition")
        .def(py::init<int,int>(),
             py::arg("path_dim"), py::arg("trunc_level"));

    m.def("compute_sig", &pathsig::computeSignature,
          py::arg("path"),
          py::arg("truncation_level"),
          py::arg("extended_precision") = false,
          py::arg("sig_decomp") = py::none(), // None maps to nullptr
          py::call_guard<py::gil_scoped_release>());

    m.def("compute_sig_gradients", &pathsig::computeSigGradients,
          py::arg("path"),
          py::arg("signature"),
          py::arg("incoming_grads"),
          py::arg("truncation_level"),
          py::arg("sig_decomp") = py::none(),
          py::call_guard<py::gil_scoped_release>());

    m.def("sig_size",
          &pathsig::computeSigSize,
          py::arg("path_dim"), py::arg("truncation_level"),
          "Total number of signature terms excluding the level 0 identity term.");
}