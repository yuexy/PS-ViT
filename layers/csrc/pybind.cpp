#include "cpp_helper.hpp"

std::string get_compiler_version();
std::string get_compiling_cuda_version();

void progressive_sampling_forward(Tensor input,
                                  Tensor point,
                                  Tensor offset,
                                  Tensor output,
                                  float gamma);

void progressive_sampling_backward(Tensor grad_output,
                                   Tensor input,
                                   Tensor point,
                                   Tensor offset,
                                   Tensor grad_input,
                                   Tensor grad_offset,
                                   float gamma);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("progressive_sampling_forward", &progressive_sampling_forward,
          "progressive sampling forward",
          py::arg("input"),
          py::arg("point"),
          py::arg("offset"),
          py::arg("output"),
          py::arg("gamma"));
    m.def("progressive_sampling_backward", &progressive_sampling_backward,
          "progressive sampling backward",
          py::arg("grad_output"),
          py::arg("input"),
          py::arg("point"),
          py::arg("offset"),
          py::arg("grad_input"),
          py::arg("grad_offset"),
          py::arg("gamma"));
}
