#include "cpp_helper.hpp"


void ProgressiveSamplingForwardCUDAKernelLauncher(Tensor input,
                                                  Tensor point,
                                                  Tensor offset,
                                                  Tensor output,
                                                  float gamma);

void ProgressiveSamplingBackwardCUDAKernelLauncher(Tensor grad_output,
                                                   Tensor input,
                                                   Tensor point,
                                                   Tensor offset,
                                                   Tensor grad_input,
                                                   Tensor grad_offset,
                                                   float gamma);


void progressive_sampling_forward(Tensor input,
                                  Tensor point,
                                  Tensor offset,
                                  Tensor output,
                                  float gamma)
{
    ProgressiveSamplingForwardCUDAKernelLauncher(input,
                                                 point,
                                                 offset,
                                                 output,
                                                 gamma);
}

void progressive_sampling_backward(Tensor grad_output,
                                   Tensor input,
                                   Tensor point,
                                   Tensor offset,
                                   Tensor grad_input,
                                   Tensor grad_offset,
                                   float gamma)
{
    ProgressiveSamplingBackwardCUDAKernelLauncher(grad_output,
                                                  input,
                                                  point,
                                                  offset,
                                                  grad_input,
                                                  grad_offset,
                                                  gamma);
}
