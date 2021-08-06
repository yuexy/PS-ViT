#include "progressive_sampling_cuda_kernel.cuh"
#include "cuda_helper.hpp"


void ProgressiveSamplingForwardCUDAKernelLauncher(Tensor input,
                                                  Tensor point,
                                                  Tensor offset,
                                                  Tensor output,
                                                  float gamma)
{
    int output_size = output.numel();
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int point_num = point.size(1);

    at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "progressive_sampling_forward_cuda_kernel", [&] {
            progressive_sampling_forward_cuda_kernel<scalar_t>
                <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                    output_size,
                    input.data_ptr<scalar_t>(),
                    point.data_ptr<scalar_t>(),
                    offset.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    channels,
                    point_num,
                    height,
                    width,
                    gamma
                );
        }
    );

    AT_CUDA_CHECK(cudaGetLastError());
}

void ProgressiveSamplingBackwardCUDAKernelLauncher(Tensor grad_output,
                                                   Tensor input,
                                                   Tensor point,
                                                   Tensor offset,
                                                   Tensor grad_input,
                                                   Tensor grad_offset,
                                                   float gamma)
{
    int output_size = grad_output.numel();
    int channels = grad_input.size(1);
    int height = grad_input.size(2);
    int width = grad_input.size(3);
    int point_num = grad_offset.size(1);

    at::cuda::CUDAGuard device_guard(grad_output.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.scalar_type(), "progressive_sampling_backward_cuda_kernel", [&] {
            progressive_sampling_backward_cuda_kernel<scalar_t>
                <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                    output_size,
                    grad_output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    point.data_ptr<scalar_t>(),
                    offset.data_ptr<scalar_t>(),
                    grad_input.data_ptr<scalar_t>(),
                    grad_offset.data_ptr<scalar_t>(),
                    channels,
                    point_num,
                    height,
                    width,
                    gamma
                );
        }
    );

    AT_CUDA_CHECK(cudaGetLastError());
}
