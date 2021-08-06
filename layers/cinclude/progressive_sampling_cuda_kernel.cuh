#ifndef PROGRESSIVE_SAMPLING_CUDA_KERNEL
#define PROGRESSIVE_SAMPLING_CUDA_KERNEL

#include "cuda_helper.hpp"


template <typename T>
__global__ void progressive_sampling_forward_cuda_kernel(const int nthreads,
                                                         const T* input,
                                                         const T* point,
                                                         const T* offset,
                                                         T* output,
                                                         const int channels,
                                                         const int point_num,
                                                         const int height,
                                                         const int width,
                                                         const T gamma)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int c = index % channels;
        int p = (index / channels) % point_num;
        int n = index / channels / point_num;

        const T* current_point = point + (n * point_num + p) * 2;
        const T* current_offset = offset + (n * point_num + p) * 2;
        const T* current_input = input + (n * channels + c) * height * width;

        const T y = current_point[0] + current_offset[0] * gamma;
        const T x = current_point[1] + current_offset[1] * gamma;

        output[index] = bilinear_interpolate(current_input, height, width, y, x);
    }
}


template <typename T>
__global__ void progressive_sampling_backward_cuda_kernel(const int nthreads,
                                                          const T* grad_output,
                                                          const T* input,
                                                          const T* point,
                                                          const T* offset,
                                                          T* grad_input,
                                                          T* grad_offset,
                                                          int channels,
                                                          int point_num,
                                                          int height,
                                                          int width,
                                                          const T gamma)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int c = index % channels;
        int p = (index / channels) % point_num;
        int n = index / channels / point_num;

        const T* current_point = point + (n * point_num + p) * 2;
        const T* current_offset = offset + (n * point_num + p) * 2;
        const T* current_input = input + (n * channels + c) * height * width;

        const T y = current_point[0] + current_offset[0] * gamma;
        const T x = current_point[1] + current_offset[1] * gamma;

        const T grad_current_output = grad_output[index];

        T* grad_current_input = grad_input + (n * channels + c) * height * width;
        T* grad_current_offset = grad_offset + (n * point_num + p) * 2;

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(height,
                                      width,
                                      y,
                                      x,
                                      w1, w2, w3, w4,
                                      y_low, y_high,
                                      x_low, x_high);

        if (x_low >= 0 && x_high >=0 && y_low >= 0 && y_high >= 0)
        {
            atomicAdd(grad_current_input + y_low * width + x_low,
                      grad_current_output * w1);
            atomicAdd(grad_current_input + y_low * width + x_high,
                      grad_current_output * w2);
            atomicAdd(grad_current_input + y_high * width + x_low,
                      grad_current_output * w3);
            atomicAdd(grad_current_input + y_high * width + x_high,
                      grad_current_output * w4);

            T input_00 = current_input[y_low * width + x_low];
            T input_10 = current_input[y_low * width + x_high];
            T input_01 = current_input[y_high * width + x_low];
            T input_11 = current_input[y_high * width + x_high];
            T ogx = gamma * grad_current_output * 
                    (input_11 * (y - y_low) + input_10 * (y_high - y) +
                    input_01 * (y_low - y) + input_00 * (y - y_high));
            T ogy = gamma * grad_current_output * 
                    (input_11 * (x - x_low) + input_01 * (x_high - x) +
                    input_10 * (x_low - x) + input_00 * (x - x_high));
            atomicAdd(grad_current_offset, ogy);
            atomicAdd(grad_current_offset + 1, ogx);
        }
    }
}

#endif  // PROGRESSIVE_SAMPLING_CUDA_KERNEL
