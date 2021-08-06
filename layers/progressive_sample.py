from torch import nn
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['progressive_sampling_forward',
                                          'progressive_sampling_backward'])


class ProgressiveSamplingFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                point,
                offset=None,
                gamma=1.0):
        ctx.gamma = float(gamma)
        if offset is None:
            offset = torch.zeros_like(point)

        output_shape = (point.size(0),
                        point.size(1),
                        input.size(1))

        output = input.new_zeros(output_shape)

        ext_module.progressive_sampling_forward(input,
                                                point,
                                                offset,
                                                output,
                                                ctx.gamma)

        ctx.save_for_backward(input, point, offset)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, point, offset = ctx.saved_tensors
        grad_input = grad_output.new_zeros(input.shape)
        grad_offset = grad_output.new_zeros(offset.shape)

        ext_module.progressive_sampling_backward(grad_output,
                                                 input,
                                                 point,
                                                 offset,
                                                 grad_input,
                                                 grad_offset,
                                                 ctx.gamma)
        return grad_input, None, grad_offset, None


progressive_sampling = ProgressiveSamplingFunction.apply


class ProgressiveSample(nn.Module):
    def __init__(self,
                 gamma=1.0):
        super(ProgressiveSample, self).__init__()
        self.gamma = gamma

    def forward(self, input, point, offset):
        """
        :param input: [n, c, h, w]
        :param point: [n, point_num, 2] (y, x)
        :param offset: [n, point_num, 2] (y, x)
        :output: [n, point_num, c]
        """
        return progressive_sampling(input, point, offset, self.gamma)
