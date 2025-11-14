import math
from typing import Tuple, Union

import mindspore
import mindspore.nn as nn
from mindspore import mint
import mindspore.mint.nn.functional as F
from mindspore.common.initializer import HeUniform, Uniform, initializer
from einops import rearrange


class DualConv3d(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(DualConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        # Ensure kernel_size, stride, padding, and dilation are tuples of length 3
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if kernel_size == (1, 1, 1):
            raise ValueError(
                "kernel_size must be greater than 1. Use make_linear_nd instead."
            )
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        # Set parameters for convolutions
        self.groups = groups
        self.bias = bias

        # Define the size of the channels after the first convolution
        intermediate_channels = (
            out_channels if in_channels < out_channels else in_channels
        )

        # Define parameters for the first convolution
        self.weight1 = mindspore.Parameter(
            mindspore.Tensor(
                intermediate_channels,
                in_channels // groups,
                1,
                kernel_size[1],
                kernel_size[2],
            )
        )
        self.stride1 = (1, stride[1], stride[2])
        self.padding1 = (0, padding[1], padding[2])
        self.dilation1 = (1, dilation[1], dilation[2])
        if bias:
            self.bias1 = mindspore.Parameter(mindspore.Tensor(intermediate_channels))
        else:
            self.register_parameter("bias1", None)

        # Define parameters for the second convolution
        self.weight2 = mindspore.Parameter(
            mindspore.Tensor(
                out_channels, intermediate_channels // groups, kernel_size[0], 1, 1
            )
        )
        self.stride2 = (stride[0], 1, 1)
        self.padding2 = (padding[0], 0, 0)
        self.dilation2 = (dilation[0], 1, 1)
        if bias:
            self.bias2 = mindspore.Parameter(mindspore.Tensor(out_channels))
        else:
            self.register_parameter("bias2", None)

        # Initialize weights and biases
        self.reset_parameters()

    def reset_parameters(self):
        self.weight1.set_data(initializer(HeUniform(negative_slope=math.sqrt(5)), self.weight1.shape, self.weight1.dtype))
        self.weight2.set_data(initializer(HeUniform(negative_slope=math.sqrt(5)), self.weight2.shape, self.weight2.dtype))
        if self.bias:
            fan_in1 = self.weight1.shape[1]
            bound1 = 1 / math.sqrt(fan_in1)
            self.bias1.set_data(initializer(Uniform(bound1), self.bias1.shape, self.bias1.dtype))
            fan_in2 = self.weight2.shape[1]
            bound2 = 1 / math.sqrt(fan_in2)
            self.bias2.set_data(initializer(Uniform(bound2), self.bias2.shape, self.bias2.dtype))

    def construct(self, x, use_conv3d=False, skip_time_conv=False):
        if use_conv3d:
            return self.forward_with_3d(x=x, skip_time_conv=skip_time_conv)
        else:
            return self.forward_with_2d(x=x, skip_time_conv=skip_time_conv)

    def forward_with_3d(self, x, skip_time_conv):
        # First convolution
        x = F.conv3d(
            x,
            self.weight1,
            self.bias1,
            self.stride1,
            self.padding1,
            self.dilation1,
            self.groups,
            padding_mode=self.padding_mode,
        )

        if skip_time_conv:
            return x

        # Second convolution
        x = F.conv3d(
            x,
            self.weight2,
            self.bias2,
            self.stride2,
            self.padding2,
            self.dilation2,
            self.groups,
            padding_mode=self.padding_mode,
        )

        return x

    def forward_with_2d(self, x, skip_time_conv):
        b, c, d, h, w = x.shape

        # First 2D convolution
        x = rearrange(x, "b c d h w -> (b d) c h w")
        # Squeeze the depth dimension out of weight1 since it's 1
        weight1 = self.weight1.squeeze(2)
        # Select stride, padding, and dilation for the 2D convolution
        stride1 = (self.stride1[1], self.stride1[2])
        padding1 = (self.padding1[1], self.padding1[2])
        dilation1 = (self.dilation1[1], self.dilation1[2])
        x = F.conv2d(
            x,
            weight1,
            self.bias1,
            stride1,
            padding1,
            dilation1,
            self.groups,
            padding_mode=self.padding_mode,
        )

        _, _, h, w = x.shape

        if skip_time_conv:
            x = rearrange(x, "(b d) c h w -> b c d h w", b=b)
            return x

        # Second convolution which is essentially treated as a 1D convolution across the 'd' dimension
        x = rearrange(x, "(b d) c h w -> (b h w) c d", b=b)

        # Reshape weight2 to match the expected dimensions for conv1d
        weight2 = self.weight2.squeeze(-1).squeeze(-1)
        # Use only the relevant dimension for stride, padding, and dilation for the 1D convolution
        stride2 = self.stride2[0]
        padding2 = self.padding2[0]
        dilation2 = self.dilation2[0]
        x = F.conv1d(
            x,
            weight2,
            self.bias2,
            stride2,
            padding2,
            dilation2,
            self.groups,
            padding_mode=self.padding_mode,
        )
        x = rearrange(x, "(b h w) c d -> b c d h w", b=b, h=h, w=w)

        return x

    @property
    def weight(self):
        return self.weight2


def test_dual_conv3d_consistency():
    # Initialize parameters
    in_channels = 3
    out_channels = 5
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)

    # Create an instance of the DualConv3d class
    dual_conv3d = DualConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    )

    # Example input tensor
    test_input = mint.randn(1, 3, 10, 10, 10)

    # Perform forward passes with both 3D and 2D settings
    output_conv3d = dual_conv3d(test_input, use_conv3d=True)
    output_2d = dual_conv3d(test_input, use_conv3d=False)

    # Assert that the outputs from both methods are sufficiently close
    assert mint.allclose(
        output_conv3d, output_2d, atol=1e-6
    ), "Outputs are not consistent between 3D and 2D convolutions."
