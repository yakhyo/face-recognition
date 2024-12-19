import torch
import torch.nn as nn
from typing import Callable, List, Optional, Type


def _make_divisible(v: float, divisor: int = 8) -> int:
    """This function ensures that all layers have a channel number divisible by 8"""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(nn.Sequential):
    """
    A modular convolutional block that combines a 2D convolutional layer, optional normalization, and optional activation.

    Args:
        in_channels (int): Number of input channels for the convolution.
        out_channels (int): Number of output channels for the convolution.
        kernel_size (int): Size of the convolution kernel. Default: 3.
        stride (int): Stride of the convolution. Default: 1.
        padding (Optional[int]): Padding for the convolution. Default: None (calculated as `(kernel_size - 1) // 2 * dilation`).
        groups (int): Number of groups for group convolution. Default: 1.
        norm_layer (Optional[Callable[..., nn.Module]]): Normalization layer to apply after convolution. Default: `None`.
        activation_layer (Optional[Callable[..., nn.Module]]): Activation layer to apply after normalization. Default: `nn.PReLU`.
        dilation (int): Dilation factor for the convolution. Default: 1.
        inplace (Optional[bool]): Whether to perform the activation in-place (for applicable activations). Default: True.
        bias (bool): Whether to include a bias term in the convolution. Default: True.

    Notes:
        - If `padding` is not specified, it is automatically calculated to preserve spatial dimensions for stride=1.
        - If `norm_layer` is set to `None`, the normalization step will be skipped.
        - If `activation_layer` is set to `None`, the activation step will be skipped.

    Example:
        >>> block = Conv2dNormActivation(64, 128, kernel_size=3, stride=2, activation_layer=nn.LeakyReLU)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = block(x)  # Output tensor with shape (1, 128, 16, 16)

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., nn.Module]] = nn.PReLU,
            dilation: int = 1,
            inplace: Optional[bool] = True,
            bias: bool = True,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            if activation_layer == nn.PReLU:
                layers.append(activation_layer(num_parameters=out_channels))
            else:
                params = {} if inplace is None else {"inplace": inplace}
                layers.append(activation_layer(**params))

        super().__init__(*layers)


class DepthWiseSeparableConv2d(nn.Sequential):
    """DepthWise Separable Convolutional with Depthwise and Pointwise layers followed by nn.BatchNorm2d and nn.ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None
    ) -> None:

        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = [
            Conv2dNormActivation(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
            ),  # Depthwise
            Conv2dNormActivation(in_channels, out_channels, kernel_size=1)  # Pointwise
        ]

        super().__init__(*layers)


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class GDC(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()
        self.features = nn.Sequential(
            LinearBlock(in_channels, in_channels, kernel_size=7, stride=1, padding=0, groups=in_channels),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
