import torch
import torch.nn as nn
from typing import Callable, List, Optional, Type

from utils.layers import Conv2dNormActivation

__all__ = ["sphere20", "sphere36", "sphere64"]


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers.
    Args:
        channels (int): Number of input and output channels.
    """

    def __init__(self, channels: int, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            Conv2dNormActivation(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                activation_layer=nn.PReLU,
                norm_layer=norm_layer,
                bias=False
            ),
            Conv2dNormActivation(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                activation_layer=nn.PReLU,
                norm_layer=norm_layer,
                bias=False
            )
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with residual connection.
        """
        return x + self.block(x)


class SphereNet(nn.Module):
    """
    SphereNet: A neural network for feature embedding, composed of Residual Blocks.
    Args:
        block (Type[ResidualBlock]): Residual block type for the network.
        layers (List[int]): Number of blocks in each layer.
        embedding_dim (int, optional): Dimension of the output embedding. Default: 512.
        in_channels (Optional[int], optional): Number of input channels. Default: 3.
        norm_layer (Optional[bool], optional): Use BatchNorm if True. Default: False.
    """

    def __init__(
            self,
            block: Type[ResidualBlock],
            layers: List[int],
            embedding_dim: int = 512,
            in_channels: Optional[int] = 3,
            norm_layer: Optional[bool] = False
    ) -> None:
        super().__init__()
        self._norm_layer = None
        if norm_layer:
            self._norm_layer = nn.BatchNorm2d

        filters = [64, 128, 256, 512]

        # Define the layers
        self.layer1 = self._make_layer(ResidualBlock, in_channels, filters[0], layers[0], stride=2)
        self.layer2 = self._make_layer(ResidualBlock, filters[0], filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(ResidualBlock, filters[1], filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(ResidualBlock, filters[2], filters[3], layers[3], stride=2)

        # Fully connected layer
        self.fc = nn.Linear(filters[3] * 7 * 7, embedding_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)

    def _make_layer(
        self,
        block: Type[ResidualBlock],
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:

        layers: List[nn.Module] = [
            Conv2dNormActivation(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation_layer=nn.PReLU,
                norm_layer=self._norm_layer
            )
        ]
        for _ in range(0, num_blocks):
            layers.append(block(out_channels, self._norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def sphere20(embedding_dim, **kwargs):
    """
    Creates a SphereNet model with 20 layers.

    Args:
        embedding_dim (int): Dimensionality of the embedding output.
        **kwargs: Additional arguments to pass to the SphereNet initializer.

    Returns:
        SphereNet: Sphere20 model instance.
    """

    return SphereNet(ResidualBlock, [1, 2, 4, 1], embedding_dim, **kwargs)


def sphere36(embedding_dim, **kwargs):
    """
    Creates a SphereNet model with 20 layers.

    Args:
        embedding_dim (int): Dimensionality of the embedding output.
        **kwargs: Additional arguments to pass to the SphereNet initializer.

    Returns:
        SphereNet: Sphere20 model instance.
    """

    return SphereNet(ResidualBlock, [2, 4, 8, 2], embedding_dim, **kwargs)


def sphere64(embedding_dim, **kwargs):
    """
    Creates a SphereNet model with 64 layers.

    Args:
        embedding_dim (int): Dimensionality of the embedding output.
        **kwargs: Additional arguments to pass to the SphereNet initializer.

    Returns:
        SphereNet: Sphere64 model instance.
    """
    return SphereNet(ResidualBlock, [3, 8, 16, 3], embedding_dim, **kwargs)
