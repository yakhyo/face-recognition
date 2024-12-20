import torch
from torch import nn, Tensor
from typing import List, Optional, Callable

from models.common import _make_divisible, Conv2dNormActivation, DepthWiseSeparableConv2d, GDC

__all__ = ["mobilenet_v1_025", "mobilenet_v1_050", "mobilenet_v1"]


class MobileNetV1(nn.Module):
    def __init__(self, embedding_dim: int = 512, width_mult: float = 0.25):
        super().__init__()

        filters = [32, 64, 128, 256, 512, 1024]
        filters = [_make_divisible(filter * width_mult) for filter in filters]

        self.stage1: List[nn.Module] = nn.Sequential(
            Conv2dNormActivation(
                in_channels=3,
                out_channels=filters[0],
                kernel_size=3,
                stride=1,  # change from 2 -> 1
                activation_layer=nn.PReLU
            ),
            DepthWiseSeparableConv2d(filters[0], out_channels=filters[1], stride=1),
            DepthWiseSeparableConv2d(filters[1], out_channels=filters[2], stride=2),
            DepthWiseSeparableConv2d(filters[2], out_channels=filters[2], stride=1),
            DepthWiseSeparableConv2d(filters[2], out_channels=filters[3], stride=2),
            DepthWiseSeparableConv2d(filters[3], out_channels=filters[3], stride=1),  # (5) P / 8 -> 640 / 8 = 80
        )
        self.stage2: List[nn.Module] = nn.Sequential(
            DepthWiseSeparableConv2d(filters[3], out_channels=filters[4], stride=2),
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[4], stride=1),  # (11) P / 16 -> 640 / 16 = 40
        )
        self.stage3: List[nn.Module] = nn.Sequential(
            DepthWiseSeparableConv2d(filters[4], out_channels=filters[5], stride=2),
            DepthWiseSeparableConv2d(filters[5], out_channels=filters[5], stride=1),  # (13) P / 32 -> 640 / 32 = 20
        )

        self.output_layer = GDC(filters[5], embedding_dim=embedding_dim)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:  # Check if bias exists
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.output_layer(x)

        return x


if __name__ == "__main__":
    model = MobileNetV1(embedding_dim=512)

    x = torch.randn(1, 3, 112, 112)

    print(model(x).shape)
