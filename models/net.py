import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu2 = nn.PReLU(planes)

    def forward(self, x):
        return x + self.prelu2(self.conv2(self.prelu1(self.conv1(x))))

class SphereNet(nn.Module):
    def __init__(self, type=20, is_gray=False):
        super(SphereNet, self).__init__()
        layers = [1, 2, 4, 1] if type == 20 else [3, 7, 16, 3]
        filter_list = [1 if is_gray else 3, 64, 128, 256, 512]
        self.layer1 = self._make_layer(Block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(Block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(Block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(Block, filter_list[3], filter_list[4], layers[3], stride=2)
        self.fc = nn.Linear(512 * 7 * 7, 512)

        self._initialize_weights()

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = [nn.Conv2d(inplanes, planes, 3, stride, 1), nn.PReLU(planes)]
        layers += [block(planes) for _ in range(blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class BlockIR(nn.Module):
    def __init__(self, inplanes, planes, stride, dim_match):
        super(BlockIR, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        ) if not dim_match else None

    def forward(self, x):
        residual = x
        out = self.conv1(self.bn1(x))
        out = self.prelu(self.bn2(out))
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return out + residual

class LResNet(nn.Module):
    def __init__(self, block, layers, filter_list, is_gray=False):
        super(LResNet, self).__init__()
        self.conv1 = nn.Conv2d(1 if is_gray else 3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 7),
            nn.Dropout(p=0.4),
            nn.Linear(filter_list[4] * 7 * 7, 512),
            nn.BatchNorm1d(512)
        )
        self._initialize_weights()

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = [block(inplanes, planes, stride, False)]
        layers += [block(planes, planes, stride=1, dim_match=True) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def LResNet50E_IR(is_gray=False):
    filter_list = [64, 64, 128, 256, 512]
    layers = [3, 4, 14, 3]
    return LResNet(BlockIR, layers, filter_list, is_gray)
