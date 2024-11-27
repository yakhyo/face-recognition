from torch.nn import Module, Parameter
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginCosineProduct(nn.Module):
    """
    Implementation of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        one_hot = F.one_hot(label, num_classes=self.out_features).float()

        # Compute the margin with cosine adjustment
        output = self.s * (cosine - one_hot * self.m)
        return output

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, s={self.s}, m={self.m})'


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-self.power))

        # Cosine and phi(theta)
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)

        theta = cos_theta.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k

        NormOfFeature = torch.norm(input, 2, 1, keepdim=True)

        # One-hot label creation
        one_hot = F.one_hot(label, num_classes=self.out_features).float()

        # Final output computation
        output = one_hot * (phi_theta - cos_theta) / (1 + self.lamb) + cos_theta
        output *= NormOfFeature

        return output

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, m={self.m})'


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, m_arc=0.50, m_am=0.0, scale=64.0):
        super().__init__()
        self.m_arc = m_arc
        self.m_am = m_am
        self.scale = scale

        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.cos_margin = math.cos(m_arc)
        self.sin_margin = math.sin(m_arc)
        self.min_cos_theta = math.cos(math.pi - m_arc)

    def forward(self, embbedings, label):
        embbedings = F.normalize(embbedings, dim=1)
        kernel_norm = F.normalize(self.weight, dim=0)

        cos_theta = torch.mm(embbedings, kernel_norm).clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin

        # torch.where doesn't support fp16 input
        is_half = cos_theta.dtype == torch.float16

        cos_theta_m = torch.where(cos_theta > self.min_cos_theta, cos_theta_m, cos_theta.float() - self.m_am)
        cos_theta_m = cos_theta_m.half() if is_half else cos_theta_m

        index = torch.zeros_like(cos_theta)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte().bool()

        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale

        return output