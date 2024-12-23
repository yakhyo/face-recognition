import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginCosineProduct(nn.Module):
    """ Reference: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, label):
        # Cosine(theta) & phi(theta)
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        one_hot = F.one_hot(label.long(), num_classes=self.out_features).float()

        # Compute the margin with cosine adjustment
        output = self.s * (cosine - one_hot * self.m)
        return output

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, s={self.s}, m={self.m})'


# modified from https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py
class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4):
        super().__init__()
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

    def forward(self, embeddings, label):
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-self.power))

        # Cosine and phi(theta)
        cos_theta = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)

        theta = cos_theta.acos()
        k = (self.m * theta / math.pi).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k

        NormOfFeature = torch.norm(embeddings, 2, 1, keepdim=True)

        # One-hot label creation
        one_hot = F.one_hot(label.long(), num_classes=self.out_features).float()

        # Final output computation
        output = one_hot * (phi_theta - cos_theta) / (1 + self.lamb) + cos_theta
        output *= NormOfFeature

        return output

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, m={self.m})'
