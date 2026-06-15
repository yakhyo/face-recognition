import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as _resnet18, ResNet18_Weights, resnet50 as _resnet50, ResNet50_Weights

__all__ = ["resnet18", "resnet50"]


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet_factory, weights, embedding_dim=512):
        super().__init__()

        # Load ResNet with ImageNet weights
        model = resnet_factory(weights=weights)

        # Get the number of input features of the final fully connected layer
        num_ftrs = model.fc.in_features

        # Replace the final layer with a new linear layer to produce the embedding
        model.fc = nn.Linear(num_ftrs, embedding_dim)

        self.feature_extractor = model

    def forward(self, x):
        x = self.feature_extractor(x)

        return x


def resnet18(embedding_dim=512, **kwargs):
    """
    ResNet-18 feature extractor with ImageNet pre-trained weights and L2 feature normalization.
    """
    return ResNetFeatureExtractor(_resnet18, ResNet18_Weights.IMAGENET1K_V1, embedding_dim)


def resnet50(embedding_dim=512, **kwargs):
    """
    ResNet-50 feature extractor with ImageNet pre-trained weights and L2 feature normalization.
    """
    return ResNetFeatureExtractor(_resnet50, ResNet50_Weights.IMAGENET1K_V1, embedding_dim)
