import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as _resnet18, ResNet18_Weights

__all__ = ["resnet18"]


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        # Load ResNet-18 with ImageNet weights
        model = _resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Get the number of input features of the final fully connected layer
        num_ftrs = model.fc.in_features

        # Replace the final layer with a new linear layer to produce the embedding
        model.fc = nn.Linear(num_ftrs, embedding_dim)

        self.feature_extractor = model

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.normalize(x)  # L2 Normalization is applied here, aligning with user's example
        return x


def resnet18(embedding_dim=512, **kwargs):
    """
    ResNet-18 feature extractor with ImageNet pre-trained weights and L2 feature normalization.
    """
    return ResNet18FeatureExtractor(embedding_dim)
