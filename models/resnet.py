import torch.nn as nn
from torchvision.models import resnet18 as _resnet18, ResNet18_Weights

__all__ = ["resnet18"]


def resnet18(embedding_dim=512, **kwargs):
    """
    ResNet-18 feature extractor with ImageNet pre-trained weights.
    The final classification layer is replaced to output the specified embedding dimension.
    """
    # Load ResNet-18 with ImageNet weights
    # We ignore the kwargs as we are hardcoding the weights for now per user request.
    model = _resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Get the number of input features of the final fully connected layer
    num_ftrs = model.fc.in_features
    
    # Replace the final layer with a new linear layer to produce the embedding
    model.fc = nn.Linear(num_ftrs, embedding_dim)
    
    # Note: If the project used a custom ResNet, we would need to check its forward method.
    # Since it's torchvision, the forward returns the output of model.fc which is now the embedding.
    
    return model