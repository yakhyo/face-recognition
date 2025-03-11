import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from uniface import RetinaFace

from models.mobilenetv1 import MobileNetV1
from models.mobilenetv2 import MobileNetV2

from models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from models.sphereface import sphere20, sphere36, sphere64


retinaface = RetinaFace(model="retinaface_mnet_v2", conf_thresh=0.45)


def get_network(model_name: str) -> torch.nn.Module:
    """
    Returns the appropriate model based on the provided model name.

    Args:
        model_name (str): Name of the model architecture.

    Returns:
        torch.nn.Module: The selected deep learning model.
    """
    models = {
        "sphere20": sphere20(embedding_dim=512, in_channels=3),
        "sphere36": sphere36(embedding_dim=512, in_channels=3),
        "sphere64": sphere64(embedding_dim=512, in_channels=3),
        "mobilenetv1": MobileNetV1(embedding_dim=512),
        "mobilenetv2": MobileNetV2(embedding_dim=512),
        "mobilenetv3_small": mobilenet_v3_small(embedding_dim=512),
        "mobilenetv3_large": mobilenet_v3_large(embedding_dim=512),
    }

    if model_name not in models:
        raise ValueError(f"Unsupported network '{model_name}'! Available options: {list(models.keys())}")

    return models[model_name]



def load_model(model_name: str, model_path: str, device: torch.device = None) -> torch.nn.Module:
    """
    Loads a deep learning model with pre-trained weights.
    """
    model = get_network(model_name)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model '{model_name}' from {model_path}: {e}")

    return model


def get_transform():
    """
    Returns the image preprocessing transformations.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def extract_features(model, device, img_path: str) -> np.ndarray:
    """
    Extracts face features from an image.
    """
    transform = get_transform()

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"Error opening image {img_path}: {e}")

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor).squeeze().cpu().numpy()
    return features


def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> np.float32:
    """
    Computes cosine similarity between two feature vectors.
    """
    feat1, feat2 = feat1.ravel(), feat2.ravel()
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity


def compare_faces(model, device, img1_path: str, img2_path: str, threshold: float = 0.35) -> tuple[float, bool]:
    """
    Compares two face images and determines if they belong to the same person.
    """
    feat1 = extract_features(model, device, img1_path)
    feat2 = extract_features(model, device, img2_path)

    similarity = compute_similarity(feat1, feat2)
    is_same = similarity > threshold

    return similarity, is_same


if __name__ == "__main__":
    # Example usage with model selection
    model_name = "mobilenetv2"
    model_path = "weights/mobilenetv2_mcp.pth"
    threshold = 0.35

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, device = load_model(model_name, model_path)

    # Compare faces
    similarity, is_same = compare_faces(
        model, device,
        img1_path="assets/b_01.jpg",
        img2_path="assets/b_02.jpg",
        threshold=threshold
    )

    print(f"Similarity: {similarity:.4f} - {'same' if is_same else 'different'} (Threshold: {threshold})")