import cv2
import uniface
import numpy as np
import onnxruntime as ort

from models import ONNXFaceEngine
from utils.face_utils import compute_similarity, face_alignment

import warnings
warnings.filterwarnings("ignore")


def compare_faces(
        model: ONNXFaceEngine,
        img1: np.ndarray,
        landmarks1: np.ndarray,
        img2: np.ndarray,
        landmarks2: np.ndarray,
        threshold: float = 0.35
) -> tuple:
    """
    Compares two face images and determines if they belong to the same person.

    Args:
        model (ONNXFaceEngine): The face recognition model instance.
        img1 (np.ndarray): First face image (BGR format).
        landmarks1 (np.ndarray): Facial landmarks for img1.
        img2 (np.ndarray): Second face image (BGR format).
        landmarks2 (np.ndarray): Facial landmarks for img2.
        threshold (float): Similarity threshold for face matching.

    Returns:
        tuple[float, bool]: Similarity score and match result (True/False).
    """
    feat1 = model.get_embedding(img1, landmarks1)
    feat2 = model.get_embedding(img2, landmarks2)
    similarity = compute_similarity(feat1, feat2)
    is_match = similarity > threshold
    return similarity, is_match


# Example usage
if __name__ == "__main__":

    uniface_inference = uniface.RetinaFace(model="retinaface_mnet_v2", conf_thresh=0.45)  # Face detector

    model_path = "mobilenetv2_mcp.onnx"  # Path to your ONNX model
    face_recognizer = ONNXFaceEngine(model_path)

    # Load images
    img1 = cv2.imread("assets/b_01.jpg")
    img2 = cv2.imread("assets/b_02.jpg")

    detections1 = uniface_inference.detect(img1)  # Detect faces in the image
    landmarks1 = np.array(detections1[0]['landmarks'], dtype=np.float32)  # Get landmarks for the first face

    detections2 = uniface_inference.detect(img2)  # Detect faces in the image
    landmarks2 = np.array(detections2[0]['landmarks'], dtype=np.float32)  # Get landmarks for the second face

    # Compare two face images
    similarity, is_same = compare_faces(face_recognizer, img1, landmarks1, img2, landmarks2, threshold=0.30)

    print(f"Similarity: {similarity:.4f} - {'Same person' if is_same else 'Different person'}")
