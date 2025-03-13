import cv2
import numpy as np
import onnxruntime as ort
from utils.alignment import face_alignment
from utils.general import compute_similarity

from uniface import RetinaFace

import warnings

warnings.filterwarnings("ignore")


class ONNXFaceRecognition:
    """
    Face recognition model using ONNX Runtime for inference and OpenCV for image preprocessing,
    utilizing an external face alignment function.
    """

    def __init__(self, model_path: str = None, session: ort.InferenceSession = None):
        """
        Initializes the ArcFace ONNX model for inference.

        Args:
            model_path (str): Path to the ONNX model file.
            session (ort.InferenceSession): Existing ONNX session (optional).
        """
        self.session = session
        self.input_mean = 127.5
        self.input_std = 127.5

        if session is None:
            assert model_path is not None, "Please provide a valid model path."
            self.session = ort.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape

        self.input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])

        outputs = self.session.get_outputs()
        output_names = []
        for output in outputs:
            output_names.append(output.name)

        self.output_names = output_names
        assert len(self.output_names) == 1, "Expected only one output node."
        self.output_shape = outputs[0].shape

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image: resize, normalize, and convert it to a blob.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Preprocessed image as a NumPy array ready for inference.
        """
        image = cv2.resize(image, self.input_size)  # Resize to (112, 112)
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / self.input_std,
            size=self.input_size,
            mean=(self.input_mean, self.input_mean, self.input_mean),
            swapRB=True  # Convert BGR to RGB
        )
        return blob

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Extracts face embedding from an aligned image.

        Args:
            image (np.ndarray): Input face image (BGR format).
            landmarks (np.ndarray): Facial landmarks (5 points for alignment).

        Returns:
            np.ndarray: 512-dimensional face embedding.
        """
        aligned_face = face_alignment(image, landmarks)  # Use your function for alignment
        blob = self.preprocess(image)  # Convert to blob
        embedding = self.session.run(None, {self.input_name: blob})[0]
        return embedding  # Return the 512-D feature vector


def compare_faces(
        model: ONNXFaceRecognition,
        img1: np.ndarray,
        landmarks1: np.ndarray,
        img2: np.ndarray,
        landmarks2: np.ndarray,
        threshold: float = 0.35
) -> tuple:
    """
    Compares two face images and determines if they belong to the same person.

    Args:
        model (ONNXFaceRecognition): The face recognition model instance.
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

    uniface_inference = RetinaFace(model="retinaface_mnet_v2", conf_thresh=0.45)  # Face detector

    model_path = "mobilenetv2_mcp.onnx"  # Path to your ONNX model
    face_recognizer = ONNXFaceRecognition(model_path)

    # Load images
    img1 = cv2.imread("assets/b_01.jpg")
    img2 = cv2.imread("assets/b_02.jpg")

    boxes, landmarks = uniface_inference.detect(img1)  # Detect faces in the image
    landmarks1 = landmarks[0]  # Get landmarks for the first face

    boxes, landmarks = uniface_inference.detect(img2)  # Detect faces in the image
    landmarks2 = landmarks[0]  # Get landmarks for the second face

    # Compare two face images
    similarity, is_same = compare_faces(face_recognizer, img1, landmarks1, img2, landmarks2, threshold=0.35)

    print(f"Similarity: {similarity:.4f} - {'Same person' if is_same else 'Different person'}")
