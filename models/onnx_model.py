# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
import onnxruntime as ort

from utils.face_utils import compute_similarity, face_alignment


class ONNXFaceEngine(object):
    """
    Face recognition model using ONNX Runtime for inference and OpenCV for image preprocessing,
    utilizing an external face alignment function.
    """

    def __init__(self, model_path: str = None, session: ort.InferenceSession = None):
        """
        Initializes the ONNXFaceEngine model for inference.

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
        blob = self.preprocess(aligned_face)  # Convert to blob
        embedding = self.session.run(self.output_names, {self.input_name: blob})[0]
        return embedding  # Return the 512-D feature vector
