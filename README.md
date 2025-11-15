# Face-Recognition Training Framework

[![Downloads](https://img.shields.io/github/downloads/yakhyo/face-recognition/total)](https://github.com/yakhyo/face-recognition/releases)
[![GitHub Repo stars](https://img.shields.io/github/stars/yakhyo/face-recognition)](https://github.com/yakhyo/face-recognition/stargazers)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/face-recognition)
[![GitHub License](https://img.shields.io/github/license/yakhyo/face-recognition)](https://github.com/yakhyo/face-recognition/blob/main/LICENSE)

---

## Updates

- `2025/03/13`: Face Detection added in `onnx_inference.py`. <!-- Added face detection capabilities to ONNX inference pipeline -->
- `2025/03/13`: ONNX Export and Inference has been added. <!-- Introduced ONNX model export functionality and inference support -->
- `2025/01/03`: We released the **Face-Recognition** training framework and pretrained model weights. <!-- Initial release with training framework and PyTorch weights -->

---

## Results

| Dataset | Backbone          | LFW (%) | CALFW (%) | CPLFW (%) | AgeDB_30 (%) | Num Params |
| ------- | ----------------- | ------- | --------- | --------- | ------------ | ---------- |
| MS1MV2  | Sphere20          | 99.67   | 95.61     | 88.75     | 96.58        | 24.5M      |
| MS1MV2  | Sphere36          | 99.72   | 95.64     | 89.92     | 96.83        | 34.6M      |
| MS1MV2  | MobileNetV1_0.25  | 98.76   | 92.02     | 82.37     | 90.02        | 0.36M      |
| MS1MV2  | MobileNetV2       | 99.55   | 94.87     | 86.89     | 95.16        | 2.29M      |
| MS1MV2  | MobileNetV3_Small | 99.30   | 93.77     | 85.29     | 92.79        | 1.25M      |
| MS1MV2  | MobileNetV3_Large | 99.53   | 94.56     | 86.79     | 95.13        | 3.52M      |

---

## Features

| Date       | Feature Description                                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2024-12-15 | **Training Pipeline**: Introduced a simple and effective pipeline for face-recognition training with support for `DDP` and single GPU configurations. |
| 2024-12-15 | **Pretrained Models**: Added support for `MobileNetV1/V2/V3`, `Sphere20`, and `Sphere36` models for versatile use-cases and performance tiers.        |
| 2024-12-15 | **Dataset Downloads**: Easy access to aligned and cropped training and validation datasets via Kaggle links.                                          |
| 2024-12-15 | **Modular Codebase**: Fully modular and reproducible codebase for easier customization and extension.                                                 |
| 2024-12-15 | **Dataset Compatibility**: Supports `CASIA-WebFace`, `VGGFace2`, and `MS1MV2` datasets, pre-aligned and cropped for streamlined training.             |

---

## Getting Started

### Installation

```bash
git clone https://github.com/yakhyo/face-recognition.git
cd face-recognition
pip install -r requirements.txt
```

### Training

Codebase supports **DDP**, to run using **DDP** please use below example command:

```bash
torchrun --nproc_per_node=2 train.py --root data/train/ms1m_112x112 --database MS1M --network mobilenetv1 --classifier MCP
```

If you have a single GPU then use the below example command:

```bash
python train.py --root data/train/ms1m_112x112 --database MS1M --network mobilenetv1 --classifier MCP
```

---

### Evaluate

To evaluate, please modify model, weights, and validation data filenames in `evaluate.py`

```bash
python evaluate.py
```

## Pretrained Model Weights (v0.0.1)

The following pretrained model weights are available for download under the release [v0.0.1](https://github.com/yakhyo/face-recognition/releases/tag/v0.0.1):

| Model             | Download Link                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------- |
| MobileNetV1_0.25  | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv1_0.25_mcp.pth)       |
| MobileNetV2       | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv2_mcp.pth)       |
| MobileNetV3_Small | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv3_small_mcp.pth) |
| MobileNetV3_Large | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv3_large_mcp.pth) |
| Sphere20          | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/sphere20_mcp.pth)          |
| Sphere36          | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/sphere36_mcp.pth)          |

### Usage

1. Download the model weights from the links above.
2. Place the weights in the desired directory (e.g., `weights/`).
3. Update your training or inference script to load the appropriate model weights.

---

## ONNX Model Weights (v0.0.1)

The following ONNX model weights are available for download under the release [v0.0.1](https://github.com/yakhyo/face-recognition/releases/tag/v0.0.1):

| Model             | Download Link                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------- |
| MobileNetV1_0.25  | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv1_0.25.onnx)       |
| MobileNetV2       | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv2.onnx)       |
| MobileNetV3_Small | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv3_small.onnx) |
| MobileNetV3_Large | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv3_large.onnx) |
| Sphere20          | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/sphere20.onnx)          |
| Sphere36          | [Download](https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/sphere36.onnx)          |

### ONNX Usage

1. Download the ONNX model weights from the links above.
2. Use the ONNX models directly for inference with ONNXRuntime.
3. These models are optimized for production deployment and cross-platform compatibility.
4. Run `python onnx_inference.py` to test ONNX model inference.

---

## ONNX Export

Run following command to export to ONNX:

```bash
python onnx_export.py -w [path/to/weight/file] -n [network/architecture/name] --dynamic[Optional]
```

## ONNX Inference

Run `onnx_inference.py` to use ONNXRuntime. This inference calculates the similarity between two face images.

```bash
python onnx_inference.py
```

## PyTorch Inference

Run `inference.py` for PyTorch model inference. This inference calculates the similarity between two face images.

---

## Dataset

You can download aligned and cropped (112x112) training and validation datasets from Kaggle.

### Training Data

- [CASIA-WebFace 112x112](https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112) from `opensphere`
  - Identities: 10.6k
  - #Images: 491k
- [VGGFace2 112x112](https://www.kaggle.com/datasets/yakhyokhuja/vggface2-112x112) from `opensphere`
  - Identities: 8.6k
  - #Images: 3.1M
- [MS1MV2 112x112](https://www.kaggle.com/datasets/yakhyokhuja/ms1m-arcface-dataset) from `insightface`
  - Identities: 85.7k
  - #Images: 5.8M

### Validation Data

Validation data contains AgeDB_30, CALFW, CPLFW, and LFW datasets.

- [AgeDB_30, CALFW, CPLFW, LFW 112x112](https://www.kaggle.com/datasets/yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112)

---

### Folder Structure

```
data/
|-- train/
|   |-- ms1m_112x112/
|   |-- vggface2_112x112/
|   |-- webface_112x112/
|-- val/
|   |-- agedb_30_112x112/
|   |-- calfw_112x112/
|   |-- cplfw_112x112/
|   |-- lfw_112x112/
|   |-- agedb_30_ann.txt
|   |-- calfw_ann.txt
|   |-- cplfw_ann.txt
|   |-- lfw_ann.txt
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
