# Face-Recognition Training Framework

[![Downloads](https://img.shields.io/github/downloads/yakhyo/face-recognition/total)](https://github.com/yakhyo/face-recognition/releases)
[![GitHub Repo stars](https://img.shields.io/github/stars/yakhyo/face-recognition)](https://github.com/yakhyo/face-recognition/stargazers)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/face-recognition)
[![GitHub License](https://img.shields.io/github/license/yakhyo/face-recognition)](https://github.com/yakhyo/face-recognition/blob/main/LICENSE)

---

## 🔥 Updates

- `2024/12/xx`: 🔥 We released the **Face-Recognition** training framework.

---

## Results

| Dataset | Backbone          | LFW (%) | CALFW (%) | CPLFW (%) | AgeDB_30 (%) | Num Params |
| ------- | ----------------- | ------- | --------- | --------- | ------------ | ---------- |
| MS1MV2  | Sphere20          | 99.67   | 95.61     | 88.75     | XX.XX        | XXM        |
| MS1MV2  | Sphere36          | 99.72   | 95.64     | 89.92     | XX.XX        | XXM        |
| MS1MV2  | Sphere64          | XX.XX   | XX.XX     | XX.XX     | XX.XX        | XXM        |
| MS1MV2  | MobileNetV1_0.25  | XX.XX   | XX.XX     | XX.XX     | XX.XX        | XXM        |
| MS1MV2  | MobileNetV2       | XX.XX   | XX.XX     | XX.XX     | XX.XX        | XXM        |
| MS1MV2  | MobileNetV3_Small | XX.XX   | XX.XX     | XX.XX     | XX.XX        | XXM        |
| MS1MV2  | MobileNetV3_Large | XX.XX   | XX.XX     | XX.XX     | XX.XX        | XXM        |

---

## Features

- **Feature Extraction**: Extract high-dimensional embeddings for each detected face.
- **Face Recognition**: Match faces against a database of known identities.
- **Real-Time Processing**: Optimized for real-time performance.
- **Pretrained Models**: Includes support for pretrained models like `MobileNetV1/V2/V3`, `Sphere20`, and `Sphere36`.

---

## Getting Started

### Installation

```bash
git clone https://github.com/yakhyo/face-recognition.git
cd face-recognition
pip install -r requirements.txt
```

### Training

Codebase supports DDP, to run using DDP please use below example command:

```bash
torchrun --nproc_per_node=2 main.py --root data/train/ms1m_112x112 --database MS1M --network mobilenetv1 --classifier MCP
```

If you have a sinlge GPU then use below example command:

```bash
python main.py --root data/train/ms1m_112x112 --database MS1M --network mobilenetv1 --classifier MCP
```

### Evaluate

To evaluate, please modify model, weights and validation data filenames in `lfw_eval.py`

```bash
python lfw_eval.py
```

---

## Dataset

You can download aligned and cropped (112x112) training and validation datasets from Kaggle.

### Training Data

- [CASIA-WebFace 112x112](https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112)
  - Identites: 10.6k
  - #Images: 491k
- [VGGFace2 112x112](https://www.kaggle.com/datasets/yakhyokhuja/vggface2-112x112)
  - Identities: 8.6k
  - #Images: 3.1M
- [MS1MV2 112x112](https://www.kaggle.com/datasets/yakhyokhuja/ms1m-arcface-dataset)
  - Identities: 85.7k
  - #Images: 5.8M

### Validation Data

Validation data contains AgeDB_30, CALFW, CPLFW, LFW datasets.

- [AgeDB_30, CALFW CPLFW, LFW 112x112](https://www.kaggle.com/datasets/yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112)

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
