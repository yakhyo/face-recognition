# CosFace with SphereNet Backbones

This repository contains the implementation of CosFace trained with SphereNet backbones (Sphere20, Sphere36, Sphere64) on the MS1M-ArcFace dataset. Evaluation was performed on the AFLW dataset.

---

## Features

- **Pretrained Weights**: Trained model weights are provided for reproducibility.
- **Multi-GPU Support**: Leverage distributed training for better performance.
- **Rewritten Code**: Simplified for ease of usability and flexibility.

---

## Results

### Evaluation on AFLW Dataset

| Backbone | Dataset      | Evaluation Dataset | Best Epoch Accuracy (%) | Best Epoch (#) | Last Epoch Accuracy (%) | Last Epoch (#) | Model Link    |
| -------- | ------------ | ------------------ | ----------------------- | -------------- | ----------------------- | -------------- | ------------- |
| Sphere20 | MS1M-ArcFace | AFLW               | 99.67                   | 30             | 99.67                   | 30             | [Download](#) |
| Sphere36 | MS1M-ArcFace | AFLW               | 99.72                   | 13             | 99.55                   | 23             | [Download](#) |
| Sphere64 | MS1M-ArcFace | AFLW               | XX.XX                   | XX             | XX.XX                   | XX             | [Download](#) |

_Note: Replace `XX.XX` and `XX` for Sphere64 with actual values after evaluation._

---

## Getting Started

### Installation

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```
