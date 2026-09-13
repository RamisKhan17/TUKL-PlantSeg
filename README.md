# Model-Training — Plant Segmentation

This repository contains the training and inference code for both **binary** and **multi-class** semantic segmentation tasks on the PlantSeg dataset. The training pipeline utilizes `segmentation-models-pytorch` and logs metrics using `wandb`.

---

## Project Structure

```
Model-Training/
├── binary/                 # Code for binary segmentation (e.g. rust vs background)
│   ├── DataLoading.py      # Dataset loader (resizes & pads to 512x512)
│   ├── patching.py         # Divides large images into patches
│   ├── predict.py          # Inference & Grad-CAM visualization
│   ├── test.py             # Evaluates performance on test split
│   ├── train.py            # Main training loop (DeepLabV3+)
│   ├── Unpatching.py       # Reassembles image patches
│   └── utils.py            # Helpers for checkpointing, loaders, and evaluation
├── multiclass/             # Code for 116-class segmentation
│   ├── DataLoading.py      # Dataset loader (multiclass specific formatting)
│   ├── predict.ipynb       # Jupyter Notebook for multiclass inference visualization
│   ├── test.py             # Evaluates multiclass performance with TTA
│   ├── train.py            # Main training loop (FPN/DeepLabV3+)
│   └── utils.py            # Helpers utilizing torchmetrics (IoU, Accuracy)
├── Data/                   # Datasets (Excluded from git)
│   ├── images/             # Raw input images
│   └── annotations/        # Segmentation masks
├── models/                 # Model checkpoints (Excluded from git)
├── saved_images/           # Output visualizations (Excluded from git)
└── wandb/                  # Weights & Biases logs (Excluded from git)
```

---

## Requirements

| Package                       | Purpose                               |
| ----------------------------- | ------------------------------------- |
| `torch` / `torchvision`       | Deep learning framework               |
| `segmentation-models-pytorch` | Architectures (DeepLabV3+, FPN, etc.) |
| `albumentations`              | Image augmentation                    |
| `torchmetrics`                | Evaluating multiclass metrics         |
| `patchify`                    | Tiling images into patches            |
| `Pillow`                      | Image I/O                             |
| `numpy`                       | Array operations                      |
| `matplotlib`                  | Visualization                         |
| `pytorch-grad-cam`            | Grad-CAM explainability               |
| `wandb`                       | Logging and tracking experiments      |

Install the dependencies:

```bash
pip install torch torchvision segmentation-models-pytorch albumentations \
            torchmetrics patchify Pillow numpy matplotlib grad-cam wandb
```

---

## Usage

> **Note:** Make sure you set your paths in the scripts properly. The main scripts expect datasets to be under the `Data/` directory relative to the `Model-Training` root. Run the scripts from the `Model-Training` folder (e.g., `python binary/train.py`).

### Binary Segmentation

- **Training:** Run `python binary/train.py`. The script uses `DeepLabV3+` with a `timm-efficientnet-b3` encoder and logs runs to Weights & Biases.
- **Evaluation:** Run `python binary/test.py` to get the average Dice score on the test set.
- **Inference & Explainability:** Run `python binary/predict.py` to evaluate a random test sample and generate a Grad-CAM heatmap visualization.

### Multi-class Segmentation

- **Training:** Run `python multiclass/train.py`. The training process implements `CrossEntropyDiceLoss`, custom per-class weights, and dynamic augmentations across epochs.
- **Evaluation:** Run `python multiclass/test.py`. Inference utilizes 4-way Test-Time Augmentation (TTA) and outputs mIoU and per-class metrics.
- **Prediction Notebook:** `multiclass/predict.ipynb` provides an interactive way to run predictions and visualize the multi-class outputs.

---

## Notes

- Large files, directories like `Data/`, `models/`, `saved_images/`, and cache folders are tracked in `.gitignore` and are not pushed to the repository. Ensure you populate the `Data` directory properly with `.jpg` images and corresponding `.png` masks before executing any scripts.
- The binary dataset relies on a `patching.py` and `Unpatching.py` strategy to split very large images, while multiclass resizes directly based on `DataLoading.py` parameters.
