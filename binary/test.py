import os
import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from utils import load_checkpoint
from DataLoading import PlantSegData
from torch.utils.data import DataLoader

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_DIR   = "Data/images/test/"
MASK_DIR    = "Data/annotations/test/"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count()
PIN_MEMORY  = True

# ── Transform ─────────────────────────────────────────────────────────────────
test_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ── Dataset & DataLoader ──────────────────────────────────────────────────────
test_ds = PlantSegData(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# ── Model ─────────────────────────────────────────────────────────────────────
model = smp.DeepLabV3Plus(
    encoder_name="timm-efficientnet-b3",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

load_checkpoint(torch.load("models/my_checkpoint.pth.tar", weights_only=False), model)
model.eval()


# ── Dice helper ───────────────────────────────────────────────────────────────
def compute_dice(pred, mask):
    intersection = np.logical_and(pred, mask).sum()
    union        = np.logical_or(pred, mask).sum()
    dice         = (2 * intersection + 1e-7) / (union + intersection + 1e-7)
    return dice


# ── Inference loop ────────────────────────────────────────────────────────────
dice_scores = []
with torch.no_grad():
    for image, mask in tqdm(test_loader):
        image   = image.to(DEVICE)
        pred    = torch.sigmoid(model(image)).squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        bin_pred = (pred > 0.5).astype(np.float32)
        dice_scores.append(compute_dice(bin_pred, mask_np))

print(f"\nAverage Dice Score: {np.mean(dice_scores):.4f}")
