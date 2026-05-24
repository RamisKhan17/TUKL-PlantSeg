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
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_DIR   = "Data/images/test/"
MASK_DIR    = "Data/annotations/test/"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count()
PIN_MEMORY  = True
NUM_CLASSES = 116
BATCH_SIZE  = 1

# ── Normalization (reused for TTA de-normalize → re-normalize cycle) ──────────
normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

img_transform = A.Compose([normalize, ToTensorV2()])

# ── Dataset & DataLoader ──────────────────────────────────────────────────────
test_ds = PlantSegData(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=img_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# ── Model ─────────────────────────────────────────────────────────────────────
model = smp.FPN(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
).to(DEVICE)

load_checkpoint(torch.load("models/FPN_resnet50.pth.tar", map_location=DEVICE), model)
model.eval()

# ── Torchmetrics ──────────────────────────────────────────────────────────────
iou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average=None).to(DEVICE)
acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average=None).to(DEVICE)


# ── Per-class Dice helper ─────────────────────────────────────────────────────
def compute_multiclass_dice(pred, target, num_classes):
    pred   = pred.flatten()
    target = target.flatten()
    scores = []
    for cls in range(1, num_classes):   # skip background (class 0)
        p     = (pred == cls)
        t     = (target == cls)
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        scores.append((2 * inter + 1e-7) / (union + inter + 1e-7))
    return scores


# ── TTA transforms ────────────────────────────────────────────────────────────
tta_transforms = [
    A.Compose([], p=1.0),                       # identity
    A.Compose([A.HorizontalFlip(p=1.0)], p=1.0),
    A.Compose([A.VerticalFlip(p=1.0)],   p=1.0),
    A.Compose([A.Transpose(p=1.0)],      p=1.0),
]


def reverse_tta(pred, transform):
    pred = pred.clone()
    if any(isinstance(t, A.HorizontalFlip) for t in transform.transforms):
        pred = torch.flip(pred, dims=[-1])
    if any(isinstance(t, A.VerticalFlip) for t in transform.transforms):
        pred = torch.flip(pred, dims=[-2])
    if any(isinstance(t, A.Transpose) for t in transform.transforms):
        pred = pred.transpose(-1, -2)
    return pred


# ── Inference with TTA ────────────────────────────────────────────────────────
all_dice = []
with torch.no_grad():
    for images, masks in tqdm(test_loader):
        b, c, h, w  = images.shape
        logits_sum  = torch.zeros((b, NUM_CLASSES, h, w), device=DEVICE)

        for tta_transform in tta_transforms:
            tta_imgs = []
            for img in images.cpu():
                # Denormalize → apply TTA → renormalize
                img_np = img.permute(1, 2, 0).numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                augmented  = tta_transform(image=img_np)["image"]
                normalized = normalize(image=augmented)["image"]
                tensor     = ToTensorV2()(image=normalized)["image"]
                tta_imgs.append(tensor)

            tta_imgs   = torch.stack(tta_imgs).to(DEVICE)
            tta_logits = model(tta_imgs)

            for i in range(b):
                tta_logits[i] = reverse_tta(tta_logits[i], tta_transform)
            logits_sum += tta_logits

        logits_avg = logits_sum / len(tta_transforms)
        preds      = torch.argmax(logits_avg, dim=1)   # [B, H, W]

        flat_p = preds.view(-1).to(iou_metric.device)
        flat_t = masks.view(-1).to(iou_metric.device)
        iou_metric.update(flat_p, flat_t)
        acc_metric.update(flat_p, flat_t)

        for i in range(images.shape[0]):
            all_dice.append(
                compute_multiclass_dice(preds[i].cpu().numpy(), masks[i].cpu().numpy(), NUM_CLASSES)
            )

# ── Aggregate ─────────────────────────────────────────────────────────────────
dice_arr      = np.array(all_dice)
mean_dice_cls = dice_arr.mean(axis=0)
mean_dice     = mean_dice_cls.mean()

iou_vals  = iou_metric.compute().cpu().numpy()
acc_vals  = acc_metric.compute().cpu().numpy()
mean_iou  = iou_vals[1:].mean()
mean_acc  = acc_vals[1:].mean()

print(f"\nAverage Dice Score (excl. background): {mean_dice:.4f}")
print(f"Mean IoU             (excl. background): {mean_iou:.4f}")
print(f"Mean Accuracy        (excl. background): {mean_acc * 100:.2f}%")
