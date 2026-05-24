import torch
from tqdm import tqdm
import torch.nn as nn
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from utils import *
import os
import wandb
import numpy as np

# ── Hyperparameters ───────────────────────────────────────────────────────────
LEARNING_RATE    = 1e-4
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE       = 16
EPOCHS           = 10
NUM_WORKERS      = os.cpu_count()
IMAGE_HEIGHT     = 512
IMAGE_WIDTH      = 512
PIN_MEMORY       = True
LOAD_MODEL       = False
LOSS_CONTROLLER  = 0.3

# ── Data paths (relative to Model-Training root) ──────────────────────────────
TRAINING_IMG_DIR  = "Data/images/train/"
TRAINING_MASK_DIR = "Data/annotations/train/"
VAL_IMG_DIR       = "Data/images/val/"
VAL_MASK_DIR      = "Data/annotations/val/"


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)

        preds = torch.sigmoid(preds)
        smooth = 1e-7

        # Compute Dice per sample
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        return LOSS_CONTROLLER * bce_loss + (1 - LOSS_CONTROLLER) * dice_loss


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, wandb_run):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data    = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward
        with torch.amp.autocast("cuda"):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Binary predictions
        preds = torch.sigmoid(predictions)
        preds = (preds > 0.5).float()

        # Accuracy
        correct  = (preds == targets).float().sum()
        total    = torch.numel(preds)
        accuracy = correct / total

        # Dice score
        smooth        = 1e-7
        intersection  = (preds * targets).sum(dim=(2, 3))
        union         = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score    = ((2 * intersection + smooth) / (union + smooth)).mean()

        # Log to wandb
        step = epoch * len(loader) + batch_idx
        wandb_run.log({
            "train/loss":       loss.item(),
            "train/accuracy":   accuracy.item(),
            "train/dice_score": dice_score.item(),
        }, step=step)
        loop.set_postfix({
            "loss": loss.item(),
            "acc":  accuracy.item(),
            "dice": dice_score.item()
        })


def main():
    global LOSS_CONTROLLER
    train_transforms = A.Compose([
        A.RandomRotate90(p=0.3),
        A.VerticalFlip(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=1, border_mode=0),
            A.ElasticTransform(alpha=50, sigma=6, p=1, border_mode=0),
        ], p=0.25),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(transpose_mask=True)
    ])
    val_transforms = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    model = smp.DeepLabV3Plus(
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),          "lr": LEARNING_RATE * 0.75},
        {"params": model.decoder.parameters(),          "lr": LEARNING_RATE},
        {"params": model.segmentation_head.parameters(),"lr": LEARNING_RATE},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    train_loader, val_loader = get_loaders(
        TRAINING_IMG_DIR, TRAINING_MASK_DIR,
        VAL_IMG_DIR,      VAL_MASK_DIR,
        BATCH_SIZE, train_transforms, val_transforms
    )
    scaler = torch.amp.GradScaler(device=DEVICE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("models/my_checkpoint.pth.tar", weights_only=False), model)

    wandb_run = wandb.init(
        project="PlantSegBin",
        name="Timm-effnet-b3",
        config={
            "Learning_rate": LEARNING_RATE,
            "Architecture":  "DeepLabV3+",
            "Encoder":       "timm-efficientnet-b3",
            "Batch-size":    BATCH_SIZE,
            "Dataset":       "PlantSeg",
            "Epochs":        EPOCHS,
            "Loss":          "BCEDiceLoss",
            "Optimizer":     "AdamW",
            "Scheduler":     "CosineAnnealing",
        },
    )
    print(f"WandB run: {wandb_run.get_url()}")
    model.train()
    for epoch in range(EPOCHS):
        train_fn(loader=train_loader, model=model, optimizer=optimizer,
                 loss_fn=loss_fn, scaler=scaler, epoch=epoch, wandb_run=wandb_run)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Check accuracy
        dice = check_accuracy(loader=val_loader, model=model, device=DEVICE, threshold=0.5)
        wandb_run.log({"val/dice_score": dice})
        scheduler.step()


if __name__ == "__main__":
    main()
