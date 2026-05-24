import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from utils import *
import os
import wandb

# ── Hyperparameters ───────────────────────────────────────────────────────────
LEARNING_RATE    = 5e-5
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE       = 8
EPOCHS           = 20
NUM_WORKERS      = os.cpu_count()
PIN_MEMORY       = True
LOAD_MODEL       = False
LOSS_CONTROLLER  = 0.3
NUM_CLASSES      = 116

# ── Data paths (relative to Model-Training root) ──────────────────────────────
TRAINING_IMG_DIR  = "Data/images/train/"
TRAINING_MASK_DIR = "Data/annotations/train/"
VAL_IMG_DIR       = "Data/images/val/"
VAL_MASK_DIR      = "Data/annotations/val/"

# Class weights — downweight background relative to foreground classes
class_weights = torch.ones(NUM_CLASSES)
class_weights[0] = 1.0 / (NUM_CLASSES ** 0.5)


def get_dynamic_aug(epoch, total_epochs):
    p_rotate  = max(0.1, 0.3 * (1 - epoch / total_epochs))
    p_flip_h  = max(0.1, 0.3 * (1 - epoch / total_epochs))
    p_bright  = max(0.05, 0.2 * (1 - epoch / total_epochs))

    return A.Compose([
        A.RandomRotate90(p=p_rotate),
        A.HorizontalFlip(p=p_flip_h),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(p=p_bright),
        A.HueSaturationValue(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(transpose_mask=True)
    ])


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_controller=0.5, weight=None):
        super().__init__()
        self.num_classes      = num_classes
        self.loss_controller  = loss_controller
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds, targets):
        # preds:   [B, C, H, W]  — raw logits
        # targets: [B, H, W]     — class indices (0 to C-1)
        ce_loss = self.ce(preds, targets)

        # One-hot encode targets → [B, C, H, W]
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)   # [B, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        probs = F.softmax(preds, dim=1)

        dims         = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality  = torch.sum(probs + targets_onehot, dims)
        dice_score   = (2.0 * intersection + 1e-7) / (cardinality + 1e-7)
        dice_loss    = 1 - dice_score.mean()

        return self.loss_controller * ce_loss + (1 - self.loss_controller) * dice_loss


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, wandb_run):
    loop = tqdm(loader)

    acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average=None, ignore_index=None).to(DEVICE)
    iou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average=None, ignore_index=None).to(DEVICE)

    for batch_idx, (data, targets) in enumerate(loop):
        step    = epoch * len(loader) + batch_idx
        data    = data.to(DEVICE)
        targets = targets.to(DEVICE).long()

        with torch.amp.autocast(DEVICE):
            predictions = model(data)            # [B, C, H, W]
            loss        = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred_labels = torch.argmax(predictions.detach(), dim=1)  # [B, H, W]
        gt          = targets.squeeze(1)                          # [B, H, W]

        acc_metric.update(pred_labels, gt)
        iou_metric.update(pred_labels, gt)

        acc_vals = acc_metric.compute().detach().cpu().numpy()
        iou_vals = iou_metric.compute().detach().cpu().numpy()
        mAcc     = acc_vals[1:].mean() * 100
        mIoU     = iou_vals[1:].mean()

        wandb_run.log({"train/loss": loss.item(), "train/mAcc": mAcc, "train/mIoU": mIoU}, step=step)
        loop.set_postfix({"loss": loss.item(), "mAcc": mAcc})

    acc_metric.reset()
    iou_metric.reset()


def main():
    val_transforms = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    model = smp.DeepLabV3Plus(
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    # Move class_weights to the same device as the model before creating the loss
    loss_fn = CrossEntropyDiceLoss(
        num_classes=NUM_CLASSES,
        loss_controller=LOSS_CONTROLLER,
        weight=class_weights.to(DEVICE)   # ← device-safe fix
    )

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),           "lr": LEARNING_RATE * 0.75},
        {"params": model.decoder.parameters(),           "lr": LEARNING_RATE},
        {"params": model.segmentation_head.parameters(), "lr": LEARNING_RATE},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)

    _, val_loader = get_loaders(
        TRAINING_IMG_DIR, TRAINING_MASK_DIR,
        VAL_IMG_DIR,      VAL_MASK_DIR,
        BATCH_SIZE,
        get_dynamic_aug(0, EPOCHS),   # placeholder — replaced each epoch
        val_transforms
    )

    scaler = torch.amp.GradScaler(device=DEVICE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("models/my_checkpoint.pth.tar", weights_only=False), model)

    wandb_run = wandb.init(
        project="PlantSegBin",
        name="Multiclass-effnet-b3",
        config={
            "Learning_rate": LEARNING_RATE,
            "Architecture":  "DeepLabV3+",
            "Encoder":       "timm-efficientnet-b3",
            "Batch-size":    BATCH_SIZE,
            "Dataset":       "PlantSeg",
            "Epochs":        EPOCHS,
            "Num_classes":   NUM_CLASSES,
            "Loss":          "CrossEntropyDiceLoss",
            "Optimizer":     "AdamW",
            "Scheduler":     "CosineAnnealing",
        },
    )
    print(f"WandB run: {wandb_run.get_url()}")

    model.train()
    for epoch in range(EPOCHS):
        # Rebuild train loader each epoch with dynamic augmentation schedule
        train_transforms = get_dynamic_aug(epoch, EPOCHS)
        train_loader, _ = get_loaders(
            TRAINING_IMG_DIR, TRAINING_MASK_DIR,
            VAL_IMG_DIR,      VAL_MASK_DIR,
            BATCH_SIZE, train_transforms, val_transforms
        )
        train_fn(loader=train_loader, model=model, optimizer=optimizer,
                 loss_fn=loss_fn, scaler=scaler, epoch=epoch, wandb_run=wandb_run)

        val_acc, val_miou = check_accuracy(loader=val_loader, model=model, device=DEVICE)
        wandb_run.log({"val/accuracy": val_acc, "val/mIoU": val_miou})
        scheduler.step()

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)


if __name__ == "__main__":
    main()
