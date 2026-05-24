import torch
import os
from torch.utils.data import DataLoader
from DataLoading import PlantSegData
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex


def save_checkpoint(state, filename="models/my_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=os.cpu_count(),
    pin_memory=True
):
    train_ds = PlantSegData(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)
    val_ds = PlantSegData(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda", num_classes=116, ignore_index=0):
    model.eval()

    acc_metric = MulticlassAccuracy(
        num_classes=num_classes,
        average=None,
        ignore_index=None
    ).to(device)

    iou_metric = MulticlassJaccardIndex(
        num_classes=num_classes,
        average=None,
        ignore_index=None
    ).to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.long().to(device).squeeze(1)  # [B, H, W]
            preds = model(x)
            pred_labels = torch.argmax(preds, dim=1)  # [B, H, W]
            acc_metric.update(pred_labels, y)
            iou_metric.update(pred_labels, y)

    per_class_acc = acc_metric.compute()   # Shape: [num_classes]
    per_class_iou = iou_metric.compute()   # Shape: [num_classes]

    # Exclude background class (index 0)
    foreground_acc = per_class_acc[1:]
    foreground_iou = per_class_iou[1:]

    mean_acc = foreground_acc.mean() * 100
    mean_iou = foreground_iou.mean()

    print(f"mAcc (foreground only): {mean_acc:.2f}% | mIoU (foreground only): {mean_iou:.4f}")
    model.train()
    return mean_acc.item(), mean_iou.item()
