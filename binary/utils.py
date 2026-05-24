import torch
import os
from torch.utils.data import DataLoader
from DataLoading import PlantSegData
from sklearn.metrics import f1_score


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
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_ds = PlantSegData(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda", threshold=0.5):
    model.eval()
    f1_scores = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  # (B, 1, H, W)
            preds = torch.sigmoid(model(x))
            preds = (preds > threshold).float()

            preds = preds.view(-1).cpu().numpy()
            targets = y.view(-1).cpu().numpy()

            score = f1_score(targets, preds, average='binary', zero_division=1)
            f1_scores.append(score)

    avg_f1 = sum(f1_scores) / len(f1_scores)
    print(f"Avg Sklearn Dice (F1) score: {avg_f1:.4f}")
    model.train()
    return avg_f1
