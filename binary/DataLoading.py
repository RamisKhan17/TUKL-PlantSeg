import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import numpy as np


def resize_and_pad(img, size=(512, 512), is_mask=False):
    img = Image.fromarray(img)
    interp = Image.BILINEAR if not is_mask else Image.NEAREST

    # Resize while keeping aspect ratio
    img.thumbnail(size, interp)  # Note: thumbnail modifies in-place

    # Calculate required padding
    delta_w = size[0] - img.size[0]
    delta_h = size[1] - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)

    # Pad and force final resize to fix any rounding issues
    img = ImageOps.expand(img, padding, fill=0)
    img = img.resize(size, interp)  # <-- Final hard resize

    return np.array(img)


class PlantSegData(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        image = resize_and_pad(np.array(Image.open(img_path).convert("RGB")), is_mask=False)
        mask = resize_and_pad(np.array(Image.open(mask_path).convert("L")).astype(np.float32), is_mask=True)
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask
