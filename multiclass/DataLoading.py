import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import numpy as np


def resize_and_pad(img, size=(512, 512), is_mask=False):
    img = Image.fromarray(img)
    interp = Image.NEAREST if is_mask else Image.BILINEAR
    img.thumbnail(size, interp)
    delta_w = size[1] - img.size[0]
    delta_h = size[0] - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    img = ImageOps.expand(img, padding, fill=0)
    return np.array(img)


class PlantSegData(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = resize_and_pad(np.array(Image.open(img_path).convert("RGB")), is_mask=False)
        mask = resize_and_pad(np.array(Image.open(mask_path).convert("L")), is_mask=True).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long().unsqueeze(0)  # shape [1, H, W]

        return image, mask
