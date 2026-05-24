import numpy as np
import os
import math
from PIL import Image, ImageOps
import random
import matplotlib.pyplot as plt

# ------------ Config (EDIT THESE) ------------
output_img_dir  = "patchedData/test/images/"
output_mask_dir = "patchedData/test/masks/"
tile_size = 224

def manual_unpatchify(patches, rows, cols, tile_size, is_mask=False):
    if is_mask:
        full = np.zeros((rows * tile_size, cols * tile_size), dtype=np.uint8)
    else:
        full = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
    
    for r in range(rows):
        for c in range(cols):
            if is_mask:
                full[r * tile_size:(r + 1) * tile_size,
                     c * tile_size:(c + 1) * tile_size] = patches[r, c]
            else:
                full[r * tile_size:(r + 1) * tile_size,
                     c * tile_size:(c + 1) * tile_size, :] = patches[r, c]
    
    return full

def unpatchImages(image_dir: str, mask_dir: str, initial: str, tilesize=224):
    image_files = sorted([f for f in os.listdir(image_dir) if f.startswith(initial + "_")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.startswith(initial + "_")])

    if not image_files or not mask_files:
        raise ValueError(f"No files found with initial '{initial}' in provided directories.")

    max_row, max_col = 0, 0
    for fname in image_files:
        parts = os.path.splitext(fname)[0].split('_')
        row = int(parts[-2])
        col = int(parts[-1])
        max_row = max(max_row, row)
        max_col = max(max_col, col)

    rows = max_row + 1
    cols = max_col + 1

    image_patch_array = np.zeros((rows, cols, tilesize, tilesize, 3), dtype=np.uint8)
    mask_patch_array = np.zeros((rows, cols, tilesize, tilesize), dtype=np.uint8)

    for img_file, mask_file in zip(image_files, mask_files):
        name_parts = os.path.splitext(img_file)[0].split('_')
        row = int(name_parts[-2])
        col = int(name_parts[-1])

        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image_patch_array[row, col] = np.array(Image.open(img_path))
        mask_patch_array[row, col] = np.array(Image.open(mask_path))

    print(f"🔁 Unpatching '{initial}'")
    print("image_patch_array.shape:", image_patch_array.shape)

    reconstructed_image = manual_unpatchify(image_patch_array, rows, cols, tilesize)
    reconstructed_mask = manual_unpatchify(mask_patch_array, rows, cols, tilesize, is_mask=True)

    return Image.fromarray(reconstructed_image), Image.fromarray(reconstructed_mask)

img, mask = unpatchImages(output_img_dir, output_mask_dir, "62")

plt.figure(figsize=(8, 4))
# Reconstructed Image
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title(f"Reconstructed Image ({img.size[1]}×{img.size[0]})")
plt.axis("off")

# Reconstructed Mask
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title(f"Reconstructed Mask ({mask.size[1]}×{mask.size[0]})")
plt.axis("off")

plt.show()
