import torch
import segmentation_models_pytorch as smp
import utils
import os
import random
from PIL import Image, ImageOps
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

def compute_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold=0.5, eps=1e-6):
    pred_bin = (pred_mask > threshold).float()
    if true_mask.ndim == 2:
        true_mask = true_mask.unsqueeze(0).unsqueeze(0)
    elif true_mask.ndim == 3:
        true_mask = true_mask.unsqueeze(0)
    intersection = (pred_bin * true_mask).sum()
    union = (pred_bin + true_mask - pred_bin * true_mask).sum()
    iou = (intersection + eps) / (union + eps)
    return iou.item()

def pad_to_multiple_of_32(img):
    img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    w, h = img.size
    pad_w = (32 - w % 32) % 32
    pad_h = (32 - h % 32) % 32
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    img = ImageOps.expand(img, padding, fill=0)
    return np.array(img)

# Load model
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

checkpoint = torch.load("models/resnet50_v1.pth.tar", map_location=DEVICE)
utils.load_checkpoint(checkpoint, model=model)

# Select random test sample
seed = random.randint(0, 2294)
image_dir = "Data/images/test/"
mask_dir = "Data/annotations/test/"
images = os.listdir(image_dir)
masks = os.listdir(mask_dir)
image_path = os.path.join(image_dir, images[seed])
mask_path = os.path.join(mask_dir, masks[seed])

# Define transform
transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0),
    ToTensorV2()
])

# Load and preprocess image/mask
image = pad_to_multiple_of_32(np.array(Image.open(image_path).convert("RGB")))
mask = pad_to_multiple_of_32(np.array(Image.open(mask_path).convert("L"))).astype(np.float32)
mask[mask > 1.0] = 1.0  # Ensure binary

augmented = transform(image=image, mask=mask)
image_tensor = augmented["image"].to(DEVICE)
mask_tensor = torch.tensor(augmented["mask"], dtype=torch.float32).to(DEVICE)

# Run inference
model.eval()
with torch.no_grad():
    pred = torch.sigmoid(model(image_tensor.unsqueeze(0)))
predicted_mask = (pred > 0.5).float()

# Compute IoU
iou_score = compute_iou(pred, mask_tensor)
print(f"IoU: {iou_score:.4f}")

# Move back to CPU for visualization and CAM
model.to("cpu")
image_tensor_cpu = image_tensor.cpu()
predicted_mask_cpu = predicted_mask.cpu()

# Prepare Grad-CAM
target_layer = model.encoder.layer4[-1]
input_tensor = image_tensor_cpu.unsqueeze(0)
H, W = image_tensor.shape[1:]
cam_mask = np.ones((H, W), dtype=np.float32)
target_category = [SemanticSegmentationTarget(category=0, mask=cam_mask)]

cam = GradCAM(model=model, target_layers=[target_layer])
grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)[0]

# Prepare visualization
img_np = image_tensor_cpu.permute(1, 2, 0).numpy()
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
mask_np = mask / 1.0
pred_np = predicted_mask_cpu.squeeze().numpy()
visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

# Plot
plt.figure(figsize=(7, 7))

plt.subplot(2, 2, 1)
plt.imshow(img_np)
plt.title(f"Input Image\n{W}×{H}")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mask_np, cmap='gray')
plt.title(f"Ground Truth Mask\n{mask_np.shape[1]}×{mask_np.shape[0]}")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(pred_np, cmap='gray')
plt.title(f"Predicted Mask\n{pred_np.shape[1]}×{pred_np.shape[0]}")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(visualization)
plt.title("Grad-CAM on DeepLabV3+")
plt.axis('off')

plt.tight_layout()
plt.show()
