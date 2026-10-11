import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from spatial_branch.model import build_spatial_model
from spatial_branch.ela_transform import compute_ela
from fusion.gradcam import GradCAM, overlay_heatmap
from fusion.coverage_dataset import CoverageDataset

model = build_spatial_model(feature_dim=128, variant='ela')
model.load_state_dict(torch.load('spatial_branch/baseline_model_ela.pt', map_location='cpu'))

target_layer = model.backbone[-2][-1]
gradcam = GradCAM(model, target_layer)

coverage_ds = CoverageDataset()

# find the index for 13t.tif specifically
target_filename = '13t.tif'
idx = None
for i in range(len(coverage_ds)):
    _, _, fname = coverage_ds[i]
    if fname == target_filename:
        idx = i
        break

print(f"Found {target_filename} at index {idx}")

img_tensor, mask_tensor, filename = coverage_ds[idx]
img_path = os.path.join(coverage_ds.image_dir, filename)

ela = compute_ela(img_path)
ela_tensor = torch.tensor(ela, dtype=torch.float32).permute(2, 0, 1)

heatmap = gradcam.generate_heatmap(
    img_tensor.unsqueeze(0), ela_tensor.unsqueeze(0), target_class=1
)
print("Heatmap min/max:", heatmap.min(), heatmap.max())
print("Heatmap values:\n", np.round(heatmap, 2))

gt_mask = mask_tensor.numpy()
print("\nGT mask unique values:", np.unique(gt_mask))
print("GT mask sum (tampered pixel count):", gt_mask.sum())

# display version of the original image
display_img = cv2.imread(img_path)
display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
display_img = cv2.resize(display_img, (224, 224))

overlaid = overlay_heatmap(display_img, heatmap)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(display_img)
axes[0].set_title(f"Original: {filename}")
axes[0].axis('off')

axes[1].imshow(overlaid)
axes[1].set_title("Grad-CAM heatmap")
axes[1].axis('off')

axes[2].imshow(gt_mask, cmap='gray')
axes[2].set_title("Ground-truth mask")
axes[2].axis('off')

plt.tight_layout()
plt.savefig('notebooks/debug_iou_13t.png')
print("\nSaved to notebooks/debug_iou_13t.png")