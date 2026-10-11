import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from spatial_branch.model import build_spatial_model
from spatial_branch.ela_transform import compute_ela
from fusion.gradcam import GradCAM, overlay_heatmap

model = build_spatial_model(feature_dim=128, variant='ela')
model.load_state_dict(torch.load('spatial_branch/baseline_model_ela.pt', map_location='cpu'))

target_layer = model.backbone[-2][-1]
gradcam = GradCAM(model, target_layer)

# authentic version of the SAME scene as 13t.tif, for a fair comparison —
# COVERAGE's authentic images drop the 't' suffix: 13t.tif -> 13.tif
img_path = 'data/COVERAGE/image/13.tif'

from preprocessing import load_and_preprocess
img = load_and_preprocess(img_path)
ela = compute_ela(img_path)

img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
ela_tensor = torch.tensor(ela, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

heatmap = gradcam.generate_heatmap(img_tensor, ela_tensor, target_class=1)
print("Heatmap min/max:", heatmap.min(), heatmap.max())
print("Heatmap values:\n", np.round(heatmap, 2))

display_img = cv2.imread(img_path)
display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
display_img = cv2.resize(display_img, (224, 224))

overlaid = overlay_heatmap(display_img, heatmap)

plt.imshow(overlaid)
plt.title("Grad-CAM on AUTHENTIC 13.tif (no real tampering)")
plt.axis('off')
plt.savefig('notebooks/debug_authentic_13.png')
print("Saved to notebooks/debug_authentic_13.png")