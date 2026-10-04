import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from preprocessing import load_and_preprocess
from spatial_branch.ela_transform import compute_ela
from spatial_branch.model import build_spatial_model
from fusion.gradcam import GradCAM, overlay_heatmap

model = build_spatial_model(feature_dim=128, variant='ela')
model.load_state_dict(torch.load('spatial_branch/baseline_model_ela.pt', map_location='cpu'))

target_layer = model.backbone[-2][-1]
gradcam = GradCAM(model, target_layer)

img_path = 'data/CASIA2/Tp/Tp_D_CND_S_N_txt00028_txt00006_10848.jpg'

img = load_and_preprocess(img_path)
ela = compute_ela(img_path)

img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
ela_tensor = torch.tensor(ela, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

heatmap = gradcam.generate_heatmap(img_tensor, ela_tensor, target_class=1)

display_img = cv2.imread(img_path)
display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
display_img = cv2.resize(display_img, (224, 224))

overlaid = overlay_heatmap(display_img, heatmap)

plt.imshow(overlaid)
plt.title("Grad-CAM heatmap overlay")
plt.axis('off')
plt.savefig('notebooks/gradcam_test.png')

print("Heatmap saved to notebooks/gradcam_test.png")