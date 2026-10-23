import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import cv2

from spatial_branch.model import build_spatial_model
from spatial_branch.ela_transform import compute_ela
from preprocessing import load_and_preprocess
from fusion.gradcam import GradCAM
from fusion.coverage_dataset import CoverageDataset


def compute_iou(heatmap, gt_mask, threshold=0.5):
    """
    heatmap: raw (7,7) Grad-CAM output, values 0-1
    gt_mask: ground-truth binary mask, already (224,224), values 0 or 1
    threshold: cutoff above which a heatmap pixel counts as "predicted tampered"
    """
    h, w = gt_mask.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))

    pred_binary = (heatmap_resized >= threshold).astype(np.uint8)
    gt_binary = gt_mask.astype(np.uint8)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        return 0.0

    return intersection / union


# --- Load model ONCE, reuse across all images (not the estimate_heatmap wrapper) ---
model = build_spatial_model(feature_dim=128, variant='ela')
model.load_state_dict(torch.load('spatial_branch/baseline_model_ela.pt', map_location='cpu'))

target_layer = model.backbone[-2][-1]
gradcam = GradCAM(model, target_layer)

# --- Load COVERAGE dataset ---
coverage_ds = CoverageDataset()
print(f"Running IoU validation on {len(coverage_ds)} COVERAGE images...")

iou_scores = []
results_log = []  # keep per-image results for picking good/weak examples later

for idx in range(len(coverage_ds)):
    img_tensor, mask_tensor, filename = coverage_ds[idx]

    # CoverageDataset gives preprocessed tensors already, but ELA needs
    # the raw file path — reconstruct it from the known pattern
    img_path = os.path.join(coverage_ds.image_dir, filename)
    # Match the shared single-image inference preprocessing.
    rgb = load_and_preprocess(img_path, size=coverage_ds.size)
    img_tensor = torch.from_numpy(
    np.asarray(rgb, dtype=np.float32)
    ).permute(2, 0, 1)
    ela = compute_ela(img_path)
    ela_tensor = torch.tensor(ela, dtype=torch.float32).permute(2, 0, 1)

    heatmap = gradcam.generate_heatmap(
        img_tensor.unsqueeze(0),
        ela_tensor.unsqueeze(0),
        target_class=1
    )

    gt_mask = mask_tensor.numpy()
    iou = compute_iou(heatmap, gt_mask, threshold=0.5)

    iou_scores.append(iou)
    results_log.append((filename, iou))

    if idx % 20 == 0:
        print(f"  Processed {idx}/{len(coverage_ds)} — running mean IoU: {np.mean(iou_scores):.3f}")

mean_iou = np.mean(iou_scores)
print(f"\n=== Mean IoU across {len(coverage_ds)} COVERAGE images: {mean_iou:.3f} ===")

results_log.sort(key=lambda x: x[1])
print("\n3 weakest results:")
for fname, score in results_log[:3]:
    print(f"  {fname}: IoU = {score:.3f}")

print("\n3 strongest results:")
for fname, score in results_log[-3:]:
    print(f"  {fname}: IoU = {score:.3f}")

import csv
from pathlib import Path

results_dir = Path("results/localization")
results_dir.mkdir(parents=True, exist_ok=True)

output_path = results_dir / "coverage_iou.csv"

with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "iou"])
    writer.writerows(results_log)

print(f"\nPer-image results saved to: {output_path}")