import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

from preprocessing import load_and_preprocess
from spatial_branch.model import build_spatial_model
from spatial_branch.ela_transform import compute_ela
from fusion.gradcam import GradCAM, overlay_heatmap


model = build_spatial_model(feature_dim=128, variant="ela")
model.load_state_dict(torch.load(
    ROOT / "spatial_branch/baseline_model_ela.pt",
    map_location="cpu",
    weights_only=True,
))
model.eval()

gradcam = GradCAM(model, model.backbone[-2][-1])

cases = [
    ("Best case", "5t.tif"),
    ("Zero-overlap case", "19t.tif"),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for row, (description, filename) in enumerate(cases):
    image_path = ROOT / "data/COVERAGE/image" / filename
    base_id = filename[:-len("t.tif")]
    mask_path = ROOT / "data/COVERAGE/mask" / f"{base_id}forged.tif"

    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(image_path)

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (224, 224))

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    mask = cv2.resize(
        mask, (224, 224), interpolation=cv2.INTER_NEAREST
    )
    mask = mask > 127

    rgb = load_and_preprocess(str(image_path))
    ela = compute_ela(str(image_path))

    rgb_tensor = torch.tensor(
        rgb, dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0)

    ela_tensor = torch.tensor(
        ela, dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0)

    # Grad-CAM requires gradients; do not use torch.no_grad().
    heatmap = gradcam.generate_heatmap(
        rgb_tensor, ela_tensor, target_class=1
    )
    resized = cv2.resize(
        heatmap, (224, 224), interpolation=cv2.INTER_LINEAR
    )
    predicted = resized >= 0.5

    intersection = np.logical_and(predicted, mask).sum()
    union = np.logical_or(predicted, mask).sum()
    iou = float(intersection / union) if union else 0.0

    axes[row, 0].imshow(original)
    axes[row, 0].set_title(f"{description}: {filename}")

    axes[row, 1].imshow(overlay_heatmap(original, heatmap))
    axes[row, 1].set_title(f"Grad-CAM overlay | IoU {iou:.3f}")

    axes[row, 2].imshow(predicted, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].set_title("Thresholded heatmap (0.5)")

    axes[row, 3].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[row, 3].set_title("Ground-truth mask")

    for ax in axes[row]:
        ax.axis("off")

    print(f"{filename}: IoU = {iou:.3f}")

fig.suptitle(
    "COVERAGE localization: best and zero-overlap examples\n"
    "Spatial ELA classifier; tampered-class Grad-CAM",
    fontsize=14,
)
fig.tight_layout(rect=(0, 0, 1, 0.92))

output = ROOT / "notebooks/day5_best_worst.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output}")