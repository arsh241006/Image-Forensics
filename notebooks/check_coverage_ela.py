import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from spatial_branch.ela_transform import compute_ela

filenames = ["13.tif", "13t.tif", "69t.tif"]
image_dir = ROOT / "data/COVERAGE/image"

fig, axes = plt.subplots(3, 2, figsize=(9, 11))

for row, filename in enumerate(filenames):
    path = image_dir / filename

    with Image.open(path) as image:
        print(f"{filename}: format={image.format}, size={image.size}")
        axes[row, 0].imshow(image.convert("RGB"))

    ela = compute_ela(str(path))
    axes[row, 1].imshow(ela)

    axes[row, 0].set_title(f"Original: {filename}")
    axes[row, 1].set_title("ELA: JPEG quality 90, scale 15")

    for ax in axes[row]:
        ax.axis("off")

output = ROOT / "notebooks/coverage_ela_check.png"
fig.tight_layout()
fig.savefig(output, dpi=160)
plt.close(fig)

print(f"Saved: {output}")