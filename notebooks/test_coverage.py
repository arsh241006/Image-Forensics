import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fusion.coverage_dataset import CoverageDataset

ds = CoverageDataset()
print("Number of tampered image/mask pairs found:", len(ds))

img, mask, filename = ds[0]
print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
print("Mask unique values:", mask.unique())
print("Filename:", filename)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img.permute(1, 2, 0))  # back to (H, W, C) for display
plt.title(f"Image: {filename}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("Ground-truth tampering mask")
plt.axis('off')

plt.tight_layout()
plt.savefig('notebooks/coverage_sanity_check.png')
plt.show()