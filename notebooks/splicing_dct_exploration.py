import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_block_dct_energy_map(image_path, block_size=8):
    """
    For each 8x8 block in the image, computes the DCT and summarizes
    it as a single 'energy' value (sum of absolute coefficients,
    excluding the DC term). Regions with a different compression
    history tend to show a different energy distribution than their
    surroundings — this builds directly on extract_dct_features()
    from Week 2, but keeps a per-block summary instead of collapsing
    straight to a single feature map for a CNN.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    h = h - (h % block_size)
    w = w - (w % block_size)
    img = img[:h, :w]

    energy_map = np.zeros((h // block_size, w // block_size), dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size].astype(np.float32)
            dct_block = cv2.dct(block)
            # exclude the DC term [0,0] — it's dominated by average
            # brightness, not compression artifacts
            ac_energy = np.sum(np.abs(dct_block)) - np.abs(dct_block[0, 0])
            energy_map[i // block_size, j // block_size] = ac_energy

    return energy_map
    # --- Step 3: run it on a real example and visualize ---
energy_map = compute_block_dct_energy_map('data/CASIA2/Tp/Tp_D_CND_S_N_ind00078_ind00077_00476.tif')

plt.figure(figsize=(8, 6))
plt.imshow(energy_map, cmap='jet')
plt.colorbar(label='DCT AC energy')
plt.title("Block-wise DCT energy map (splicing example)")
plt.savefig('notebooks/splicing_dct_energy_map.png')
plt.show()