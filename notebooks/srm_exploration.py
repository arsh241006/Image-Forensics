import cv2
import numpy as np
import matplotlib.pyplot as plt

# one standard SRM high-pass kernel (3x3 "SQUARE3x3" residual filter,
# commonly used as a simple, effective starting point)
SRM_KERNEL = np.array([
    [-1, 2, -1],
    [ 2, -4, 2],
    [-1, 2, -1]
], dtype=np.float32) / 4.0

def apply_srm_filter(img_gray):
    """
    Applies the SRM high-pass filter to a grayscale image, suppressing
    normal image content and amplifying noise-residual patterns.
    """
    filtered = cv2.filter2D(img_gray.astype(np.float32), -1, SRM_KERNEL)
    return filtered

authentic_img = cv2.imread('data/CASIA2/Au/Au_ani_00024.jpg', cv2.IMREAD_GRAYSCALE)
tampered_img = cv2.imread('data/CASIA2/Tp/Tp_D_CNN_M_N_arc00086_arc00086_00306.tif', cv2.IMREAD_GRAYSCALE)

auth_srm = apply_srm_filter(authentic_img)
tamp_srm = apply_srm_filter(tampered_img)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(auth_srm, cmap='gray')
plt.title("Authentic — SRM filtered")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(tamp_srm, cmap='gray')
plt.title("Tampered — SRM filtered")
plt.axis('off')
plt.tight_layout()
plt.savefig('notebooks/srm_comparison.png')
plt.show()