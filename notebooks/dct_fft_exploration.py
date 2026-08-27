import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_block_dct(img_gray, block_size=8):
    """
    Splits the image into 8x8 blocks (same size JPEG uses) and computes
    the DCT of each block. Returns an array of all DCT coefficients.
    """
    h, w = img_gray.shape
    # crop so height/width divide evenly into 8x8 blocks
    h = h - (h % block_size)
    w = w - (w % block_size)
    img_gray = img_gray[:h, :w]

    coeffs = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_gray[i:i+block_size, j:j+block_size].astype(np.float32)
            dct_block = cv2.dct(block)
            coeffs.append(dct_block)
    return np.array(coeffs)

# load both images in grayscale — DCT here is computed on luminance, not color
authentic_img = cv2.imread('data/CASIA2/Au/Au_ani_00024.jpg', cv2.IMREAD_GRAYSCALE)
tampered_img = cv2.imread('data/CASIA2/Tp/Tp_D_CNN_M_N_arc00086_arc00086_00306.tif', cv2.IMREAD_GRAYSCALE)

auth_coeffs = compute_block_dct(authentic_img)
tamp_coeffs = compute_block_dct(tampered_img)

# flatten and plot histograms of DCT coefficient values side by side
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(auth_coeffs.flatten(), bins=100, range=(-50, 50))
plt.title("Authentic — DCT coefficient histogram")
plt.subplot(1, 2, 2)
plt.hist(tamp_coeffs.flatten(), bins=100, range=(-50, 50))
plt.title("Tampered — DCT coefficient histogram")
plt.tight_layout()
plt.savefig('notebooks/dct_comparison.png')
plt.show()



def compute_fft_magnitude(img_gray):
    """
    Computes the 2D FFT of the image and returns the magnitude spectrum,
    log-scaled so it's actually visible (raw FFT values span a huge range).
    """
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)  # move the zero-frequency component to the center
    magnitude = np.log(np.abs(fshift) + 1)  # log scale for visibility, +1 avoids log(0)
    return magnitude

auth_fft = compute_fft_magnitude(authentic_img)
tamp_fft = compute_fft_magnitude(tampered_img)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(auth_fft, cmap='gray')
plt.title("Authentic — FFT magnitude spectrum")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(tamp_fft, cmap='gray')
plt.title("Tampered — FFT magnitude spectrum")
plt.axis('off')
plt.tight_layout()
plt.savefig('notebooks/fft_comparison.png')
plt.show()

