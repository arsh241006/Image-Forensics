import cv2
import numpy as np
import base64


def compute_fft_magnitude(img_gray):
    """
    Computes the 2D FFT of the image and returns the magnitude spectrum,
    log-scaled so it's actually visible (raw FFT values span a huge range).
    """
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return magnitude


def get_frequency_view(image_path):
    """
    Returns a base64-encoded PNG of the image's FFT magnitude spectrum,
    for display in the frontend's frequency panel.
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    magnitude = compute_fft_magnitude(img_gray)

    # normalize to 0-255 for display
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    success, buffer = cv2.imencode('.png', normalized)
    if not success:
        raise RuntimeError("Failed to encode frequency view image")

    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str