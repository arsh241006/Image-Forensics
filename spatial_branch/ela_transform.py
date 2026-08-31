import numpy as np
from PIL import Image
import io
 
def compute_ela(image_path, quality=90, scale=15, size=224):
    """
    Shared ELA extraction function — takes an image filepath and returns
    a (size, size, 3) ELA array, ready to be used alongside the raw RGB
    image as an auxiliary input to the spatial CNN.
    """
    original = Image.open(image_path).convert('RGB')
 
    buffer = io.BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
 
    original_arr = np.array(original).astype(np.int16)
    resaved_arr = np.array(resaved).astype(np.int16)
    diff = np.abs(original_arr - resaved_arr)
 
    ela_image = np.clip(diff * scale, 0, 255).astype(np.uint8)
 
    # resize to match the model's expected input size
    ela_pil = Image.fromarray(ela_image).resize((size, size))
    ela_array = np.array(ela_pil).astype(np.float32) / 255.0  # scale to 0-1
 
    return ela_array  # shape: (224, 224, 3)
