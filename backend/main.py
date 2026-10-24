import os
import tempfile
import base64

import cv2
import torch

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline_utils import prepare_inputs, predict_from_inputs, pipeline
from fusion.manipulation_type import estimate_manipulation_type
from fusion.gradcam import GradCAM, overlay_heatmap
from backend.frequency_view import get_frequency_view


app = FastAPI(
    title="Image Forensics API",
    description="Backend API for detecting image tampering",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

target_layer = pipeline.spatial_model.backbone[-2][-1]
gradcam = GradCAM(pipeline.spatial_model, target_layer)


def encode_image_array(img_array):
    """Encodes a raw RGB numpy image array as a base64 PNG string."""
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode('.png', img_bgr)
    if not success:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/")
def root():
    return {"message": "Image Forensics API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    warnings = []

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        inputs = prepare_inputs(tmp_path)
        predicted_class, tampered_probability = predict_from_inputs(inputs)
        label = "tampered" if predicted_class == 1 else "authentic"

        if 0.4 <= tampered_probability <= 0.6:
            warnings.append("Model confidence is low for this image; treat this result as uncertain.")

        if suffix.lower() not in ['.jpg', '.jpeg']:
            warnings.append(
                "This image is not a JPEG. Results are most reliable for JPEG photos — "
                "accuracy may be lower for other formats, since the model was trained "
                "on JPEG-compressed data."
            )

        manipulation_type = None
        manipulation_details = None
        heatmap_base64 = None

        if predicted_class == 1:
            manipulation_type, manipulation_details = estimate_manipulation_type(tmp_path)

            heatmap = gradcam.generate_heatmap(inputs["img_t"], inputs["ela_t"], target_class=1)

            display_img = cv2.imread(tmp_path)
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            display_img = cv2.resize(display_img, (224, 224))

            overlaid = overlay_heatmap(display_img, heatmap)
            heatmap_base64 = encode_image_array(overlaid)

        frequency_view_base64 = get_frequency_view(tmp_path)

        return {
            "filename": file.filename,
            "label": label,
            "tampered_probability": round(tampered_probability, 4),
            "manipulation_type": manipulation_type,
            "manipulation_details": manipulation_details,
            "heatmap": heatmap_base64,
            "frequency_view": frequency_view_base64,
            "warnings": warnings
        }

    finally:
        os.remove(tmp_path)