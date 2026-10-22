import os
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline_utils import prepare_inputs, predict_from_inputs


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


@app.get("/")
def root():
    return {
        "message": "Image Forensics API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Runs the real fusion pipeline on an uploaded image.
    Returns: label (authentic/tampered), probability, and any warnings.
    """
    warnings = []

    # save the uploaded file to a temp path — our pipeline functions
    # expect a real file path, not raw bytes in memory
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        inputs = prepare_inputs(tmp_path)
        predicted_class, tampered_probability = predict_from_inputs(inputs)

        label = "tampered" if predicted_class == 1 else "authentic"

        # honest-wording guard: don't claim more certainty than the
        # model actually has — flag genuinely uncertain predictions
        if 0.4 <= tampered_probability <= 0.6:
            warnings.append("Model confidence is low for this image; treat this result as uncertain.")

        return {
            "filename": file.filename,
            "label": label,
            "tampered_probability": round(tampered_probability, 4),
            "warnings": warnings
        }

    finally:
        # always clean up the temp file, even if prediction raised an error
        os.remove(tmp_path)