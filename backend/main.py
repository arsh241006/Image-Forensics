from fastapi import FastAPI, UploadFile, File

app = FastAPI(
    title="Image Forensics API",
    description="Backend API for detecting image tampering",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Image Forensics API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Placeholder prediction endpoint.

    Later, this endpoint will:
    1. Receive an uploaded image
    2. Preprocess the image
    3. Pass it through the trained ML model
    4. Return the prediction and confidence
    """

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "prediction": "placeholder",
        "confidence": 0.0,
        "message": "Model prediction will be implemented here."
    }