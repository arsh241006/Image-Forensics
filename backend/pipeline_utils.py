import torch
from fusion.pipeline import ForensicsPipeline
from preprocessing import load_and_preprocess
from spatial_branch.ela_transform import compute_ela
from freq_branch.dct_transform import extract_dct_features
from fusion.extract_features import extract_spatial_features, extract_frequency_features

# load all 3 models ONCE, reused across every request
pipeline = ForensicsPipeline()


def prepare_inputs(image_path):
    """
    Mirrors ForensicsPipeline.predict()'s preprocessing exactly, but
    returns the intermediate tensors too — needed so the backend can
    reuse them for Grad-CAM and frequency views without recomputing.
    """
    img = load_and_preprocess(image_path)
    img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    ela = compute_ela(image_path)
    ela_t = torch.tensor(ela, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    dct = extract_dct_features(img)
    dct_t = torch.tensor(dct, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return {"img": img, "img_t": img_t, "ela": ela, "ela_t": ela_t, "dct_t": dct_t}


def predict_from_inputs(inputs):
    """
    Same math as ForensicsPipeline.predict(), split out so it can run
    on tensors already computed by prepare_inputs() — avoids
    recomputing ELA/DCT a second time for Grad-CAM / frequency views.
    """
    with torch.no_grad():
        spatial = extract_spatial_features(pipeline.spatial_model, inputs["img_t"], inputs["ela_t"], device=pipeline.device)
        freq = extract_frequency_features(pipeline.freq_model, inputs["dct_t"], device=pipeline.device)
        probs = torch.softmax(pipeline.fusion_model(spatial, freq), dim=1)[0]

    return int(probs.argmax().item()), float(probs[1].item())