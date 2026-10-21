import torch
from preprocessing import load_and_preprocess
from spatial_branch.ela_transform import compute_ela
from freq_branch.dct_transform import extract_dct_features
from fusion.extract_features import (
    load_spatial_model, extract_spatial_features,
    load_frequency_model, extract_frequency_features
)
from fusion.fusion_model import FusionModel
 
 
class ForensicsPipeline:
    """
    Single source of truth for 'image file in, prediction out'.
    Uses exactly the same steps as FusionDataset + fusion/train.py,
    so robustness tests, cross-dataset tests and the web backend
    all behave identically.
    """
    def __init__(self, device='cpu', fusion_weights='fusion/fusion_model.pt'):
        self.device = device
        self.spatial_model = load_spatial_model(device=device)
        self.freq_model = load_frequency_model(device=device)
        self.fusion_model = FusionModel(spatial_feature_dim=128, freq_feature_dim=128).to(device)
        self.fusion_model.load_state_dict(torch.load(fusion_weights, map_location=device))
        self.fusion_model.eval()
 
    def predict(self, image_path):
        """Returns (predicted_class, tampered_probability). 0=authentic, 1=tampered."""
        img = load_and_preprocess(image_path)
        img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
 
        # ELA reads the real file, same as during training
        ela = compute_ela(image_path)
        ela_t = torch.tensor(ela, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
 
        # DCT works on the preprocessed array, same as during training
        dct = extract_dct_features(img)
        dct_t = torch.tensor(dct, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
 
        with torch.no_grad():
            spatial = extract_spatial_features(self.spatial_model, img_t, ela_t, device=self.device)
            freq = extract_frequency_features(self.freq_model, dct_t, device=self.device)
            probs = torch.softmax(self.fusion_model(spatial, freq), dim=1)[0]
 
        return int(probs.argmax().item()), float(probs[1].item())