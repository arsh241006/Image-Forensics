import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """
    Implements Grad-CAM for the frozen spatial (ELA) model.
    Hooks into the last convolutional layer of the ResNet18 backbone
    to capture both its output (the feature maps) and the gradients
    flowing back into it during a backward pass.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, img_tensor, ela_tensor, target_class=1):
        """
        target_class=1 means 'Tampered' — we specifically want to see
        what the model looked at when deciding an image IS tampered.
        """
        self.model.eval()

        # force input tensors to require grad — even though the backbone's
        # weights are frozen, we need SOME path requiring grad so PyTorch
        # actually builds a gradient graph through the frozen conv layers
        img_tensor = img_tensor.clone().requires_grad_(True)
        ela_tensor = ela_tensor.clone().requires_grad_(True)

        output = self.model(img_tensor, x_aux=ela_tensor)

        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)

        heatmap = heatmap.squeeze().cpu().numpy()
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)



        return heatmap


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    original_img: the original image as a numpy array, 0-255, HxWx3, in RGB order
    heatmap: the small Grad-CAM output, e.g. (7, 7)
    """
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # fix: match RGB order
    overlaid = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid
from preprocessing import load_and_preprocess
from spatial_branch.ela_transform import compute_ela
from spatial_branch.model import build_spatial_model


def estimate_heatmap(image_path, weights_path='spatial_branch/baseline_model_ela.pt', target_class=1):
    """
    Consistent image_path-based interface, matching the team's other
    Phase 4 functions (estimate_manipulation_type, etc).

    Handles preprocessing internally — loads the model fresh each call,
    which is simpler but slower than reusing a loaded model across many
    images. Fine for single-image/demo use; for batch processing (like
    Day 4's IoU validation), load the model once and call generate_heatmap
    directly instead.

    Returns: raw (7,7) heatmap array, values 0-1 — same as generate_heatmap.
    Call overlay_heatmap() separately if a visual overlay is needed.
    """
    model = build_spatial_model(feature_dim=128, variant='ela')
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    target_layer = model.backbone[-2][-1]
    gradcam = GradCAM(model, target_layer)

    img = load_and_preprocess(image_path)
    ela = compute_ela(image_path)

    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    ela_tensor = torch.tensor(ela, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    heatmap = gradcam.generate_heatmap(img_tensor, ela_tensor, target_class=target_class)
    return heatmap