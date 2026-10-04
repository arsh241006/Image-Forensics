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