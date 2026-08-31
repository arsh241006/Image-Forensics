import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
 
class SpatialCNN(nn.Module):
    def __init__(self, feature_dim=128, variant='raw'):
        """
        variant: 'raw' (RGB only), 'ela', or 'srm'
        Both auxiliary variants add a small conv branch for the extra
        input, which gets concatenated with the ResNet18 backbone output.
        """
        super().__init__()
        self.variant = variant
 
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in base_model.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        # outputs (batch, 512, 1, 1)
 
        if variant in ('ela', 'srm'):
            # small conv branch for the auxiliary input (ELA or SRM image)
            self.aux_branch = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            combined_dim = 512 + 16
        else:
            self.aux_branch = None
            combined_dim = 512
 
        self.fc_feature = nn.Linear(combined_dim, feature_dim)
        self.fc_classifier = nn.Linear(feature_dim, 2)
 
    def forward(self, x_rgb, x_aux=None, return_features=False):
        rgb_feat = self.backbone(x_rgb)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)  # (batch, 512)
 
        if self.variant in ('ela', 'srm') and x_aux is not None:
            aux_feat = self.aux_branch(x_aux)
            aux_feat = aux_feat.view(aux_feat.size(0), -1)  # (batch, 16)
            combined = torch.cat([rgb_feat, aux_feat], dim=1)  # (batch, 528)
        else:
            combined = rgb_feat
 
        features = self.fc_feature(combined)  # (batch, 128)
 
        if return_features:
            return features
 
        out = self.fc_classifier(features)
        return out
 
 
def build_spatial_model(feature_dim=128, variant='raw'):
    return SpatialCNN(feature_dim=feature_dim, variant=variant)
