import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SpatialCNN(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # freeze all pretrained layers
        for param in base_model.parameters():
            param.requires_grad = False

        # keep everything except ResNet18's original final layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        # this outputs a (batch, 512, 1, 1) feature map

        self.fc_feature = nn.Linear(512, feature_dim)    # 512 -> 128
        self.fc_classifier = nn.Linear(feature_dim, 2)    # 128 -> 2, for this week's baseline only

    def forward(self, x, return_features=False):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)        # flatten (batch, 512, 1, 1) -> (batch, 512)
        features = self.fc_feature(x)    # (batch, 128)

        if return_features:
            return features

        out = self.fc_classifier(features)
        return out


def build_spatial_model(feature_dim=128):
    return SpatialCNN(feature_dim=feature_dim)