import torch
import torch.nn as nn

class FrequencyCNN(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        # a small stack of conv layers — each one shrinks the image and
        # increases the number of feature "channels" it's tracking
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
        )
        # squash everything down to a single feature vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_feature = nn.Linear(64, feature_dim)  # feature_dim=128 agreed with Person A
        self.fc_classifier = nn.Linear(feature_dim, 2)  # for standalone baseline training only

    def forward(self, x, return_features=False):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten to (batch, 64)
        features = self.fc_feature(x)  # (batch, 128) — this is what Week 4 fusion will use
        if return_features:
            return features
        out = self.fc_classifier(features)  # (batch, 2) — used only for this week's baseline check
        return out