import torch
import torch.nn as nn
 
class FusionModel(nn.Module):
    """
    Combines a spatial branch feature vector and a frequency branch
    feature vector into a single tampered/authentic prediction.
    Both input vectors are expected to already be 128-dim, produced by
    calling each branch's own model with return_features=True.
    """
    def __init__(self, spatial_feature_dim=128, freq_feature_dim=128, num_classes=2):
        super().__init__()
        combined_dim = spatial_feature_dim + freq_feature_dim
 
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),   # helps prevent overfitting on a small combined layer
            nn.Linear(64, num_classes)
        )
 
    def forward(self, spatial_features, freq_features):
        combined = torch.cat([spatial_features, freq_features], dim=1)
        return self.classifier(combined)
