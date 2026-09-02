import torch
from fusion.fusion_model import FusionModel

# 1. Create the model
model = FusionModel()

# 2. Create dummy feature vectors
# Batch size = 4
# Each branch produces 128 features
spatial_features = torch.randn(4, 128)
freq_features = torch.randn(4, 128)

# 3. Forward pass
output = model(spatial_features, freq_features)

# 4. Print shapes
print("Spatial features shape:", spatial_features.shape)
print("Frequency features shape:", freq_features.shape)
print("Model output shape:", output.shape)

# 5. Print raw predictions
print("\nRaw output:")
print(output)

# 6. Convert logits to predicted class
predictions = torch.argmax(output, dim=1)

print("\nPredicted classes:")
print(predictions)