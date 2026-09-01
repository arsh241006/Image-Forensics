import torch
import torch.nn as nn
import torch.optim as optim

from fusion.fusion_model import FusionModel

# Create model
model = FusionModel()

# Dummy input features
batch_size = 4

spatial_features = torch.randn(batch_size, 128)
freq_features = torch.randn(batch_size, 128)

# Dummy labels
# 0 = Authentic
# 1 = Tampered
labels = torch.tensor([0, 1, 0, 1])

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# Training step
# -------------------------

# Forward pass
output = model(spatial_features, freq_features)

print("Output shape:", output.shape)

# Calculate loss
loss = criterion(output, labels)

print("Loss before backward:", loss.item())

# Backpropagation
optimizer.zero_grad()
loss.backward()

# Update weights
optimizer.step()

print("Backpropagation successful!")
print("Weights updated successfully.")

# Predictions
predictions = torch.argmax(output, dim=1)

print("Predictions:", predictions)
print("Actual labels:", labels)