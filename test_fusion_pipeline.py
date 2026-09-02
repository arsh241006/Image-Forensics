import torch
import torch.nn as nn
import torch.optim as optim

from fusion.fusion_model import FusionModel


# -----------------------------------
# 1. Create Fusion Model
# -----------------------------------

model = FusionModel()

# -----------------------------------
# 2. Simulate Spatial Branch
# -----------------------------------

# In the real system, the spatial model will
# produce a 128-dimensional feature vector.

spatial_features = torch.randn(4, 128)


# -----------------------------------
# 3. Simulate Frequency Branch
# -----------------------------------

# In the real system, the frequency model will
# produce a 128-dimensional feature vector.

freq_features = torch.randn(4, 128)


# -----------------------------------
# 4. Simulate Ground Truth Labels
# -----------------------------------

# 0 = Authentic
# 1 = Tampered

labels = torch.tensor([0, 1, 0, 1])


# -----------------------------------
# 5. Fusion
# -----------------------------------

output = model(spatial_features, freq_features)

print("Spatial features:", spatial_features.shape)
print("Frequency features:", freq_features.shape)
print("Fusion output:", output.shape)


# -----------------------------------
# 6. Loss
# -----------------------------------

criterion = nn.CrossEntropyLoss()

loss = criterion(output, labels)

print("Loss:", loss.item())


# -----------------------------------
# 7. Backpropagation
# -----------------------------------

optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("Backpropagation successful!")
print("Optimizer step successful!")


# -----------------------------------
# 8. Prediction
# -----------------------------------

predictions = torch.argmax(output, dim=1)

print("Predictions:", predictions)
print("Actual labels:", labels)


# -----------------------------------
# 9. Accuracy
# -----------------------------------

accuracy = (predictions == labels).float().mean()

print("Dummy accuracy:", accuracy.item())