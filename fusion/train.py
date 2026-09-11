import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fusion.dataset import FusionDataset
from fusion.fusion_model import FusionModel
from fusion.extract_features import (
    load_spatial_model, extract_spatial_features,
    load_frequency_model, extract_frequency_features
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training fusion model on:", device)

train_ds = FusionDataset(split='train')
val_ds = FusionDataset(split='val')
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

# load BOTH frozen branch models once, outside the loop —
# they never change, so no need to reload them every batch
spatial_model = load_spatial_model(device=device)
freq_model = load_frequency_model(device=device)

fusion_model = FusionModel(spatial_feature_dim=128, freq_feature_dim=128).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)

EPOCHS = 15
best_val_acc = 0.0

for epoch in range(EPOCHS):
    fusion_model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (img, ela, dct, labels) in enumerate(train_loader):
        img, ela, dct, labels = img.to(device), ela.to(device), dct.to(device), labels.to(device)

        spatial_features = extract_spatial_features(spatial_model, img, ela, device=device)
        freq_features = extract_frequency_features(freq_model, dct, device=device)

        optimizer.zero_grad()
        outputs = fusion_model(spatial_features, freq_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} — running loss: {loss.item():.3f}")

    train_acc = correct / total

    fusion_model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for img, ela, dct, labels in val_loader:
            img, ela, dct, labels = img.to(device), ela.to(device), dct.to(device), labels.to(device)
            spatial_features = extract_spatial_features(spatial_model, img, ela, device=device)
            freq_features = extract_frequency_features(freq_model, dct, device=device)
            outputs = fusion_model(spatial_features, freq_features)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {total_loss:.3f}, "
          f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(fusion_model.state_dict(), 'fusion/fusion_model.pt')
        print(f"  New best fusion model saved (val acc: {val_acc:.3f})")