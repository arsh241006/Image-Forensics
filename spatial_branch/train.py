import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spatial_branch.dataset import TamperedImageDataset
from spatial_branch.model import build_spatial_model

VARIANT = 'raw'  # change to 'ela' or 'srm' to train other variants

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training variant '{VARIANT}' on:", device)

train_ds = TamperedImageDataset(split='train', variant=VARIANT)
val_ds = TamperedImageDataset(split='val', variant=VARIANT)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

model = build_spatial_model(feature_dim=128, variant=VARIANT).to(device)
criterion = nn.CrossEntropyLoss()

aux_params = list(model.aux_branch.parameters()) if model.aux_branch is not None else []
optimizer = torch.optim.Adam(
    list(model.fc_feature.parameters()) + list(model.fc_classifier.parameters()) + aux_params,
    lr=0.001
)

EPOCHS = 10
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for img, aux, labels in train_loader:
        img, aux, labels = img.to(device), aux.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(img, x_aux=aux)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for img, aux, labels in val_loader:
            img, aux, labels = img.to(device), aux.to(device), labels.to(device)
            outputs = model(img, x_aux=aux)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {total_loss:.3f}, "
          f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f'spatial_branch/baseline_model_{VARIANT}.pt')
        print(f"  New best model saved (val acc: {val_acc:.3f})")

print(f"Training complete. Best val accuracy ({VARIANT}): {best_val_acc:.3f}")