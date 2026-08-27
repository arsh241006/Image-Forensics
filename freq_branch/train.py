import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from freq_branch.dataset import FrequencyDataset
from freq_branch.model import FrequencyCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on:", device)

train_ds = FrequencyDataset(split='train')
val_ds = FrequencyDataset(split='val')
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

model = FrequencyCNN(feature_dim=128).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for dct_maps, labels in train_loader:
        dct_maps, labels = dct_maps.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(dct_maps)
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
        for dct_maps, labels in val_loader:
            dct_maps, labels = dct_maps.to(device), labels.to(device)
            outputs = model(dct_maps)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {total_loss:.3f}, "
          f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

torch.save(model.state_dict(), 'freq_branch/baseline_model.pt')
print("Model saved.")