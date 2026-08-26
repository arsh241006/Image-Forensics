import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spatial_branch.dataset import TamperedImageDataset
from spatial_branch.model import build_spatial_model
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on:", device)
 
# load data
train_ds = TamperedImageDataset(split='train')
val_ds = TamperedImageDataset(split='val')
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
 
# load model
model = build_spatial_model().to(device)
 
# loss function measures how wrong the model's guesses are
criterion = nn.CrossEntropyLoss()
# optimizer decides how to adjust the model's numbers based on that "wrongness"
# note: only model.fc.parameters() are being trained, since everything else is frozen
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
 
EPOCHS = 8
 
for epoch in range(EPOCHS):
    # ---- TRAINING ----
    model.train()  # tells PyTorch we're in training mode
    total_loss, correct, total = 0, 0, 0
 
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
 
        optimizer.zero_grad()             # clear old gradients
        outputs = model(images)           # get model's predictions
        loss = criterion(outputs, labels) # how wrong were we?
        loss.backward()                   # calculate how to adjust
        optimizer.step()                  # actually adjust the model
 
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
 
    train_acc = correct / total
 
    # ---- VALIDATION ----
    model.eval()  # tells PyTorch we're just checking, not learning
    val_correct, val_total = 0, 0
    with torch.no_grad():  # don't calculate gradients, we're not training here
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
 
    val_acc = val_correct / val_total
 
    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {total_loss:.3f}, "
          f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
 
# save the trained model so you don't have to retrain from scratch every time
torch.save(model.state_dict(), 'spatial_branch/baseline_model.pt')
print("Model saved.")
