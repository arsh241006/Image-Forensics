import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from fusion.dataset import FusionDataset
from fusion.fusion_model import FusionModel
from fusion.extract_features import (
    load_spatial_model, extract_spatial_features,
    load_frequency_model, extract_frequency_features
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Evaluating fusion model on:", device)

test_ds = FusionDataset(split='test')
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

spatial_model = load_spatial_model(device=device)
freq_model = load_frequency_model(device=device)

fusion_model = FusionModel(spatial_feature_dim=128, freq_feature_dim=128).to(device)
fusion_model.load_state_dict(torch.load('fusion/fusion_model.pt', map_location=device))
fusion_model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for img, ela, dct, labels in test_loader:
        img, ela, dct, labels = img.to(device), ela.to(device), dct.to(device), labels.to(device)

        spatial_features = extract_spatial_features(spatial_model, img, ela, device=device)
        freq_features = extract_frequency_features(freq_model, dct, device=device)

        outputs = fusion_model(spatial_features, freq_features)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"\n=== Fusion Model — Test Set Results ===")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"\nConfusion Matrix:")
print(f"                 Predicted Authentic  Predicted Tampered")
print(f"Actual Authentic        {cm[0][0]:>6}              {cm[0][1]:>6}")
print(f"Actual Tampered         {cm[1][0]:>6}              {cm[1][1]:>6}")