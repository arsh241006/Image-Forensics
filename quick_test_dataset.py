from spatial_branch.dataset import TamperedImageDataset
from torch.utils.data import DataLoader

train_ds = TamperedImageDataset(split='train')
print("Number of training images:", len(train_ds))

img, label = train_ds[0]
print("Image shape:", img.shape)
print("Label:", label)

# ---- DataLoader / batching check ----
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

images, labels = next(iter(train_loader))
print("Batch of images shape:", images.shape)   # expect torch.Size([16, 3, 224, 224])
print("Batch of labels shape:", labels.shape)     # expect torch.Size([16])