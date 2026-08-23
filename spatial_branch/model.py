import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
 
def build_spatial_model():
    # load ResNet18 with pretrained ImageNet weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
 
    # freeze all existing layers — this means "don't change what they've already learned"
    for param in model.parameters():
        param.requires_grad = False
 
    # replace the final layer: 512 features in, 2 classes out (Authentic/Tampered)
    # this new layer starts untrained, and requires_grad=True by default, so it WILL learn
    model.fc = nn.Linear(in_features=512, out_features=2)
 
    return model
