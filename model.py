import torch
import torch.nn as nn
from torchvision import models

def get_model():
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    
    # Freeze all layers (we only train the last layer)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer for binary classification
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)  # 2 classes: NORMAL, PNEUMONIA
    )
    
    return model