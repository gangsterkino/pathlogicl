# custom_model.py
import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader

input_size = 2048
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.file_list = [file for file in os.listdir(folder_path) if file.endswith('.pt')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)
        # Load the feature tensor
        feature_tensor = torch.load(file_path)
        # Extract information from the file name
        parts = file_name.split('_')
        classification_name = parts[0]
        number_label = int(parts[1])

        # Apply the transform if available
        if self.transform:
            feature_tensor = self.transform(feature_tensor)

        return feature_tensor, number_label, classification_name

class CustomClassifier(nn.Module):
    def __init__(self, input_size, num_classes, supervised_weight=1.0, self_supervised_weight=0.1):
        super(CustomClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.supervised_weight = supervised_weight
        self.self_supervised_weight = self_supervised_weight

    def forward(self, x):
        x = self.fc(x)
        return x


class MyResNet50(nn.Module):
    def __init__(self, input_size, num_classes, pretrained=True):
        super(MyResNet50, self).__init__()
        # Load the pre-trained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=pretrained)

        # Freeze all layers
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Modify the final classification layer
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnet50(x)
