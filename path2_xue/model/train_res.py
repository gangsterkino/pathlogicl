import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torchvision import models

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Assuming your tensor has a feature size of 2048
feature_size = 1000
num_classes = 3

class_labels = {
    'basic cell': 0,
    'shadow cell': 1,
    'Other': 2,
    'None': 3
}

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

        return feature_tensor, number_label, classification_name

# Using torchvision's pre-trained ResNet-50
resnet = models.resnet50(pretrained=True)
# Modify the final fully connected layer for the desired number of classes
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Continue with the training loop using resnet as your model

root_folder = '/Users/tanxinyu/毛母质瘤/datasets'
resolutions = [0, 1, 2]

transform = None

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

for resolution_level in resolutions:
    folder_name = f'window_images_resolution_{resolution_level}'
    folder_path = os.path.join(root_folder, folder_name)
    dataset = CustomDataset(folder_path, transform=None)
    print(f'start training {folder_name}')
    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)
    num_test = num_samples - num_train - num_val

    train_data, val_data, test_data = random_split(dataset, [num_train, num_val, num_test])

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)

    num_epochs = 10

    for epoch in range(num_epochs):
        resnet.train()
        total_correct = 0
        total_samples = 0

        for features, labels, classification_name in train_dataloader:
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = resnet(features)
            supervised_loss = criterion(outputs, labels)

            # ... 其他训练步骤

            # Calculate mixed loss
            loss = supervised_loss  # 这里暂时只使用监督损失，你可能需要根据自己的需求添加其他损失

            # ... 其他训练步骤

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        train_accuracy = total_correct / total_samples

        # Evaluate on validation set after each epoch
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for features, labels, _ in val_dataloader:
                features = features.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = resnet(features)
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

            accuracy = total_correct / total_samples
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {accuracy:.4f}')

    # Save the pre-trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_percentage = round(accuracy * 100, 2)
    model_name = f'resnet50_{timestamp}_{acc_percentage}_{epoch}.pth'
    model_save_path = os.path.join('model', model_name)

    # Create the directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    torch.save(resnet.state_dict(), model_save_path)
    print(f'Model saved at: {model_save_path}')

# Evaluate on the test set after training completes
resnet.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []

    for features, labels, _ in test_dataloader:
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = resnet(features)
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {round(accuracy * 100, 3)}%')

    # 计算其他指标
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Precision: {round(precision, 3)}, Recall: {round(recall, 3)}, F1 Score: {round(f1, 3)}')

    # 输出分类报告
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
