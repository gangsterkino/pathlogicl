import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import logging

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Assuming your tensor has a feature size of 2048
# feature_size = 2048
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

# Define your custom classification head
class CustomClassifier(nn.Module):
    def __init__(self, input_size, num_classes, supervised_weight=1.0, self_supervised_weight=0.1):
        super(CustomClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.supervised_weight = supervised_weight
        self.self_supervised_weight = self_supervised_weight

    def forward(self, x):
        x = self.fc(x)
        return x

# Self-supervised label assignment logic using the classification name

model = CustomClassifier(feature_size, num_classes)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

root_folder = '/path/to/datasets'  # Update the path
resolutions = [0, 1, 2]

transform = None

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Set up logging
log_file = '/path/to/logs/train_log.txt'  # Update the path

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info(f"Training started at {datetime.now()}")
print(f"Training started at {datetime.now()}")  # Added print statement

for resolution_level in resolutions:
    folder_name = f'window_images_resolution_{resolution_level}'
    folder_path = os.path.join(root_folder, folder_name)
    dataset = CustomDataset(folder_path, transform=None)
    logging.info(f"Start training {folder_name}")
    print(f"Start training {folder_name}")  # Added print statement

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
        model.train()
        epoch_train_loss = np.array([])
        epoch_train_accuracy = np.array([])

        with tqdm(total=num_train, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='tensor') as pbar:
            for features, labels, classification_name in train_dataloader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(features)
                supervised_loss = criterion(outputs, labels)

                # Your self-supervised label assignment logic here...
                if classification_name == 'none':
                    with torch.no_grad():
                        # Forward pass to get predicted labels
                        predicted_labels = torch.argmax(model(features), dim=1)
                    self_supervised_loss = F.cross_entropy(outputs, predicted_labels)
                else:
                    self_supervised_loss = 0.0
                loss = model.supervised_weight * supervised_loss + model.self_supervised_weight * self_supervised_loss

                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())

                epoch_train_loss = np.append(epoch_train_loss, loss.item())
                epoch_train_accuracy = np.append(epoch_train_accuracy, accuracy)

                pbar.set_postfix(loss=loss.item(), accuracy=accuracy)
                pbar.update(features.size(0))

        avg_train_loss = np.mean(epoch_train_loss)
        avg_train_accuracy = np.mean(epoch_train_accuracy)

        # Validation
        model.eval()
        epoch_val_accuracy = np.array([])

        with torch.no_grad():
            for features, labels, _ in val_dataloader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())

                epoch_val_accuracy = np.append(epoch_val_accuracy, accuracy)

        avg_val_accuracy = np.mean(epoch_val_accuracy)

        logging.info(
            f'Epoch {epoch + 1}/{num_epochs}, Avg Train Loss: {avg_train_loss:.4f}, Avg Train Accuracy: {avg_train_accuracy:.4f}, Avg Validation Accuracy: {avg_val_accuracy:.4f}')
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Avg Train Loss: {avg_train_loss:.4f}, Avg Train Accuracy: {avg_train_accuracy:.4f}, Avg Validation Accuracy: {avg_val_accuracy:.4f}')  # Added print statement
    print("Training complete!")  # Added print statement
    logging.info("Training complete!")
# Evaluate on the test set after training completes
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []

    for features, labels, _ in test_dataloader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {round(accuracy * 100, 3)}%')
    logging.info(f'Test Accuracy: {round(accuracy * 100, 3)}%')
    print(f'Test Accuracy: {round(accuracy * 100, 3)}%')  # Added print statement

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Precision: {round(precision, 3)}, Recall: {round(recall, 3)}, F1 Score: {round(f1, 3)}')
    logging.info(f'Precision: {round(precision, 3)}, Recall: {round(recall, 3)}, F1 Score: {round(f1, 3)}')
    print(
        f'Precision: {round(precision, 3)}, Recall: {round(recall, 3)}, F1 Score: {round(f1, 3)}')  # Added print statement

    logging.info("Classification Report:")
    logging.info(classification_report(y_true, y_pred))
    print(f'Classification Report:\n'
          f'{classification_report(y_true, y_pred)}')


