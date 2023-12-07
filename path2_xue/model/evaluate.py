import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import logging
from custom_model import CustomDataset, MyResNet50, CustomClassifier  # assuming you have a custom_model module

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Assuming your tensor has a feature size of 2048
feature_size = 2048
num_classes = 3
class_labels = {
    'basic cell': 0,
    'shadow cell': 1,
    'Other': 2,
    'None': 3
}

# Instantiate the model
model = MyResNet50(input_size=feature_size, num_classes=num_classes).to(device)
classifier = CustomClassifier(input_size=feature_size, num_classes=num_classes).to(device)

# Load the trained model weights
model_checkpoint_path = 'models/model/best_model_checkpoint.pth'
model.load_state_dict(torch.load(model_checkpoint_path))
model.eval()

# Assuming your test dataset is organized in a folder similar to the training and validation datasets
test_folder_path = '/path/to/your/test/folder'
test_dataset = CustomDataset(test_folder_path, transform=None)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Evaluate on the test set after training completes
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []

    for features, labels, _ in test_dataloader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        classifier_outputs = classifier(outputs)
        _, predicted = torch.max(classifier_outputs, 1)

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

    print("Testing complete!")  # Added print statement
    logging.info("Testing complete!")
