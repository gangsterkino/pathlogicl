import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import logging
import torch.nn.functional as F
import numpy as np
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

# Instantiate the dataset
root_folder = '/Users/tanxinyu/毛母质瘤/datasets'
resolutions = [0, 1, 2]

transform = None

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Set up logging
log_file = '/Users/tanxinyu/毛母质瘤/logs/train_log.txt'

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.info(f"Training started at {datetime.now()}")



for resolution_level in resolutions:
    folder_name = f'window_images_resolution_{resolution_level}'
    folder_path = os.path.join(root_folder, folder_name)
    dataset = CustomDataset(folder_path, transform=None)
    logging.info(f"Start training {folder_name}")

    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)
    num_test = num_samples - num_train - num_val

    train_data, val_data, test_data = random_split(dataset, [num_train, num_val, num_test])

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False)

    # Instantiate the model
    model = MyResNet50(input_size=feature_size, num_classes=num_classes).to(device)
    classifier = CustomClassifier(input_size=feature_size, num_classes=num_classes).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_val_accuracy = 0.0  # Variable to keep track of the best validation accuracy

    # Lists to store results for each epoch
    epoch_train_losses = []
    epoch_train_accuracies = []
    epoch_train_precisions = []
    epoch_train_recalls = []
    epoch_train_f1_scores = []
    epoch_val_losses = []
    epoch_val_accuracies = []
    epoch_val_precisions = []
    epoch_val_recalls = []
    epoch_val_f1_scores = []

    # Self-supervised training parameters
    self_supervised_loss_weight = 0.1  # You can adjust this weight based on your specific setup

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        y_true_train = []
        y_pred_train = []

        with tqdm(total=num_train, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='tensor') as pbar:
            for features, labels, classification_name in train_dataloader:
                features = features.to(device)
                labels = labels.to(device)

                # Forward pass through your ResNet model
                resnet_outputs = model(features)

                # Forward pass through your custom classifier
                classifier_outputs = classifier(resnet_outputs)

                # Calculate the supervised loss
                supervised_loss = criterion(classifier_outputs, labels)

                # Identify samples with "None" label
                none_mask = (labels == class_labels['None'])

                # If there are samples with "None" label, apply self-supervised loss
                if none_mask.any():
                    # Forward pass through ResNet model for self-supervised samples
                    self_supervised_outputs = model(features[none_mask])

                    # Calculate self-supervised loss
                    # Define self-supervised loss function
                    self_supervised_loss = F.mse_loss(self_supervised_outputs, labels[none_mask])
                    self_supervised_loss *= self_supervised_loss_weight

                    # Combine supervised and self-supervised losses
                    loss = supervised_loss + self_supervised_loss
                else:
                    # No "None" label samples, only use supervised loss
                    loss = supervised_loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(classifier_outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(predicted.cpu().numpy())

                pbar.update(labels.size(0))
                pbar.set_postfix({'loss': loss.item(), 'accuracy': total_correct / total_samples})

            # Print and log metrics for both supervised and self-supervised training
            train_accuracy = total_correct / total_samples
            epoch_train_losses.append(loss.item())
            epoch_train_accuracies.append(train_accuracy)

            precision_train = precision_score(y_true_train,y_pred_train, average = 'weighted')
            recall_train = recall_score(y_true_train, y_pred_train, average='weighted')
            f1_train = f1_score(y_true_train, y_pred_train, average='weighted')

            epoch_train_precisions.append(precision_train)
            epoch_train_recalls.append(recall_train)
            epoch_train_f1_scores.append(f1_train)

            logging.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, '
                         f'Train Accuracy: {train_accuracy:.4f}, Train Precision: {precision_train:.4f}, '
                         f'Train Recall: {recall_train:.4f}, Train F1 Score: {f1_train:.4f}')

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}, Train Precision: {precision_train:.4f}, '
                  f'Train Recall: {recall_train:.4f}, Train F1 Score: {f1_train:.4f}')

            # Print and log additional metrics for self-supervised training
            if none_mask.any():
                self_supervised_accuracy = accuracy_score(y_true_train[none_mask], y_pred_train[none_mask])
                logging.info(f'Epoch {epoch + 1}/{num_epochs}, Self-Supervised Accuracy: {self_supervised_accuracy:.4f}')
                print(f'Epoch {epoch + 1}/{num_epochs}, Self-Supervised Accuracy: {self_supervised_accuracy:.4f}')

        # Validation after each epoch
        with torch.no_grad():
            model.eval()
            total_correct = 0
            total_samples = 0
            val_losses = []
            y_true_val = []
            y_pred_val = []

            for val_features, val_labels, _ in val_dataloader:
                val_features = val_features.to(device)
                val_labels = val_labels.to(device)

                # Forward pass through your ResNet model
                val_resnet_outputs = model(val_features)

                # Forward pass through your custom classifier
                val_classifier_outputs = classifier(val_resnet_outputs)

                # Calculate loss
                val_loss = criterion(val_classifier_outputs, val_labels)
                val_losses.append(val_loss.item())

                # Calculate accuracy
                _, val_predicted = torch.max(val_classifier_outputs, 1)
                total_samples += val_labels.size(0)
                total_correct += (val_predicted == val_labels).sum().item()

                y_true_val.extend(val_labels.cpu().numpy())
                y_pred_val.extend(val_predicted.cpu().numpy())

            val_accuracy = total_correct / total_samples
            epoch_val_accuracies.append(val_accuracy)

            logging.info(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')

            # Save the model if it has the best validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_checkpoint_path = f'models/model/best_model_checkpoint_resolution_{resolution_level}.pth'
                torch.save(model.state_dict(), model_checkpoint_path)
                logging.info(f"Model checkpoint saved: {model_checkpoint_path}")

            # Append validation metrics
            epoch_val_losses.append(np.mean(val_losses))
            precision_val = precision_score(y_true_val, y_pred_val, average='weighted')
            recall_val = recall_score(y_true_val, y_pred_val, average='weighted')
            f1_val = f1_score(y_true_val, y_pred_val, average='weighted')

            epoch_val_precisions.append(precision_val)
            epoch_val_recalls.append(recall_val)
            epoch_val_f1_scores.append(f1_val)

            logging.info(f'Epoch {epoch + 1}/{num_epochs}, Validation Precision: {precision_val:.4f}, '
                         f'Validation Recall: {recall_val:.4f}, Validation F1 Score: {f1_val:.4f}')
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Precision: {precision_val:.4f}, '
                  f'Validation Recall: {recall_val:.4f}, Validation F1 Score: {f1_val:.4f}')

    # Append all metrics to the results CSV
    results_csv_path = f'results/training_results_resolution_{resolution_level}.csv'
    np.savetxt(results_csv_path,
               np.column_stack((epoch_train_losses, epoch_train_accuracies, epoch_train_precisions, epoch_train_recalls, epoch_train_f1_scores,
                                epoch_val_losses, epoch_val_accuracies, epoch_val_precisions, epoch_val_recalls, epoch_val_f1_scores)),
               delimiter=',',
               header='Train_Loss,Train_Accuracy,Train_Precision,Train_Recall,Train_F1,Val_Loss,Val_Accuracy,Val_Precision,Val_Recall,Val_F1',
               comments='')

    logging.info(f"Training results saved to {results_csv_path}")

logging.info("Training complete!")
