import argparse
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import logging
from tqdm import tqdm
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_folder = ImageFolder(root=folder_path, transform=transform)
        self.class_to_idx = self.image_folder.class_to_idx
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, label = self.image_folder[idx]

        # Extract information from the image file name
        file_name = self.image_folder.imgs[idx][0]
        parts = file_name.split(os.sep)
        resolution_name = parts[-1].split('_')[5]  # Extracting resolution name from the filename
        number_label = int(parts[-1].split('_')[1])

        return image, number_label, resolution_name

# Custom dataset class

class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x



def get_args():
    parser = argparse.ArgumentParser(description='Train the ResNet model for image classification')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--val_percent', '-vp', metavar='VP', type=float, default=10, help='Percentage of data for validation')
    parser.add_argument('--save_checkpoint', action='store_true', default=True, help='Save model checkpoint')
    parser.add_argument('--root', type=str, required=True,default='dataset_png', help='Root directory of the classification dataset')
    # Add any other necessary arguments for your specific use case

    return parser.parse_args()

# Function to train the classification model


def train_classification_model(
        model,
        device,
        train_loader,
        val_loader,
        resolution_level,  # Add resolution_level as a parameter
        epochs: int = 5,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        results_csv: str = 'results/results.csv',
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    all_results = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        true_labels_train = []  # Define true_labels_train for this epoch
        outputs_train = []
        predicted_labels_train = []

        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='image') as pbar:
            for features, labels, classification_name in train_loader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Your self-supervised label assignment logic here...
                if 'None' in classification_name:
                    # If only one element is 'none', change all elements to 'none'
                    if classification_name.count('None') >= 1:
                        classification_name = ['None'] * len(classification_name)

                    model.eval()  # Set the model to evaluation mode
                    with torch.no_grad():
                        # Forward pass to get predicted labels
                        predicted_labels = torch.argmax(model(features), dim=1)
                        # Ensure predicted labels are within the valid range (0, 1, 2)
                        predicted_labels = torch.clamp(predicted_labels, 0, 2)

                    model.train()

                    previous_supervised_loss = 0.0
                else:
                    model.train()  # Set the model back to training mode
                    predicted_labels = labels  # Use the ground truth labels for other classes

                outputs = model(features)
                outputs_train.extend(outputs.cpu().numpy())  # 保存训练集上的输出结果
                predicted_labels_train.extend(predicted_labels.cpu().numpy())  # 保存训练集上的预测标签

                self_supervised_loss = 0.0
                supervised_loss = criterion(outputs, predicted_labels)
                loss = model.supervised_weight * supervised_loss + model.self_supervised_weight * self_supervised_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == predicted_labels).sum().item()
                total_samples += labels.size(0)

                true_labels_train.extend(labels.cpu().numpy())  # Append true labels for this batch

                # Only calculate metrics if the label is not 'None'
                if 'None' not in classification_name:
                    accuracy = accuracy_score(predicted_labels.cpu().numpy(), predicted.cpu().numpy())
                    precision = precision_score(predicted_labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
                    recall = recall_score(predicted_labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
                    f1 = f1_score(predicted_labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')

                    pbar.set_postfix(loss=loss.item(), accuracy=accuracy, precision=precision, recall=recall, f1=f1)
                    pbar.update(features.size(0))

        # Calculate and log training metrics only if 'None' is not present
        if 'None' not in classification_name:
            accuracy_train = correct_predictions / total_samples
            precision_train = precision
            recall_train = recall
            f1_train = f1

            logging.info(f'Epoch {epoch + 1}/{epochs}, Training Accuracy: {accuracy_train}, '
                         f'Precision: {precision_train}, Recall: {recall_train}, F1: {f1_train}')

        # Validation loop
        model.eval()
        true_labels_val = []
        predicted_labels_val = []
        outputs_val = []

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch['image'], batch['label']
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                # Convert output probabilities to predicted class
                _, predicted = torch.max(outputs.data, 1)

                true_labels_val.extend(labels.cpu().numpy())
                predicted_labels_val.extend(predicted.cpu().numpy())
                outputs_val.extend(outputs.cpu().numpy())

        # Calculate and log validation metrics only if 'None' is not present
        if 'None' not in classification_name:
            acc_val = accuracy_score(true_labels_val, predicted_labels_val)
            precision_val = precision_score(true_labels_val, predicted_labels_val, average='weighted')
            recall_val = recall_score(true_labels_val, predicted_labels_val, average='weighted')
            f1_val = f1_score(true_labels_val, predicted_labels_val, average='weighted')

            logging.info(f'Epoch {epoch + 1}/{epochs}, Validation Accuracy: {acc_val}, '
                         f'Precision: {precision_val}, Recall: {recall_val}, F1: {f1_val}')

            # Save the model if it has the best validation accuracy
            if save_checkpoint and acc_val > best_val_acc:
                best_val_acc = acc_val
                torch.save(model.state_dict(), f'model/best_model_resolution_{resolution_level}_epoch{epoch + 1}.pth')

                # Calculate and save additional metrics for both training and validation
                confusion_mat_train = confusion_matrix(true_labels_train, predicted_labels_train)
                roc_auc_basic_cell_train = roc_auc_score(true_labels_train == class_labels['basic cell'],
                                                         outputs_train[:, class_labels['basic cell']])
                roc_auc_shadow_cell_train = roc_auc_score(true_labels_train == class_labels['shadow cell'],
                                                          outputs_train[:, class_labels['shadow cell']])
                roc_auc_other_train = roc_auc_score(true_labels_train == class_labels['Other'],
                                                    outputs_train[:, class_labels['Other']])

                confusion_mat_val = confusion_matrix(true_labels_val, predicted_labels_val)
                roc_auc_basic_cell_val = roc_auc_score(true_labels_val == class_labels['basic cell'],
                                                       outputs_val[:, class_labels['basic cell']])
                roc_auc_shadow_cell_val = roc_auc_score(true_labels_val == class_labels['shadow cell'],
                                                        outputs_val[:, class_labels['shadow cell']])

                # Append results to the list
                results = {
                    'epoch': epoch + 1,
                    'train_accuracy': accuracy_train,
                    'train_precision': precision_train,
                    'train_recall': recall_train,
                    'train_f1': f1_train,
                    'val_accuracy': acc_val,
                    'val_precision': precision_val,
                    'val_recall': recall_val,
                    'val_f1': f1_val,
                    'confusion_matrix_train': confusion_mat_train,
                    'roc_auc_basic_cell_train': roc_auc_basic_cell_train,
                    'roc_auc_shadow_cell_train': roc_auc_shadow_cell_train,
                    'roc_auc_other_train': roc_auc_other_train,
                    'confusion_matrix_val': confusion_mat_val,
                    'roc_auc_basic_cell_val': roc_auc_basic_cell_val,
                    'roc_auc_shadow_cell_val': roc_auc_shadow_cell_val,
                }

                all_results.append(results)

                # Save all results to CSV
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(results_csv, index=False)

if __name__ == '__main__':
    class_labels = {
        'basic cell': 0,
        'shadow cell': 1,
        'Other': 2,
        'None': 3
    }
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')
    num_classes = 3
    resolutions = [0, 1, 2, 3]  # Add the resolutions you want to train on

    # 初始化模型
    classification_model = ClassificationModel(num_classes=num_classes)
    classification_model.to(device)

    for resolution_level in resolutions:
        folder_name = f'window_images_resolution_{resolution_level}'
        csv_path = f'logs/resolution_{resolution_level}.csv'
        # dataset = CustomDataset(os.path.join(args.root, folder_name), transform=None)
        dataset = CustomDataset(root=args.root, transform=transforms.ToTensor())

        # Split into train / validation partitions
        n_val = int(len(dataset) * (args.val_percent / 100))
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        # Create data loaders
        loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

        # 载入先前分辨率级别的模型参数
        if resolution_level > 0:
            previous_model_path = f'model/model/best_model_resolution_{resolution_level - 1}.pth'
            classification_model.load_state_dict(torch.load(previous_model_path))

        try:
            train_classification_model(
                model=classification_model,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                learning_rate=args.lr,
                save_checkpoint=args.save_checkpoint,
                results_csv=csv_path
            )

            # 保存当前分辨率级别的模型参数
            torch.save(classification_model.state_dict(), f'model/model/best_model_resolution_{resolution_level}.pth')
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError! Handle as needed.')
            torch.cuda.empty_cache()

