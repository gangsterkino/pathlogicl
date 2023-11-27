import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from custom_model import CustomDataset, MyResNet50, CustomClassifier

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Assuming your tensor has a feature size of 2048
feature_size = 2048
num_classes = 3  # Update with your actual number of classes

class_labels = {
    'basic cell': 0,
    'shadow cell': 1,
    'Other': 2,
    'None': 3
}
# Instantiate the dataset
root_folder = '/path/to/your/dataset'  # Update with your dataset path
transform = None  # Add your data transformation if needed
dataset = CustomDataset(root_folder, transform=transform)

# Split dataset into train, validation, and test sets
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

num_samples = len(dataset)
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)
num_test = num_samples - num_train - num_val

train_data, val_data, test_data = random_split(dataset, [num_train, num_val, num_test])

# Instantiate data loaders
batch_size = 4  # Adjust as needed
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Instantiate the model
model = MyResNet50(input_size=feature_size, num_classes=num_classes).to(device)
classifier = CustomClassifier(input_size=feature_size, num_classes=num_classes).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ...

# Training parameters
# ...

# 训练参数
num_epochs = 10
best_val_accuracy = 0.0  # 用于跟踪最佳验证准确性的变量

for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0

    for features, labels in train_dataloader:
        features = features.to(device)
        labels = labels.to(device)

        # 通过您的 ResNet 模型进行前向传播
        resnet_outputs = model(features)

        # 通过您的自定义分类器进行前向传播
        classifier_outputs = classifier(resnet_outputs)

        # 计算损失
        loss = criterion(classifier_outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确性
        _, predicted = torch.max(classifier_outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        # "none" 类样本的自监督标签分配
        model.eval()
        with torch.no_grad():
            none_class_indices = (labels == class_labels['None']).nonzero().squeeze()
            if len(none_class_indices) > 0:
                none_features = features[none_class_indices]
                none_labels = torch.argmax(model(none_features), dim=1)
                none_loss = F.cross_entropy(resnet_outputs[none_class_indices], none_labels)
                total_samples += len(none_class_indices)
                total_correct += (none_labels == labels[none_class_indices]).sum().item()
                loss += model.self_supervised_weight * none_loss

        model.train()

    train_accuracy = total_correct / total_samples

    # 每个 epoch 后在验证集上评估
    with torch.no_grad():
        model.eval()
        total_correct = 0
        total_samples = 0

        for val_features, val_labels in val_dataloader:
            val_features = val_features.to(device)
            val_labels = val_labels.to(device)

            # 通过您的 ResNet 模型进行前向传播
            val_resnet_outputs = model(val_features)

            # 通过您的自定义分类器进行前向传播
            val_classifier_outputs = classifier(val_resnet_outputs)

            # 计算准确性
            _, val_predicted = torch.max(val_classifier_outputs, 1)
            total_samples += val_labels.size(0)
            total_correct += (val_predicted == val_labels).sum().item()

        val_accuracy = total_correct / total_samples
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # 如果是迄今为止最佳的验证准确性，则保存模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f'best_my_resnet50_classifier_{timestamp}.pth'
            model_save_path = os.path.join('model', model_name)
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved at: {model_save_path}')

# Save the trained model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f'my_resnet50_classifier_{timestamp}_{epoch}.pth'
model_save_path = os.path.join('model', model_name)

# Create the directory if it doesn't exist
os.makedirs('model', exist_ok=True)

torch.save(model.state_dict(), model_save_path)
print(f'Model saved at: {model_save_path}')

# Evaluate on the test set after training completes
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []

    for test_features, test_labels in test_dataloader:
        test_features = test_features.to(device)
        test_labels = test_labels.to(device)

        # Forward pass through your ResNet model
        test_resnet_outputs = model(test_features)

        # Forward pass through your custom classifier
        test_classifier_outputs = classifier(test_resnet_outputs)

        # Calculate accuracy
        _, test_predicted = torch.max(test_classifier_outputs, 1)
        total_samples += test_labels.size(0)
        total_correct += (test_predicted == test_labels).sum().item()

        y_true.extend(test_labels.cpu().numpy())
        y_pred.extend(test_predicted.cpu().numpy())

    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {round(accuracy * 100, 3)}%')

    # Calculate other metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Precision: {round(precision, 3)}, Recall: {round(recall, 3)}, F1 Score: {round(f1, 3)}')

    # Output classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
