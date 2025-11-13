# train_model.py
# Author: Same Kiflu
# Description: Train and compare CNN vs. ResNet models for facial emotion recognition

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#  CONFIG
DATASET_DIR = "emotion_dataset"
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TRANSFORMS
# Data augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

# Simpler transforms for validation/testing (no augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# DATASETS
train_data = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transforms)
val_data = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=val_test_transforms)
test_data = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=val_test_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = train_data.classes
print(f"Detected emotion classes: {CLASS_NAMES}")


# CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Resnet model
def get_resnet(num_classes):
    # Load ResNet with pretrained ImageNet weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze earlier layers, fine-tune deeper layers + final FC layer
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True  # fine-tune last block
        else:
            param.requires_grad = False  # keep earlier layers frozen

    # Replace the final fully-connected layer for emotion classification
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, model_name):
    print(f"\n Training {model_name} on {DEVICE}...")
    model.to(DEVICE)
    best_val_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_acc = 100 * correct_val / total_val

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")

    print(f"Finished training {model_name}. Best Val Accuracy: {best_val_acc:.2f}%")

    # Log results
    with open("training_log.txt", "a") as log:
        log.write(f"{model_name}: Best Val Accuracy = {best_val_acc:.2f}%\n")

    return model


# evaluation
def evaluate_model(model, test_loader, model_name):
    print(f"\n Evaluating {model_name}...")
    model.eval()
    preds, labels_list = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.numpy())

    print(classification_report(labels_list, preds, target_names=CLASS_NAMES))
    cm = confusion_matrix(labels_list, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# main
if __name__ == "__main__":
    print("Starting emotion recognition model training pipeline...")

    # Initialize models
    cnn_model = SimpleCNN(num_classes=len(CLASS_NAMES))
    resnet_model = get_resnet(num_classes=len(CLASS_NAMES))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    optimizer_resnet = optim.Adam(filter(lambda p: p.requires_grad, resnet_model.parameters()), lr=LEARNING_RATE)

    # Train models
    trained_cnn = train_model(cnn_model, train_loader, val_loader, criterion, optimizer_cnn, "CNN_Model")
    trained_resnet = train_model(resnet_model, train_loader, val_loader, criterion, optimizer_resnet, "ResNet_Model")

    # Evaluate both
    evaluate_model(trained_cnn, test_loader, "CNN_Model")
    evaluate_model(trained_resnet, test_loader, "ResNet_Model")