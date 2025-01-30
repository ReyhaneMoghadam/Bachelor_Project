# import libraries
import os
import zipfile
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
zip_path = 'download/Dataset_Genre_224x224_SingleLabel.zip'
extract_path = 'download/'

if not os.path.exists('download/Dataset_Genre_224x224_SingleLabel'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted {zip_path} to {extract_path}")

# Data augmentation and normalization
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomCrop(224, padding=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha)

    def forward(self, outputs, labels):
        ce_loss = self.ce_loss(outputs, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Weighted sampler
train_dataset = ImageFolder(root='download/Total_Split/train', transform=transform['train'])
class_counts = np.bincount(train_dataset.targets)
class_weights = 1.0 / class_counts
samples_weights = [class_weights[label] for label in train_dataset.targets]
sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)

# Loss function
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = FocalLoss(alpha=class_weights, gamma=2)

# DataLoader
batch_size = 16
num_classes = 15
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_dataset = ImageFolder(root='download/Total_Split/valid', transform=transform['val'])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = ImageFolder(root='download/Total_Split/test', transform=transform['test'])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model architecture
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4),  # Adjusted dropout for balance
    nn.Linear(512, num_classes),
)
model = model.to(device)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    best_model_wts = model.state_dict()
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            all_labels = []
            all_preds = []

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
            print(f"{phase} Loss: {epoch_loss:.4f} Weighted F1: {epoch_f1:.4f}")

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = model.state_dict()

        scheduler.step()

    model.load_state_dict(best_model_wts)
    print(f"Best Val F1: {best_f1:.4f}")
    return model

# Train and evaluate
best_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute metrics
    test_loss = running_loss / len(dataloader.dataset)
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    class_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Weighted F1: {test_f1:.4f}")
    print("\nClassification Report:\n", class_report)

    return test_loss, test_accuracy, test_f1

# Evaluate the model on the test dataset
test_loss, test_accuracy, test_f1 = evaluate_model(best_model, test_loader, criterion)