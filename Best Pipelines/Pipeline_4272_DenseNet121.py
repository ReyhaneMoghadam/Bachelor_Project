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
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.functional import log_softmax
from collections import Counter
import torch.nn.functional as F


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
zip_path = 'download/Dataset_Genre_224x224_SingleLabel_Customized.zip'
extract_path = 'download/'

if not os.path.exists('download/Dataset_Genre_224x224_SingleLabel_Customized'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted {zip_path} to {extract_path}")

# Data augmentation and normalization
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # Vertical flips may apply to some genres
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
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

class CBLoss(nn.Module):
    def __init__(self, beta, samples_per_cls):
        super(CBLoss, self).__init__()
        self.beta = beta
        self.samples_per_cls = samples_per_cls

    def forward(self, logits, labels):
        # Compute effective number of samples
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(self.samples_per_cls)
        weights = torch.tensor(weights).float().to(logits.device)  # Move to same device as logits

        # Create one-hot encoding for labels and move to the same device
        labels_one_hot = torch.eye(len(weights), device=logits.device)[labels]

        # Calculate the class weights for each sample
        weights = weights.unsqueeze(0) * labels_one_hot
        weights = weights.sum(1)

        # Calculate loss using weighted log-softmax
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -weights * log_probs[range(logits.size(0)), labels]
        return loss.mean()

# Weighted sampler
train_dataset = ImageFolder(root='download/Total_Split - Copy/train', transform=transform['train'])
class_counts = np.bincount(train_dataset.targets)
class_weights = 1.0 / class_counts
samples_weights = [class_weights[label] for label in train_dataset.targets]
sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)

# Loss function
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

class_counts = Counter(train_dataset.targets)
class_distribution = list(dict(class_counts).values())
criterion = CBLoss(beta = 0.99, samples_per_cls = class_distribution)

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label, index

    def __len__(self):
        return len(self.dataset)

# DataLoader
batch_size = 16
num_classes = 13
num_epochs = 50
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_dataset = ImageFolder(root='download/Total_Split - Copy/valid', transform=transform['val'])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = ImageFolder(root='download/Total_Split - Copy/test', transform=transform['test'])
indexed_test_dataset = IndexedDataset(test_dataset)
test_loader = DataLoader(indexed_test_dataset, batch_size=batch_size, shuffle=False)

# Model architecture
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes),
)
model = model.to(device)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, gradient_accumulation_steps=1):
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

            optimizer.zero_grad()  # Ensure optimizer is reset outside the loop
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        # Scale loss for gradient accumulation
                        loss = loss / gradient_accumulation_steps
                        loss.backward()

                        # Perform optimizer step only after accumulation steps
                        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                            optimizer.step()
                            optimizer.zero_grad()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
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

def evaluate_model(model, dataloader, criterion, dataset):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_labels = []
    all_preds = []
    correctly_predicted_image = []
    correctly_predicted_label = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for inputs, labels, indices in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            # Track correctly predicted images
            for idx, label, pred in zip(indices, labels, preds):
                if label.item() == pred.item():
                    img_path = dataset.samples[idx.item()][0]  # Get the image path
                    correctly_predicted_image.append(os.path.basename(img_path))  # Extract file name
                    correctly_predicted_label.append(label.item())

    # Compute metrics
    test_loss = running_loss / len(dataloader.dataset)
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    class_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Weighted F1: {test_f1:.4f}")
    print("\nClassification Report:\n", class_report)

    # Print correctly predicted image names and their labels
    print("\nCorrectly Predicted Images and Labels:")
    for img_name, label in zip(correctly_predicted_image, correctly_predicted_label):
        print(f"Image: {img_name}, Label: {label}")

    return test_loss, test_accuracy, test_f1

# Evaluate the model on the test dataset
test_loss, test_accuracy, test_f1 = evaluate_model(best_model, test_loader, criterion, test_dataset)