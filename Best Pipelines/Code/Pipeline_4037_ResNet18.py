import os
import zipfile
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import glob
from collections import Counter
import numpy as np

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 16
epochs = 100
learning_rate = 5e-4

# Dataset paths
zip_path = 'download/Dataset_Genre_224x224_MultiLabel.zip'
extract_path = 'download/'

if not os.path.exists('download/Dataset_Genre_224x224_MultiLabel'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted {zip_path} to {extract_path}")

# Paths to folders and CSVs
train_folder = 'download/Dataset_Genre_224x224_MultiLabel/Train'
train_csv = 'download/Dataset_Genre_224x224_MultiLabel/train_multi_label_dataset.csv'
val_folder = 'download/Dataset_Genre_224x224_MultiLabel/Validation'
val_csv = 'download/Dataset_Genre_224x224_MultiLabel/valid_multi_label_dataset.csv'
test_folder = 'download/Dataset_Genre_224x224_MultiLabel/Test'
test_csv = 'download/Dataset_Genre_224x224_MultiLabel/test_multi_label_dataset.csv'

# Data augmentation and normalization
custom_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomCrop(224, padding=10),
    transforms.ToTensor(),
])

normalize_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = {
    'train': transforms.Compose([custom_train_transform, normalize_transform]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_transform
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_transform
    ])
}

# Custom dataset class for multi-label classification
class MultiLabelDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None, genre_list=None):
        self.image_folder = image_folder
        self.csv_data = pd.read_csv(csv_file, quotechar='"')
        self.transform = transform

        # Extract all unique genres and create a mapping
        if genre_list is None:
            all_genres = set()
            for genres in self.csv_data['Labels']:
                all_genres.update(genres.split(','))
            self.genre_list = sorted(all_genres)
        else:
            self.genre_list = genre_list

        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genre_list)}

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]
        img_hash = row['Hash']
        genres = row['Labels'].split(',')

        # Load image dynamically
        img_path = f"{self.image_folder}/{img_hash}.*"
        img_files = glob.glob(img_path)
        if not img_files:
            raise FileNotFoundError(f"No image found for hash: {img_hash}")
        image = Image.open(img_files[0]).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert genres to multi-hot vector
        genre_indices = [self.genre_to_idx[g] for g in genres]
        label = torch.zeros(len(self.genre_list), dtype=torch.float32)
        label[genre_indices] = 1.0

        return image, label


# Load datasets
train_dataset = MultiLabelDataset(train_folder, train_csv, transform=transform['train'])
val_dataset = MultiLabelDataset(val_folder, val_csv, transform=transform['val'], genre_list=train_dataset.genre_list)
test_dataset = MultiLabelDataset(test_folder, test_csv, transform=transform['test'], genre_list=train_dataset.genre_list)

# Save genre-to-index mapping for future use
with open('genre_to_idx.json', 'w') as f:
    json.dump(train_dataset.genre_to_idx, f)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
data_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
num_classes = len(train_dataset.genre_list)

# Model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes),
)
model = model.to(device)

train_csv_weight = pd.read_csv(train_csv)
all_genres = []
for labels in train_csv_weight['Labels']:
    all_genres.extend(labels.split(','))  # Split multi-label genres into individual genres

genre_counts = Counter(all_genres)
genre_counts = dict(genre_counts)
samples_per_class = np.array([genre_counts[genre] for genre in train_dataset.genre_list], dtype=np.float32)

# Calculate class weights (inverse of frequencies)
class_weights = 1.0 / (samples_per_class + 1e-6)  # Add epsilon to avoid division by zero
class_weights = class_weights / np.sum(class_weights) * len(samples_per_class)  # Normalize
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Loss function with class weights, optimizer and scheduler
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
'''
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 
'''
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1, verbose=True)


# Helper function to compute metrics
def compute_macro_metrics(preds, labels_one_hot):
    preds = preds.cpu().numpy()
    labels_one_hot = labels_one_hot.cpu().numpy()
    precision = precision_score(labels_one_hot, preds, average='macro', zero_division=0)
    recall = recall_score(labels_one_hot, preds, average='macro', zero_division=0)
    f1 = f1_score(labels_one_hot, preds, average='macro', zero_division=0)
    return precision, recall, f1

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            epoch_preds = []
            epoch_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    preds = (outputs > 0.5).type(torch.FloatTensor).to(device)
                    epoch_preds.append(preds.cpu())
                    epoch_labels.append(labels.cpu())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_preds = torch.cat(epoch_preds)
            epoch_labels = torch.cat(epoch_labels)
            epoch_loss = running_loss / data_sizes[phase]

            epoch_precision, epoch_recall, epoch_f1 = compute_macro_metrics(epoch_preds, epoch_labels)
            print(f'{phase} Loss: {epoch_loss:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1:.4f}')

             # Update scheduler only on validation phase
            if phase == 'val':
                scheduler.step(epoch_f1)

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = model.state_dict()
        
        ''' scheduler.step() '''

    print(f'Best val F1: {best_f1:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# Train the model
best_model = train_model(model, dataloaders, criterion, optimizer, epochs)

# Save the model
torch.save(best_model.state_dict(), 'resnet_multilabel.pth')

# Evaluate on the test dataset
def evaluate_model(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    epoch_preds = []
    epoch_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            preds = (outputs > 0.5).type(torch.FloatTensor).to(device)
            epoch_preds.append(preds.cpu())
            epoch_labels.append(labels.cpu())

    test_loss = test_loss / len(dataloader.dataset)
    epoch_preds = torch.cat(epoch_preds)
    epoch_labels = torch.cat(epoch_labels)

    precision, recall, f1 = compute_macro_metrics(epoch_preds, epoch_labels)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')

    return test_loss, precision, recall, f1

# Evaluate the best model on the test set
test_loss, test_precision, test_recall, test_f1 = evaluate_model(best_model, dataloaders['test'], criterion)
