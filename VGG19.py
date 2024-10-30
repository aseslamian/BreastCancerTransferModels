## VGG19 Pretrained Model for Breast Cancer Detection
## This code is run on Broka in breat 

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the dataset class
class IDCDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Set the dataset directory
dataset_dir = '/home/data3/Ali/Code/Saina/Brea/Dataset/'
# dataset_dir = '/home/data3/Ali/Code/Saina/Brea/TestData/'

# Collect image paths and labels
logger.info('Collecting image paths and labels...')
image_paths = []
labels = []

for folder_name in os.listdir(dataset_dir):
    class_dir_0 = os.path.join(dataset_dir, folder_name, '0')
    class_dir_1 = os.path.join(dataset_dir, folder_name, '1')
    for img_name in os.listdir(class_dir_0):
        image_paths.append(os.path.join(class_dir_0, img_name))
        labels.append(0)
    for img_name in os.listdir(class_dir_1):
        image_paths.append(os.path.join(class_dir_1, img_name))
        labels.append(1)

logger.info(f'Collected {len(image_paths)} images.')

# Split the data into training, validation, and testing sets
logger.info('Splitting data into training, validation, and testing sets...')
train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=0.3, stratify=labels, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.3333, stratify=temp_labels, random_state=42)

logger.info(f'Training set size: {len(train_paths)}')
logger.info(f'Validation set size: {len(val_paths)}')
logger.info(f'Test set size: {len(test_paths)}')

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create datasets
logger.info('Creating datasets...')
train_dataset = IDCDataset(train_paths, train_labels, transform=data_transforms['train'])
val_dataset = IDCDataset(val_paths, val_labels, transform=data_transforms['val'])
test_dataset = IDCDataset(test_paths, test_labels, transform=data_transforms['test'])

# Create dataloaders
logger.info('Creating dataloaders...')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pretrained VGG19 model
logger.info('Loading pretrained VGG19 model...')
model = models.vgg19(pretrained=True)

# Modify the final layer for binary classification
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 1)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Early stopping
early_stopping_patience = 5
early_stopping_counter = 0
best_loss = float('inf')

# Training and evaluation
num_epochs = 10
best_model_wts = None

train_acc_history, val_acc_history = [], []
train_loss_history, val_loss_history = [], []

logger.info('Starting training...')
for epoch in range(num_epochs):
    logger.info(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            dataloader = train_loader
        else:
            model.eval()   # Set model to evaluate mode
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs) > 0.5

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        if phase == 'train':
            train_acc_history.append(epoch_acc.cpu().numpy())
            train_loss_history.append(epoch_loss)
        else:
            val_acc_history.append(epoch_acc.cpu().numpy())
            val_loss_history.append(epoch_loss)

            # Deep copy the model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    scheduler.step(epoch_loss)

    if early_stopping_counter >= early_stopping_patience:
        logger.info('Early stopping triggered')
        break

logger.info('Training complete')

# Load best model weights
model.load_state_dict(best_model_wts)

# Evaluate on test set
logger.info('Evaluating on test set...')
model.eval()
all_preds = []
all_labels = []

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device).float().unsqueeze(1)
    outputs = model(inputs)
    preds = torch.sigmoid(outputs) > 0.5
    all_preds.append(preds.cpu().numpy())
    all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

logger.info('Generating classification report...')
print(classification_report(all_labels, all_preds, target_names=['0', '1']))

# Plot Accuracy and Loss
logger.info('Plotting accuracy and loss...')
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc_history, label='Training accuracy')
plt.plot(val_acc_history, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_history, label='Training loss')
plt.plot(val_loss_history, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('/home/data3/Ali/Code/Saina/Brea/OutPut/NewFigure13.png')
