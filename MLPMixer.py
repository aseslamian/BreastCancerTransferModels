import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import logging
import torch.nn.functional as F
import math
from functools import partial


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
# dataset_dir = '/home/data3/Ali/Code/Saina/Brea/Dataset/'
dataset_dir = '/home/data3/Ali/Code/Saina/Brea/TestData/'

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

########################################### Implement One #################################################################
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(128 * 28 * 28, 512)
#         self.fc2 = nn.Linear(512, 1)
#         self.dropout = nn.Dropout(0.5)
    
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# # Initialize the CNN model
# logger.info('Initializing CNN model...')
# model = SimpleCNN()

############################################## Implement Two ##############################################################
# class GELU(nn.Module):
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# class AdjustableCNN(nn.Module):
#     def __init__(self, num_classes=1, num_layers=3, dropout=0.5):
#         super(AdjustableCNN, self).__init__()
#         layers = []
#         in_channels = 3
#         out_channels = 32

#         for _ in range(num_layers):
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
#             layers.append(nn.ReLU())
#             layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
#             in_channels = out_channels
#             out_channels *= 2

#         self.conv_layers = nn.Sequential(*layers)
#         self.fc1 = nn.Linear(out_channels // 2 * 28 * 28, 512)
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# model = AdjustableCNN(num_classes=1, num_layers=3, dropout=0.5)

################################################ Implement Three ############################################################
# class GELU(nn.Module):
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# class PreNormResidual(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         return self.fn(self.norm(x)) + x

# def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
#     return nn.Sequential(
#         dense(dim, dim * expansion_factor),
#         GELU(),
#         nn.Dropout(dropout),
#         dense(dim * expansion_factor, dim),
#         nn.Dropout(dropout)
#     )

# class MLPMixer(nn.Module):
#     def __init__(self, image_size, patch_size, dim, depth, num_classes, expansion_factor=4, dropout=0.):
#         super(MLPMixer, self).__init__()
        
#         num_patches = (image_size // patch_size) ** 2
#         self.patch_size = patch_size
#         self.num_patches = num_patches
        
#         self.embedding = nn.Linear(patch_size * patch_size * 3, dim)
        
#         chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        
#         self.mixer = nn.Sequential(
#             *[nn.Sequential(
#                 PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
#                 PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
#             ) for _ in range(depth)]
#         )
        
#         self.ln0 = nn.LayerNorm(dim)
#         self.output = nn.Sequential(nn.Linear(dim, num_classes), nn.Sigmoid())
        
#     def forward(self, x):
#         batch_size, channels, height, width = x.shape
        
#         x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#         x = x.contiguous().view(batch_size, channels, -1, self.patch_size * self.patch_size)
#         x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.patch_size * self.patch_size * channels)
        
#         # print(x.shape)
        
#         x = self.embedding(x)
#         x = self.mixer(x)
        
#         print(x.shape)
        
#         x = self.ln0(x)
#         x = x.mean(dim=1)
#         x = self.output(x)
#         return x
    
# model = MLPMixer(
#     image_size=224,  # Size of the input image
#     patch_size=16,   # Size of each patch
#     dim=512,        # Dimension of the embedding
#     depth=8,        # Number of Mixer layers
#     num_classes=1,  # Number of output classes
#     expansion_factor=4, # Expansion factor for feed-forward layers
#     dropout=0.1     # Dropout rate
# )

############################################# Implement Four ###############################################################
# class GELU(nn.Module):
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
# def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
#     return nn.Sequential(
#         dense(dim, dim * expansion_factor),
#         GELU(),
#         nn.Dropout(dropout),
#         dense(dim * expansion_factor, dim),
#         nn.Dropout(dropout)
#     )
# class MLPMixer(nn.Module):
#     def __init__(self, image_size, patch_size, dim, depth, num_classes, expansion_factor=4, dropout=0.):
#         super(MLPMixer, self).__init__()

#         num_patches = (image_size // patch_size) ** 2
#         patch_dim = patch_size * patch_size * 3 
#         self.patch_size = patch_size
#         self.dim = dim
#         self.num_patches = num_patches
#         self.embedding = nn.Linear(patch_dim, dim)
#         self.norm = nn.LayerNorm(dim)
#         chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
#         self.token_mixing = nn.Sequential(
#             *[nn.Sequential(
#                 nn.LayerNorm(dim),
#                 FeedForward(num_patches, expansion_factor, dropout, chan_first)
#             ) for _ in range(depth)]
#         )
#         self.channel_mixing = nn.Sequential(
#             *[nn.Sequential(
#                 nn.LayerNorm(dim),
#                 FeedForward(dim, expansion_factor, dropout, chan_last)
#             ) for _ in range(depth)]
#         )
#         self.ln0 = nn.LayerNorm(dim)
#         self.output = nn.Sequential(nn.Linear(dim, num_classes), nn.Sigmoid())

#     def forward(self, x):
#         batch_size, channels, height, width = x.shape
#         x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#         x = x.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
#         x = x.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, self.patch_size * self.patch_size * channels)
#         x = self.embedding(x)
#         x = self.norm(x)
#         for layer in self.token_mixing:
#             x = x + layer(x)
#         for layer in self.channel_mixing:
#             x = x + layer(x)
#         x = self.ln0(x)
#         x = x.mean(dim=1)
#         x = self.output(x)
#         return x

# model = MLPMixer(
#     image_size=224,  # Size of the input image
#     patch_size=16,   # Size of each patch
#     dim=512,        # Dimension of the embedding
#     depth=8,        # Number of Mixer layers
#     num_classes=1,  # Number of output classes
#     expansion_factor=4, # Expansion factor for feed-forward layers
#     dropout=0.1     # Dropout rate
# )

########################################## Implemnt Five ##################################################################
class GELU(nn.Module):
    """
    GELU activation function used in BERT and other models.
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class MLPMixer(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, num_classes, expansion_factor=4, dropout=0., layer_norm_eps=1e-5, device=None, dtype=None):
        super(MLPMixer, self).__init__()
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 3  # 3 for RGB channels
        
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = num_patches
        
        self.embedding = nn.Linear(patch_dim, dim)
        self.norm = nn.LayerNorm(dim, eps=layer_norm_eps, device=device, dtype=dtype)

        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        self.mlp_layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim, eps=layer_norm_eps, device=device, dtype=dtype),
                nn.Sequential(
                    nn.Linear(num_patches, dim * expansion_factor, device=device, dtype=dtype),
                    GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * expansion_factor, num_patches, device=device, dtype=dtype),
                    nn.Dropout(dropout)
                ),
                nn.LayerNorm(dim, eps=layer_norm_eps, device=device, dtype=dtype),
                nn.Sequential(
                    nn.Linear(dim, dim * expansion_factor, device=device, dtype=dtype),
                    GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * expansion_factor, dim, device=device, dtype=dtype),
                    nn.Dropout(dropout)
                )
            ])
            for _ in range(depth)
        ])

        self.ln0 = nn.LayerNorm(dim, eps=layer_norm_eps, device=device, dtype=dtype)
        self.output = nn.Sequential(nn.Linear(dim, num_classes, device=device, dtype=dtype), nn.Sigmoid())

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Create patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, self.patch_size * self.patch_size * channels)

        # Embedding
        x = self.embedding(x)
        
        # Layer normalization
        x = self.norm(x)

        # MLP-Mixer blocks
        for ln1, mlp1, ln2, mlp2 in self.mlp_layers:
            y = ln1(x)
            y = y.transpose(-1, -2)
            y = mlp1(y)
            y = y.transpose(-1, -2)
            x = x + y
            y = ln2(x)
            y = mlp2(y)
            x = x + y
        x = self.ln0(x)
        x = x.mean(dim=1)

        x = self.output(x)

        return x

# Example usage
model = MLPMixer(
    image_size=224,  # Size of the input image
    patch_size=16,   # Size of each patch
    dim=512,        # Dimension of the embedding
    depth=8,        # Number of Mixer layers
    num_classes=1,  # Number of output classes
    expansion_factor=4, # Expansion factor for feed-forward layers
    dropout=0.1     # Dropout rate
)
############################################################################################################

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
num_epochs = 30
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
# plt.show()
