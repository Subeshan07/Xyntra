import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pennylane as qml
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Improved quantum parameters
num_qubits = 6  # Increased number of qubits for better expressivity
num_layers = 6  # Keep reasonable depth to avoid barren plateau

# Quantum device configuration
dev = qml.device("default.qubit", wires=num_qubits)

# Advanced quantum circuit
def quantum_circuit(inputs, weights):
    # Initial state preparation
    qml.AngleEmbedding(inputs, wires=range(num_qubits))
    
    # Multi-layer parameterized circuit
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    
    # Measure all qubits in Z basis
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

# QNode with simplified parameter shapes
weight_shapes = {"weights": (num_layers, num_qubits, 3)}  # 3 rotation parameters per qubit
qnode = qml.QNode(quantum_circuit, dev, interface="torch")

# Quantum Layer
class EnhancedQuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.scale = nn.Parameter(torch.tensor(0.1))  # Learnable scaling factor

    def forward(self, x):
        # Apply learnable scaling to input
        scaled_input = x * self.scale
        return self.q_layer(scaled_input)

# Multiclass Dataset for Lung Cancer CT scans
class LungCancerDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        # Scan for cancer type directories
        cancer_types = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        self.class_names = sorted(cancer_types)  # Sort for consistent class indexing
        
        for class_idx, cancer_type in enumerate(self.class_names):
            class_dir = os.path.join(base_path, cancer_type)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.dcm')):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
                    
        if len(self.image_paths) == 0:
            raise ValueError(f"❌ No images found in {base_path}!")
        
        print(f"Found {len(self.image_paths)} images in {len(cancer_types)} classes: {cancer_types}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image and the same label in case of errors
            blank_image = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224), (0, 0, 0))
            return blank_image, self.labels[idx]

# Enhanced image transformations with augmentation
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Set paths according to your folder structure - UPDATED PATH HERE
data_dir = "archive (2)/Data"  # Updated path as requested
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid") 
test_dir = os.path.join(data_dir, "test")

# Try to load datasets with path correction
try:
    train_dataset = LungCancerDataset(train_dir, transform=transform_train)
except Exception as e:
    print(f"Error loading training data: {e}")
    # Try to detect the correct path from current directory
    potential_paths = [".", "Data", "archive (2)/Data"]
    for path in potential_paths:
        if os.path.exists(os.path.join(path, "train")):
            train_dir = os.path.join(path, "train")
            valid_dir = os.path.join(path, "valid")
            test_dir = os.path.join(path, "test")
            print(f"Using detected path: {path}")
            break

# Retry with corrected paths
try:
    train_dataset = LungCancerDataset(train_dir, transform=transform_train)
    valid_dataset = LungCancerDataset(valid_dir, transform=transform_val)
    test_dataset = LungCancerDataset(test_dir, transform=transform_val)
except Exception as e:
    print(f"Still unable to load data: {e}")
    print("Please enter the correct absolute paths to your data folders:")
    base_path = input("Enter the absolute path to your data directory: ")
    if os.path.exists(base_path):
        train_dir = os.path.join(base_path, "train")
        valid_dir = os.path.join(base_path, "valid")
        test_dir = os.path.join(base_path, "test")
        train_dataset = LungCancerDataset(train_dir, transform=transform_train)
        valid_dataset = LungCancerDataset(valid_dir, transform=transform_val)
        test_dataset = LungCancerDataset(test_dir, transform=transform_val)

# Handle class imbalance with weighted sampling
class_counts = [train_dataset.labels.count(i) for i in range(len(train_dataset.class_names))]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for label in train_dataset.labels]
weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# DataLoaders with reduced batch sizes for CPU training
batch_size = 8 if device.type == 'cpu' else 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define Hybrid Model using ResNet50 as requested
class QuantumHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Feature extractor using ResNet50
        self.backbone = models.resnet50(pretrained=True)
        self.feature_dim = self.backbone.fc.in_features  # 2048 for ResNet50
        self.backbone.fc = nn.Identity()  # Remove classification layer
        
        # Feature reduction for quantum processing
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_qubits)
        )
        
        # Quantum layer
        self.quantum_layer = EnhancedQuantumLayer()
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(num_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # Extract features from CNN backbone
        features = self.backbone(x)
        
        # Reduce dimensions for quantum processing
        quantum_input = self.dim_reduction(features)
        
        # Apply quantum layer
        quantum_output = self.quantum_layer(quantum_input)
        
        # Classify
        logits = self.classifier(quantum_output)
        return logits

# Initialize the model
model = QuantumHybridModel(len(train_dataset.class_names)).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 5e-5},  # Lower learning rate for pretrained weights
    {'params': model.dim_reduction.parameters()},
    {'params': model.quantum_layer.parameters(), 'lr': 5e-3},  # Higher learning rate for quantum layer
    {'params': model.classifier.parameters()}
], lr=1e-3, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Training and validation functions
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return running_loss / len(dataloader), accuracy, precision, recall, f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return running_loss / len(dataloader), accuracy, precision, recall, f1, cm, all_preds, all_labels

# Training loop
num_epochs = 20 if device.type == 'cpu' else 30  # Reduce epochs for CPU training
best_val_f1 = 0.0
early_stop_patience = 8
early_stop_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
           'train_f1': [], 'val_f1': [], 'lr': []}

print(f"Starting training for {num_epochs} epochs...")

try:
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation phase
        val_loss, val_acc, val_prec, val_rec, val_f1, conf_matrix, _, _ = validate(
            model, valid_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - start_time
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f} | Valid F1: {val_f1:.4f}")
        print(f"Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | LR: {current_lr:.6f}")
        
        # Early stopping & model saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0
            # Save the model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'class_names': train_dataset.class_names
            }, 'best_quantum_lung_cancer_model.pth')
            print("✅ Model improved and saved!")
        else:
            early_stop_counter += 1
            print(f"⚠️ No improvement for {early_stop_counter} epochs")
            
            if early_stop_counter >= early_stop_patience:
                print("⛔ Early stopping triggered. Training halted.")
                break
        
        print("-" * 60)
except KeyboardInterrupt:
    print("Training interrupted by user.")

# Final evaluation on test set
print("\nEvaluating on test set...")
try:
    # Load best model
    checkpoint = torch.load('best_quantum_lung_cancer_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with F1 score: {checkpoint['val_f1']:.4f}")
except:
    print("Warning: Could not load best model. Using current model instead.")

test_loss, test_acc, test_prec, test_rec, test_f1, test_cm, test_preds, test_labels = validate(
    model, test_loader, criterion, device
)

print(f"Test Results:")
print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1 Score: {test_f1:.4f}")

# Plotting confusion matrix
try:
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=train_dataset.class_names,
                yticklabels=train_dataset.class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
except Exception as e:
    print(f"Could not save confusion matrix: {e}")

# Plot training history
try:
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Validation')
    plt.title('F1 Score')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
except Exception as e:
    print(f"Could not save training history plot: {e}")

print("🎉 Training and evaluation complete!")
print(f"Model saved as best_quantum_lung_cancer_model.pth")