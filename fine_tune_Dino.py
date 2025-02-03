import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


# Global definitions
map_disease = {
        'Band Neutrophil': 0,
        'Basophil': 1,
        'Eosinophil': 2,
        'Erythroblast': 3,
        'Lymphocyte': 4,
        'Metamyelocyte': 5,
        'Monocyte': 6,
        'Myelocyte': 7,
        'Platelet': 8,
        'Promyelocyte': 9,
        'Segmented Neutrophil': 10
    }

label_to_disease = {v: k for k, v in map_disease.items()}


# =============================================================================
def get_dino_bloom(modelpath="/content/dinobloom-b.pth",modelname="dinov2_vitb14", embed_size=768):
    # load the original DINOv2 model with the correct architecture and parameters.
    model=torch.hub.load('facebookresearch/dinov2', modelname)
    # load finetuned weights
    if modelpath is not None:
        pretrained = torch.load(modelpath, map_location=torch.device('cpu'))
        # make correct state dict for loading
        new_state_dict = {}
        for key, value in pretrained['teacher'].items():
            if 'dino_head' in key or "ibot_head" in key:
                pass
            else:
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = value

        #corresponds to 224x224 image. patch size=14x14 => 16*16 patches
        pos_embed = torch.nn.Parameter(torch.zeros(1, 257, embed_size))
        model.pos_embed = pos_embed

        model.load_state_dict(new_state_dict, strict=True)
    return model
# -------------------------
# 1. Custom Dataset
# -------------------------
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, return_path=False):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            transform (callable, optional): Transformations to be applied on a sample.
            return_path (bool): If True, return the image path along with image and label.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Read row from CSV
        row = self.data.iloc[idx]
        img_path = row['original_image_path']
        label = int(map_disease[row['label']])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.return_path:
            return image, label, img_path
        else:
            return image, label

# -------------------------
# 2. Data Transforms
# -------------------------
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Training transform with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

# Validation / Inference transform
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

# -------------------------
# 3. Model: ViT + MLP Classifier
# -------------------------
class ViTMLPClassifier(nn.Module):
    def __init__(self, vit_model, num_classes, feature_dim=384):
        """
        Args:
            vit_model (nn.Module): Pre-trained ViT acting as feature extractor.
            num_classes (int): Number of classes for classification.
            feature_dim (int): Dimension of the ViT feature vector.
        """
        super(ViTMLPClassifier, self).__init__()
        self.vit = vit_model
        # Freeze ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # MLP classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.vit(x)  # Expected shape: [batch_size, feature_dim]
        logits = self.classifier(features)
        return logits

# -------------------------
# 4. Training and Evaluation Functions
# -------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels).item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc

# -------------------------
# 5. Inference Function (saving predictions to CSV)
# -------------------------
def run_inference(model, csv_file, batch_size, device, output_csv='predictions.csv'):
    """
    Run inference on images in csv_file, then save predictions with corresponding image_path.
    """
    # Set return_path=True to get image paths along with image and label.
    dataset = CustomImageDataset(csv_file, transform=val_transform, return_path=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    results = []
    with torch.no_grad():
        for images, _, paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy().tolist()
            for img_path, pred in zip(paths, preds):
                results.append((img_path, pred))
    # Save the results to a CSV file
    df = pd.DataFrame(results, columns=['image_path', 'prediction'])
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    return results

def run_testing(model, csv_file, batch_size, device, output_csv='predictions.csv', metrics_csv='metrics.csv'):
    """
    Run testing on the dataset provided by csv_file.
    
    This function makes predictions for each image in the CSV, then creates a new CSV
    with the columns: image_path, true label, and predicted label.
    
    It also computes the following metrics:
      - Accuracy
      - Balanced Accuracy
      - Weighted F1 Score
      
    The metrics are saved to a separate CSV file.
    """
    dataset = CustomImageDataset(csv_file, transform=val_transform, return_path=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            # Move predictions and labels back to CPU and convert to list
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_paths.extend(paths)

    # Calculate metrics using scikit-learn.
    acc = accuracy_score(all_labels, all_preds)
    bAcc = balanced_accuracy_score(all_labels, all_preds)
    wF1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Save metrics to a CSV file.
    metrics_dict = {
        'accuracy': [acc],
        'balanced_accuracy': [bAcc],
        'weighted_f1': [wF1]
    }
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Saved metrics to {metrics_csv}")
    print(f"Accuracy: {acc:.4f}, Balanced Accuracy: {bAcc:.4f}, Weighted F1: {wF1:.4f}")
    
    all_labels = [label_to_disease[l] for l in all_labels]
    all_preds = [label_to_disease[l] for l in all_preds]
    # Create a DataFrame with image_path, label, and prediction.
    df_results = pd.DataFrame({
        'image_path': all_paths,
        'label': all_labels,
        'prediction': all_preds
    })
    df_results.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    

    
    #return df_results, df_metrics

# -------------------------
# 6. Main Training Loop
# -------------------------
def main():

    parser = argparse.ArgumentParser(description="Train a classifier on top of a pre-trained ViT model.")
    parser.add_argument('--train_csv', type=str, required=True, help="Path to the training CSV file")
    parser.add_argument('--val_csv', type=str, required=True, help="Path to the validation CSV file")
    parser.add_argument('--test_csv', type=str, required=True, help="Path to the test CSV file")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini-batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--modelname', type=str, default='dinov2_vits14', help="Model name")
    parser.add_argument('--modelpath', type=str, default=None, help="Model path")
    parser.add_argument('--results_path', type=str, default='RESULTS-DINO', help="root dir of results saving")
    args = parser.parse_args()

    # Hyper-parameters
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    train_csv = args.train_csv  
    val_csv = args.val_csv    
    test_csv = args.test_csv
    modelpath = args.modelpath if args.modelpath != "None" else None

    num_of_train_sample = len(pd.read_csv(train_csv))
    num_of_train_sample = int(num_of_train_sample//len(pd.read_csv(train_csv).label.unique()))
    if modelpath is not None:
        RESULTS = Path(f'{args.results_path}/Results_{num_of_train_sample}')
    else:
        RESULTS = Path(f'{args.results_path}/Results_{num_of_train_sample}_baseline')
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Create datasets and dataloaders
    train_dataset = CustomImageDataset(train_csv, transform=train_transform)
    val_dataset = CustomImageDataset(val_csv, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Determine number of classes from the training CSV
    num_classes = len(pd.read_csv(train_csv)['label'].unique())

    # Load the pre-trained ViT model
    modelname = args.modelname
    embed_sizes={"dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536}
    feature_dim = embed_sizes[modelname] 
    vit = get_dino_bloom(modelpath,modelname,feature_dim)

    # Build the combined model (ViT feature extractor + MLP classifier)
    model = ViTMLPClassifier(vit, num_classes, feature_dim=feature_dim)

    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    best_loss = np.inf
    best_model = None

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), str(RESULTS / 'vit_mlp_classifier_best.pth'))
            best_model = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch+1}/{num_epochs} --> "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the final model
    torch.save(model.state_dict(), str(RESULTS / 'vit_mlp_classifier_last.pth'))
    # Load the best model weights before inference
    model.load_state_dict(best_model)

    # -------------------------
    # Run Inference and save predictions to CSV
    # -------------------------
    run_testing(model, test_csv, batch_size, device, output_csv=str(RESULTS / 'predictions.csv'), metrics_csv=str(RESULTS / 'metrics.csv'))
    print("Testing completed:")

if __name__ == '__main__':
    main()