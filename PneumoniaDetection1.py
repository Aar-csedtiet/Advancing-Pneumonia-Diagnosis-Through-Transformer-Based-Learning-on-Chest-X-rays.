# -*- coding: utf-8 -*-
"""
Pneumonia detection master training script.
- Dataset: folder structure ./chest_xray/Normal and ./chest_xray/Pneumonia
- Backbones: ViT (google/vit-base-patch16-224), DeiT (facebook/deit-base-patch16-224), MambaVision (nvidia/MambaVision-S-1K)
- Preprocessors: CLAHE, SRCNN, SRGAN
- No clinical features, no CSV required
- No early stopping, epochs fixed to 50
- Saves best model (w.r.t validation loss) for each experiment
"""
import os
import sys
import pathlib
import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef, precision_recall_curve,
    average_precision_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import ViTModel, AutoModel
from torch.cuda.amp import GradScaler, autocast
from scipy.stats import chi2_contingency

# ----------------- Reproducibility & Device -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def clean_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ----------------- General Config -----------------
BATCH_SIZE = 32
NUM_EPOCHS = 50           # per your request
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
NUM_WORKERS = 8
DATA_ROOT = './chest_xray'  # change if needed

# ----------------- Pre-processor Implementations -----------------
class CLAHEPreprocessor(nn.Module):
    """Apply CLAHE per-channel using OpenCV. Works on tensor input [B,3,H,W] float [0,1]."""
    def __init__(self, clipLimit=2.0, tileGridSize=(8,8)):
        super(CLAHEPreprocessor, self).__init__()
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def forward(self, x):
        # x: torch.Tensor on any device with shape (B,3,H,W) and values ~[0,1]
        cpu = x.detach().cpu().numpy()
        B, C, H, W = cpu.shape
        out = np.zeros_like(cpu, dtype=np.float32)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        for b in range(B):
            img = (cpu[b].transpose(1,2,0) * 255.0).astype(np.uint8)  # H,W,3
            # convert to LAB and apply CLAHE on L channel to preserve colors
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b_ch = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b_ch))
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            out[b] = (rgb.astype(np.float32) / 255.0).transpose(2,0,1)
        return torch.from_numpy(out).to(x.device)

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features), nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorSRGAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=8):
        super(GeneratorSRGAN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())
        res_blocks = [ResidualBlock(64) for _ in range(n_residual_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64))
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.conv3(out)
        # SRGAN generator typically outputs -1..1 after tanh; bring to 0..1
        out = (out + 1.0) / 2.0
        return out

# ----------------- Master Model Definition -----------------
class CustomModel(nn.Module):
    def __init__(self, model_architecture, preprocessor=None, num_classes=2):
        super(CustomModel, self).__init__()
        self.model_architecture = model_architecture
        self.preprocessor = preprocessor
        if self.preprocessor:
            for param in self.preprocessor.parameters():
                param.requires_grad = False

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Load backbone
        try:
            if self.model_architecture == 'vit':
                self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
            elif self.model_architecture == 'deit':
                # use AutoModel for DeiT
                self.backbone = AutoModel.from_pretrained("facebook/deit-base-patch16-224")
            elif self.model_architecture == 'mamba':
                self.backbone = AutoModel.from_pretrained("nvidia/MambaVision-S-1K", trust_remote_code=True)
            else:
                raise ValueError(f"Unsupported architecture: {self.model_architecture}")
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-trained model for '{self.model_architecture}'. Error: {e}")

        # Freeze backbone then unfreeze last few layers (best-effort; names vary across models)
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Attempt to unfreeze last layers (safe no-op if names differ)
        for name, param in self.backbone.named_parameters():
            if any(k in name for k in ['encoder.layer.10', 'encoder.layer.11', 'stages.3', 'layernorm', 'norm', 'pooler']):
                param.requires_grad = True

        # get output dim
        cfg = getattr(self.backbone, 'config', None)
        if cfg is not None:
            output_dim = getattr(cfg, 'hidden_size', None) or (getattr(cfg, 'hidden_sizes', None) and cfg.hidden_sizes[-1]) or 768
        else:
            output_dim = 768

        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        # images: tensor B,3,H,W, values approx 0..1
        if self.preprocessor:
            with torch.no_grad():
                processed_images = torch.clamp(self.preprocessor(images), 0.0, 1.0)
        else:
            processed_images = images

        normalized_images = self.normalize(processed_images)

        # pass through backbone with flexible handling of return types
        try:
            if self.model_architecture == 'vit':
                outputs = self.backbone(pixel_values=normalized_images)
            else:
                outputs = self.backbone(normalized_images)
        except TypeError:
            # fallback: try positional call
            outputs = self.backbone(normalized_images)

        # extract feature vector (CLS token or pooled output)
        if hasattr(outputs, 'last_hidden_state'):
            feat = outputs.last_hidden_state
            if feat.dim() == 3:
                image_features = feat[:, 0, :]
            else:
                image_features = feat
        elif isinstance(outputs, tuple):
            # many AutoModels return (last_hidden_state, ...)
            out0 = outputs[0]
            if out0.dim() == 3:
                image_features = out0[:, 0, :]
            else:
                image_features = out0
        else:
            image_features = outputs
            if image_features.dim() == 3:
                image_features = image_features[:, 0, :]

        return self.classifier(image_features)

# ----------------- Dataset -----------------
class ChestXrayDataset(Dataset):
    def __init__(self, records, transform=None):
        """
        records: list of (image_path, label_int)
        transform: torchvision transforms to apply to PIL image (returns tensor)
        """
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img_t = self.transform(img)
            else:
                img_t = transforms.ToTensor()(img)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}. Using black placeholder.")
            img_t = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
        return img_t, torch.tensor(label, dtype=torch.long)

def build_records_from_folder(root_dir):
    """
    Expects:
    root_dir/
        Pneumonia/
        Normal/
    returns list of (path, label) with label 1 for Pneumonia, 0 for Normal
    """
    classes = {'Normal': 0, 'Pneumonia': 1}
    records = []
    for cls_name, label in classes.items():
        folder = os.path.join(root_dir, cls_name)
        if not os.path.isdir(folder):
            print(f"Warning: expected folder {folder} not found.")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                records.append((os.path.join(folder, fname), label))
    if not records:
        raise RuntimeError(f"No images found under {root_dir}. Check your dataset.")
    return records

# ----------------- Training & Evaluation -----------------
def run_experiment(config, train_loader, val_loader, test_loader):
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n----- Running Experiment: {config['name']} -----")
    clean_gpu_memory()

    preprocessor = config['preprocessor']() if config['preprocessor'] else None
    model = CustomModel(model_architecture=config['model_architecture'], preprocessor=preprocessor).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", unit="batch")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            preds = torch.max(outputs, 1)[1]
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct_train/total_train:.4f}")

        train_losses.append(running_loss / total_train if total_train>0 else 0.0)
        train_accuracies.append(correct_train / total_train if total_train>0 else 0.0)

        # validation
        model.eval()
        val_running_loss, correct_val, total_val = 0.0, 0, 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Valid]", unit="batch")
        with torch.no_grad():
            for images, labels in pbar_val:
                images = images.to(device)
                labels = labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_running_loss += loss.item() * labels.size(0)
                preds = torch.max(outputs, 1)[1]
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                pbar_val.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct_val/total_val:.4f}")

        epoch_val_loss = val_running_loss / total_val if total_val>0 else float('inf')
        val_losses.append(epoch_val_loss)
        val_accuracies.append(correct_val / total_val if total_val>0 else 0.0)
        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch+1} Summary | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accuracies[-1]:.4f}")

        # save best
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    # testing using best model
    print("Evaluating on test set...")
    try:
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
    except Exception as e:
        print(f"Warning: Could not load best model: {e}. Using current weights.")
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            with autocast():
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.max(probs, 1)[1].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    all_probs_pos = all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs[:, 0]

    # metrics
    metrics = {}
    try:
        cm = confusion_matrix(all_labels, all_preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # handle weird cases
            tn = fp = fn = tp = 0

        metrics['Accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['Precision'] = precision_score(all_labels, all_preds, zero_division=0)
        metrics['Recall (Sensitivity)'] = recall_score(all_labels, all_preds, zero_division=0)
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['F1-Score (Dice)'] = f1_score(all_labels, all_preds, zero_division=0)
        metrics['AUC'] = roc_auc_score(all_labels, all_probs_pos) if len(np.unique(all_labels)) > 1 else 0.0
        metrics['Kappa'] = cohen_kappa_score(all_labels, all_preds)
        metrics['MCC'] = matthews_corrcoef(all_labels, all_preds)
        metrics['DOR'] = (tp * tn) / ((fp * fn) + 1e-9)

        confidences = np.max(all_probs, axis=1)
        correct_mask = np.array(all_labels) == np.array(all_preds)
        metrics['Avg Confidence (Correct)'] = np.mean(confidences[correct_mask]) if np.any(correct_mask) else 0.0
        metrics['Avg Confidence (Incorrect)'] = np.mean(confidences[~correct_mask]) if np.any(~correct_mask) else 0.0
    except Exception as e:
        print(f"ERROR: Could not calculate metrics for {config['name']}. Reason: {e}")
        return None

    # Save metrics
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"--- Evaluation Results for {config['name']} ---\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # Plotting
    try:
        # Accuracy/Loss Curves
        plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1)
        plt.plot(train_accuracies, label='Training Accuracy'); plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True); plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Training Loss'); plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'training_curves.pdf')); plt.close()

        # Confusion Matrix
        plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix'); plt.savefig(os.path.join(output_dir, 'confusion_matrix.pdf')); plt.close()

        # ROC and PR Curves
        fpr, tpr, _ = roc_curve(all_labels, all_probs_pos)
        precision, recall, _ = precision_recall_curve(all_labels, all_probs_pos)
        avg_precision = average_precision_score(all_labels, all_probs_pos)

        plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.2f}"); plt.plot([0,1],[0,1], 'r--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2); plt.plot(recall, precision, label=f"AP = {avg_precision:.2f}")
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'roc_pr_curves.pdf')); plt.close()
    except Exception as e:
        print(f"ERROR: Could not generate individual plots for {config['name']}. Reason: {e}")

    print(f"----- Experiment {config['name']} Finished. Results saved to {output_dir} -----")
    return {
        'name': config['name'], 'metrics': metrics,
        'histories': {'train_loss': train_losses, 'val_loss': val_losses, 'train_acc': train_accuracies, 'val_acc': val_accuracies},
        'curves': {'fpr': fpr, 'tpr': tpr, 'precision': precision, 'recall': recall},
        'cm': cm
    }

# ----------------- Comparative Plotting & Statistical Analysis -----------------
def plot_comparative_metrics_heatmap(results_data, output_dir):
    try:
        df = pd.DataFrame([res['metrics'] for res in results_data], index=[res['name'] for res in results_data])
        plt.figure(figsize=(16, 12))
        sns.heatmap(df, annot=True, fmt=".3f", cmap="viridis")
        plt.title('Comparative Performance Metrics', fontsize=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparative_metrics_heatmap.pdf'), dpi=300)
        plt.close()
        print("Comparative metrics heatmap saved successfully.")
    except Exception as e:
        print(f"ERROR: Could not generate comparative metrics heatmap. Reason: {e}")

def plot_all_curves(results_data, output_dir):
    try:
        plt.figure(figsize=(18, 8)); plt.subplot(1, 2, 1)
        for res in results_data:
            plt.plot(res['curves']['fpr'], res['curves']['tpr'], label=f"{res['name']} (AUC={res['metrics']['AUC']:.3f})")
        plt.plot([0,1],[0,1], 'r--', label='Chance'); plt.title('Comparative ROC Curves'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(fontsize=8); plt.grid(True)

        plt.subplot(1, 2, 2)
        for res in results_data:
            plt.plot(res['curves']['recall'], res['curves']['precision'], label=f"{res['name']}")
        plt.title('Comparative PR Curves'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(fontsize=8); plt.grid(True)

        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'comparative_roc_pr_curves.pdf'), dpi=300); plt.close()
        print("Comparative ROC and PR curve plot saved successfully.")
    except Exception as e:
        print(f"ERROR: Could not generate comparative ROC/PR curves plot. Reason: {e}")

def perform_statistical_analysis(results_data, baseline_name='vit_clahe'):
    print("\n----- Performing Statistical Analysis -----")
    try:
        if not results_data:
            print("No results to analyze."); return

        baseline_res = next((res for res in results_data if res['name'] == baseline_name), None)
        if not baseline_res:
            print(f"Baseline model '{baseline_name}' not found for comparison."); return

        best_res = max(results_data, key=lambda x: x['metrics']['AUC'])

        print(f"Baseline Model: {baseline_res['name']} (AUC: {baseline_res['metrics']['AUC']:.4f})")
        print(f"Best Model: {best_res['name']} (AUC: {best_res['metrics']['AUC']:.4f})")

        if best_res['name'] == baseline_res['name']:
            print("Best model is the baseline model. No comparison needed."); return

        best_cm = best_res['cm']; base_cm = baseline_res['cm']

        contingency_table = np.array([
            [best_cm.diagonal().sum(), best_cm.sum() - best_cm.diagonal().sum()],
            [base_cm.diagonal().sum(), base_cm.sum() - base_cm.diagonal().sum()]
        ])

        if contingency_table.sum() == 0 or np.any(contingency_table.sum(axis=1) == 0):
             print("Skipping Chi-Squared test due to invalid contingency table (e.g., zero predictions).")
             return

        chi2, p, _, _ = chi2_contingency(contingency_table)

        print(f"\nChi-Squared Test between '{best_res['name']}' and '{baseline_res['name']}':")
        print(f"Chi2 Statistic: {chi2:.4f}, P-value: {p:.4f}")

        if p < 0.05:
            print("Result is statistically significant (p < 0.05). The difference in performance is unlikely due to chance.")
        else:
            print("Result is not statistically significant (p >= 0.05). We cannot conclude a real difference in performance.")
    except Exception as e:
        print(f"ERROR: Could not perform statistical analysis. Reason: {e}")

# ----------------- Main Execution -----------------
if __name__ == '__main__':
    try:
        # build dataset records
        records = build_records_from_folder(DATA_ROOT)
        # simple dataframe for stratified splitting
        df = pd.DataFrame(records, columns=['image_path', 'label'])

        train_val_df, test_df = train_test_split(df, test_size=TEST_SPLIT, stratify=df['label'], random_state=SEED)
        train_df, val_df = train_test_split(train_val_df, test_size=VAL_SPLIT/(1-TEST_SPLIT), stratify=train_val_df['label'], random_state=SEED)

        train_transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor()
        ])
        val_test_transforms = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

        train_dataset = ChestXrayDataset(list(train_df.itertuples(index=False, name=None)), transform=train_transforms)
        val_dataset = ChestXrayDataset(list(val_df.itertuples(index=False, name=None)), transform=val_test_transforms)
        test_dataset = ChestXrayDataset(list(test_df.itertuples(index=False, name=None)), transform=val_test_transforms)

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)

        # Experiment configurations (3 backbones x 3 preprocessors = 9)
        architectures = ['vit', 'deit', 'mamba']
        preprocessors = {'clahe': CLAHEPreprocessor, 'srcnn': SRCNN, 'srgan': GeneratorSRGAN}
        configurations = []
        for arch in architectures:
            for preproc_name, preproc_class in preprocessors.items():
                configurations.append({
                    'name': f"{arch}_{preproc_name}",
                    'model_architecture': arch,
                    'preprocessor': preproc_class,
                    'output_dir': f"./results_new/{arch}/{arch}_{preproc_name}"
                })

        all_results = []
        comparative_plots_dir = './comparative_plots_new'
        os.makedirs(comparative_plots_dir, exist_ok=True)

        for config in configurations:
            try:
                result = run_experiment(config, train_loader, val_loader, test_loader)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"Experiment {config['name']} failed: {e}")
                import traceback; traceback.print_exc()

        if all_results:
            plot_comparative_metrics_heatmap(all_results, comparative_plots_dir)
            plot_all_curves(all_results, comparative_plots_dir)
            # choose a reasonable baseline name present in your configs, e.g., 'vit_clahe'
            perform_statistical_analysis(all_results, baseline_name='vit_clahe')
        else:
            print("No experiments completed successfully. Skipping final analysis.")

        print("\nMaster script execution finished.")
    except Exception as e:
        print(f"A fatal error occurred in the main execution block: {e}")
        import traceback; traceback.print_exc()
