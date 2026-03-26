"""
TrustLens - Training Script
Fine-tune EfficientNet-B4 on deepfake datasets.

Supported datasets:
  - FaceForensics++ (FF++) — https://github.com/ondyari/FaceForensics
  - Celeb-DF v2         — https://github.com/yuezunli/celeb-deepfakeforensics
  - DFDC (Kaggle)       — https://www.kaggle.com/competitions/deepfake-detection-challenge

Usage:
  python train.py --data_dir /path/to/dataset --epochs 20 --batch_size 16

Dataset folder structure expected:
  data_dir/
    train/
      real/   ← real face images/frames
      fake/   ← deepfake images/frames
    val/
      real/
      fake/
"""

import os
import argparse
import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from deepfake_detector import EfficientNetDeepfakeDetector, FaceExtractor, extract_frequency_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Dataset that loads real/fake images from a folder structure.
    Applies face extraction and frequency analysis.
    """

    TRANSFORM_TRAIN = transforms.Compose([
        transforms.Resize((420, 420)),
        transforms.RandomCrop((380, 380)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    TRANSFORM_VAL = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, root_dir: str, split: str = 'train', use_face_extraction: bool = True):
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.use_face_extraction = use_face_extraction
        self.transform = self.TRANSFORM_TRAIN if split == 'train' else self.TRANSFORM_VAL
        self.face_extractor = FaceExtractor() if use_face_extraction else None

        self.samples = []
        self.labels = []

        EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        for label, class_name in [(0, 'real'), (1, 'fake')]:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in EXTENSIONS:
                    self.samples.append(str(img_path))
                    self.labels.append(label)

        logger.info(f"[{split}] Loaded {len(self.samples)} samples | "
                    f"Real: {self.labels.count(0)} | Fake: {self.labels.count(1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)

        if self.face_extractor:
            face_np = self.face_extractor.extract_face(image_np)
            image = Image.fromarray(face_np)
        
        freq_features = extract_frequency_features(np.array(image))
        freq_tensor = torch.tensor(freq_features, dtype=torch.float32)

        image_tensor = self.transform(image)
        return image_tensor, freq_tensor, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training on: {self.device}")

        # Model
        self.model = EfficientNetDeepfakeDetector(dropout_rate=0.4, pretrained=True)
        self.model = self.model.to(self.device)

        # Freeze backbone initially for warm-up
        if args.freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen for warm-up phase")

        # Loss, Optimizer, Scheduler
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, args.fake_weight]).to(self.device)
        )
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6
        )

        # Data
        train_dataset = DeepfakeDataset(args.data_dir, split='train')
        val_dataset = DeepfakeDataset(args.data_dir, split='val')

        self.train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True
        )

        self.best_auc = 0.0
        os.makedirs(args.output_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        # Unfreeze backbone after warm-up
        if epoch == self.args.unfreeze_epoch:
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info(f"Epoch {epoch}: Backbone unfrozen for full fine-tuning")
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.args.lr / 10, weight_decay=1e-4
            )

        for batch_idx, (images, freq_feats, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            freq_feats = freq_feats.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images, freq_feats)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch} | Step {batch_idx}/{len(self.train_loader)} | "
                            f"Loss: {loss.item():.4f} | Acc: {correct/total*100:.2f}%")

        return total_loss / len(self.train_loader), correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_probs, all_labels = [], []
        total_loss = 0

        for images, freq_feats, labels in self.val_loader:
            images = images.to(self.device)
            freq_feats = freq_feats.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images, freq_feats)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

        preds = [1 if p > 0.5 else 0 for p in all_probs]
        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, preds)

        logger.info(f"\nValidation Results:\n"
                    f"  Loss: {total_loss/len(self.val_loader):.4f}\n"
                    f"  Accuracy: {acc*100:.2f}%\n"
                    f"  ROC-AUC: {auc:.4f}\n"
                    f"{classification_report(all_labels, preds, target_names=['Real', 'Fake'])}")

        return total_loss / len(self.val_loader), acc, auc

    def save_checkpoint(self, epoch, val_auc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_auc': val_auc,
        }
        path = os.path.join(self.args.output_dir, f'checkpoint_epoch{epoch}.pth')
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.args.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"✅ New best model saved! AUC: {val_auc:.4f}")

    def run(self):
        history = []

        for epoch in range(1, self.args.epochs + 1):
            logger.info(f"\n{'='*50}\nEpoch {epoch}/{self.args.epochs}\n{'='*50}")

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_auc = self.validate()
            self.scheduler.step()

            is_best = val_auc > self.best_auc
            if is_best:
                self.best_auc = val_auc

            self.save_checkpoint(epoch, val_auc, is_best)

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc
            })

        # Save training history
        with open(os.path.join(self.args.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"\n🎉 Training complete! Best Val AUC: {self.best_auc:.4f}")
        logger.info(f"Best model saved to: {self.args.output_dir}/best_model.pth")


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='TrustLens - Deepfake Detector Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Where to save models')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--fake_weight', type=float, default=1.5, help='Class weight for fake samples')
    parser.add_argument('--freeze_backbone', action='store_true', default=True)
    parser.add_argument('--unfreeze_epoch', type=int, default=5, help='Epoch to unfreeze backbone')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()
