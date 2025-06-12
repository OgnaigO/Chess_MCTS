import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from chessmc.model import ImprovedChessModel, TransformerChessModel
from torch.cuda.amp import autocast, GradScaler

# Dataset import
try:
    from chessmc.data.dataset import ImprovedChessDataset
except ImportError:
    from data.dataset import ImprovedChessDataset

class ImprovedMultiTaskLoss(nn.Module):
    """
    Multi-task loss combining policy cross-entropy and value MSE.
    Includes label smoothing for the policy head.
    """
    def __init__(self, policy_weight=1.0, value_weight=1.0, label_smoothing=0.1):
        super().__init__()
        self.policy_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.value_loss = nn.MSELoss()
        self.policy_weight = policy_weight
        self.value_weight = value_weight

    def forward(self, policy_logits, policy_target, value_pred, value_target):
        policy_target = policy_target.clamp(0, policy_logits.size(1)-1)
        loss_p = self.policy_loss(policy_logits, policy_target)
        loss_v = self.value_loss(value_pred.view(-1), value_target)
        return self.policy_weight * loss_p + self.value_weight * loss_v

class AdvancedTrainerConfig:
    def __init__(self):
        # Model settings
        self.model_type    = 'cnn'  # choose 'cnn' or 'transformer'
        self.device        = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Data paths
        self.train_data    = 'processed/train.npz'
        self.val_data      = 'processed/val.npz'
        # Save directory
        self.save_dir      = 'models'
        os.makedirs(self.save_dir, exist_ok=True)
        # Training hyperparameters
        self.batch_size    = 64
        self.num_epochs    = 50
        self.learning_rate = 1e-4
        self.weight_decay  = 1e-4
        self.use_fp16      = False
        self.num_workers   = 0
        # Loss weighting
        self.policy_weight    = 1.0
        self.value_weight     = 1.0
        self.label_smoothing  = 0.1
        # Sampling
        self.use_uncertainty  = True
        # Logging and early stopping
        self.log_dir               = 'runs'
        self.early_stopping_patience = 5

class AdvancedTrainer:
    def __init__(self, cfg: AdvancedTrainerConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)

        # Model selection
        if cfg.model_type == 'cnn':
            self.model = ImprovedChessModel()
        else:
            self.model = TransformerChessModel()
        self.model.to(self.device)

        # Loss, optimizer, scaler
        self.criterion = ImprovedMultiTaskLoss(
            policy_weight=cfg.policy_weight,
            value_weight=cfg.value_weight,
            label_smoothing=cfg.label_smoothing
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        if cfg.use_fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        # Early stopping state
        self.early_stopping_patience = cfg.early_stopping_patience
        self.no_improve = 0
        self.best_loss = float('inf')

        # Datasets and loaders
        train_ds = ImprovedChessDataset(cfg.train_data)
        val_ds   = ImprovedChessDataset(cfg.val_data)
        if cfg.use_uncertainty:
            labels = train_ds.targets[:, 2].astype(np.int64)
            class_counts = np.bincount(labels)
            weights = 1.0 / class_counts
            sample_weights = weights[labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            self.train_loader = DataLoader(
                train_ds, batch_size=cfg.batch_size, sampler=sampler,
                num_workers=cfg.num_workers
            )
        else:
            self.train_loader = DataLoader(
                train_ds, batch_size=cfg.batch_size, shuffle=True,
                num_workers=cfg.num_workers
            )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.train_loader, desc=f"Train {epoch}/{self.cfg.num_epochs}")
        for board, policy_tgt, value_tgt in pbar:
            board      = board.to(self.device)
            policy_tgt = policy_tgt.to(self.device)
            value_tgt  = value_tgt.to(self.device)
            self.optimizer.zero_grad()
            if self.scaler:  # FP16
                with autocast():
                    value_pred, policy_logits, _ = self.model(board)
                    loss = self.criterion(policy_logits, policy_tgt, value_pred, value_tgt)
                if not torch.isfinite(loss):
                    print(f"Warning: non-finite loss at epoch {epoch}, skip batch")
                    continue
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:           # FP32
                value_pred, policy_logits, _ = self.model(board)
                loss = self.criterion(policy_logits, policy_tgt, value_pred, value_tgt)
                if not torch.isfinite(loss):
                    print(f"Warning: non-finite loss at epoch {epoch}, skip batch")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            # Metrics
            preds = policy_logits.argmax(dim=1)
            correct += (preds == policy_tgt).sum().item()
            total += board.size(0)
            total_loss += loss.item() * board.size(0)

        avg_loss = total_loss / len(self.train_loader.dataset)
        accuracy = correct / total
        print(f"Epoch {epoch} train loss: {avg_loss:.4f}, acc: {accuracy:.4f}")
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/train', accuracy, epoch)
        return avg_loss

    def validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validate {epoch}/{self.cfg.num_epochs}")
            for board, policy_tgt, value_tgt in pbar:
                board      = board.to(self.device)
                policy_tgt = policy_tgt.to(self.device)
                value_tgt  = value_tgt.to(self.device)

                value_pred, policy_logits, _ = self.model(board)
                loss = self.criterion(policy_logits, policy_tgt, value_pred, value_tgt)
                # Metrics
                preds = policy_logits.argmax(dim=1)
                correct += (preds == policy_tgt).sum().item()
                total += board.size(0)
                total_loss += loss.item() * board.size(0)
        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = correct / total
        print(f"Epoch {epoch} val loss: {avg_loss:.4f}, acc: {accuracy:.4f}")
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/val', accuracy, epoch)
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn_state_dict': self.criterion.state_dict()
        }
        path = os.path.join(self.cfg.save_dir, f"{self.cfg.model_type}_epoch{epoch}.pth")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        if is_best:
            best_path = os.path.join(self.cfg.save_dir, f"{self.cfg.model_type}_best.pth")
            torch.save(checkpoint, best_path)
            print(f"✅ New best model saved: {best_path}")

    def run(self):
        for epoch in range(1, self.cfg.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.validate_epoch(epoch)
            self.scheduler.step(val_loss)
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                self.no_improve = 0
            else:
                self.no_improve += 1
            self.save_checkpoint(epoch, is_best)
            if self.no_improve >= self.early_stopping_patience:
                print(f"No improvement for {self.early_stopping_patience} epochs. Early stopping.")
                break

if __name__ == '__main__':
    cfg = AdvancedTrainerConfig()
    trainer = AdvancedTrainer(cfg)
    trainer.run()
