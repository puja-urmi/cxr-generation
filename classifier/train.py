"""
Training script for X-ray binary classifier
"""

import os
import time
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import config
from data_loaders import ChestXRayDataset, ModelSpecificDataLoaders
from model import get_model, load_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for X-ray classification model
    """
    def __init__(self, img_dir, resume=False):
        """
        Initialize trainer
        
        Args:
            img_dir: Directory with images
            resume: Whether to resume from checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Get data loaders using config settings
        dataloaders = ModelSpecificDataLoaders.create_dataloaders(
            data_dir=img_dir
            # All other parameters will use config defaults
        )
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val'] 
        self.test_loader = dataloaders['test']
        logger.info(f"Data loaders created. Train: {len(self.train_loader.dataset)}, "
                   f"Val: {len(self.val_loader.dataset)}, Test: {len(self.test_loader.dataset)}")
        
        # Initialize model
        self.model = get_model(self.device)
        
        # Define loss and optimizer using config settings
        # Class weights disabled since dataset is now balanced (50-50 split)
        self.criterion = nn.CrossEntropyLoss()
        
        # Create optimizer based on config
        if config.OPTIMIZER == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        else:
            raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")
        
        # Create scheduler based on config
        if config.USE_SCHEDULER:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)
        else:
            self.scheduler = None
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Initialize TensorBoard writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(config.TENSORBOARD_DIR, f"{config.MODEL_NAME}_{current_time}")
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs will be saved to {log_dir}")
        
        # Resume from checkpoint if requested
        if resume and os.path.exists(config.CHECKPOINT_PATH):
            self._resume_from_checkpoint()
    
    def _resume_from_checkpoint(self):
        """Resume training from checkpoint"""
        try:
            self.model, checkpoint = load_checkpoint(self.model, config.CHECKPOINT_PATH)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['val_loss']
            logger.info(f"Resumed from epoch {self.start_epoch} with validation loss {self.best_val_loss:.4f}")
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        start_time = time.time()
        for i, batch in enumerate(self.train_loader):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Log progress
            if (i+1) % 20 == 0:
                logger.info(f'Epoch [{epoch+1}/{config.EPOCHS}], Step [{i+1}/{len(self.train_loader)}], '
                           f'Loss: {loss.item():.4f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_time = time.time() - start_time
        
        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        self.writer.add_scalar('Time/train', epoch_time, epoch)
        
        logger.info(f'Epoch [{epoch+1}/{config.EPOCHS}], '
                   f'Train Loss: {epoch_loss:.4f}, '
                   f'Train Acc: {epoch_acc:.4f}, '
                   f'Time: {epoch_time:.2f}s')
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model with class-specific metrics"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability for positive class
        
        # Calculate metrics
        val_loss = val_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='binary')
        val_recall = recall_score(all_labels, all_preds, average='binary')
        val_f1 = f1_score(all_labels, all_preds, average='binary')
        val_auc = roc_auc_score(all_labels, all_probs)
        
        # Calculate class-specific metrics to monitor imbalance
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(all_labels, all_preds)
        
        # Class-specific accuracy
        normal_correct = cm[0, 0] if cm.shape[0] > 0 else 0
        normal_total = cm[0, 0] + cm[0, 1] if cm.shape[0] > 0 else 1
        pneumonia_correct = cm[1, 1] if cm.shape[0] > 1 else 0
        pneumonia_total = cm[1, 0] + cm[1, 1] if cm.shape[0] > 1 else 1
        
        normal_acc = normal_correct / normal_total if normal_total > 0 else 0
        pneumonia_acc = pneumonia_correct / pneumonia_total if pneumonia_total > 0 else 0
        
        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/validation', val_acc, epoch)
        self.writer.add_scalar('Accuracy/normal', normal_acc, epoch)
        self.writer.add_scalar('Accuracy/pneumonia', pneumonia_acc, epoch)
        self.writer.add_scalar('Precision/validation', val_precision, epoch)
        self.writer.add_scalar('Recall/validation', val_recall, epoch)
        self.writer.add_scalar('F1/validation', val_f1, epoch)
        self.writer.add_scalar('AUC/validation', val_auc, epoch)
        
        # Log example predictions occasionally
        if epoch % 5 == 0:
            self._log_prediction_examples(epoch)
        
        # Enhanced logging with class-specific metrics
        logger.info(f'Epoch [{epoch+1}/{config.EPOCHS}] VALIDATION RESULTS:')
        logger.info(f'  Overall - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}')
        logger.info(f'  Class-specific - Normal Acc: {normal_acc:.4f} ({normal_correct}/{normal_total}), '
                   f'Pneumonia Acc: {pneumonia_acc:.4f} ({pneumonia_correct}/{pneumonia_total})')
        logger.info(f'  Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
        
        # Log confusion matrix details
        if cm.shape[0] >= 2 and cm.shape[1] >= 2:
            logger.info(f'  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}')
        
        # Update learning rate based on validation loss
        if self.scheduler is not None:
            self.scheduler.step(val_loss)
        
        # Save checkpoint if best model so far
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint(epoch, val_loss)
            logger.info(f'Saved new best model with val_loss: {val_loss:.4f}')
        
        return val_loss, val_acc, val_precision, val_recall, val_f1, val_auc
    
    def _log_prediction_examples(self, epoch):
        """Log example predictions to TensorBoard"""
        self.model.eval()
        
        # Get a batch of validation images
        for batch in self.val_loader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
            
            # Take first 8 examples
            img_grid = make_grid(images[:8].cpu(), nrow=4, normalize=True)
            self.writer.add_image('Predictions', img_grid, global_step=epoch)
            break
    
    def _save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        torch.save(checkpoint, config.CHECKPOINT_PATH)
    
    def train(self):
        """Full training loop"""
        logger.info(f"Starting training for {config.EPOCHS} epochs")
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': []
        }
        
        # Early stopping setup
        patience_counter = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.start_epoch, config.EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = self.validate(epoch)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Close the TensorBoard writer
        self.writer.close()
        
        logger.info("Training completed")
        logger.info(f"TensorBoard logs saved to {self.writer.log_dir}")
        
        return history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train X-ray classifier")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    trainer = Trainer(img_dir=args.img_dir, resume=args.resume)
    history = trainer.train()