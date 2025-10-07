"""
Preprocessing Utilities for Chest X-Ray Dataset
===============================================

This module provides preprocessing utilities, training helpers, and 
configuration management for the chest X-ray pneumonia detection project.

Key Features:
- Training configuration management
- Model factory for DenseNet and EfficientNet
- Training loop utilities
- Evaluation metrics
- Visualization tools
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd


class ConfigManager:
    """Configuration management for training experiments."""
    
    DEFAULT_CONFIGS = {
        'densenet121': {
            'model_name': 'densenet121',
            'image_size': [224, 224],
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'epochs': 50,
            'patience': 10,
            'augmentation': 'moderate',
            'optimizer': 'adam',
            'scheduler': 'reduce_on_plateau',
            'pretrained': True,
            'freeze_backbone': False,
            'dropout': 0.5
        },
        'densenet169': {
            'model_name': 'densenet169',
            'image_size': [224, 224],
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'epochs': 50,
            'patience': 10,
            'augmentation': 'moderate',
            'optimizer': 'adam',
            'scheduler': 'reduce_on_plateau',
            'pretrained': True,
            'freeze_backbone': False,
            'dropout': 0.5
        },
        'densenet201': {
            'model_name': 'densenet201',
            'image_size': [224, 224],
            'batch_size': 16,
            'learning_rate': 0.0005,
            'weight_decay': 0.0001,
            'epochs': 50,
            'patience': 10,
            'augmentation': 'moderate',
            'optimizer': 'adam',
            'scheduler': 'reduce_on_plateau',
            'pretrained': True,
            'freeze_backbone': False,
            'dropout': 0.5
        },
        'efficientnet_b0': {
            'model_name': 'efficientnet_b0',
            'image_size': [224, 224],
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'epochs': 50,
            'patience': 10,
            'augmentation': 'strong',
            'optimizer': 'adamw',
            'scheduler': 'cosine_annealing',
            'pretrained': True,
            'freeze_backbone': False,
            'dropout': 0.3
        },
        'efficientnet_b1': {
            'model_name': 'efficientnet_b1',
            'image_size': [240, 240],
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'epochs': 50,
            'patience': 10,
            'augmentation': 'strong',
            'optimizer': 'adamw',
            'scheduler': 'cosine_annealing',
            'pretrained': True,
            'freeze_backbone': False,
            'dropout': 0.3
        },
        'efficientnet_b3': {
            'model_name': 'efficientnet_b3',
            'image_size': [300, 300],
            'batch_size': 16,
            'learning_rate': 0.0005,
            'weight_decay': 0.00001,
            'epochs': 50,
            'patience': 10,
            'augmentation': 'strong',
            'optimizer': 'adamw',
            'scheduler': 'cosine_annealing',
            'pretrained': True,
            'freeze_backbone': False,
            'dropout': 0.3
        },
        'efficientnet_b4': {
            'model_name': 'efficientnet_b4',
            'image_size': [380, 380],
            'batch_size': 8,
            'learning_rate': 0.0003,
            'weight_decay': 0.00001,
            'epochs': 50,
            'patience': 15,
            'augmentation': 'strong',
            'optimizer': 'adamw',
            'scheduler': 'cosine_annealing',
            'pretrained': True,
            'freeze_backbone': False,
            'dropout': 0.4
        }
    }
    
    @classmethod
    def get_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_name not in cls.DEFAULT_CONFIGS:
            raise ValueError(f"Model {model_name} not supported. Available: {list(cls.DEFAULT_CONFIGS.keys())}")
        return cls.DEFAULT_CONFIGS[model_name].copy()
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], filepath: str):
        """Save configuration to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
        elif filepath.suffix in ['.yml', '.yaml']:
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError("Config file must be .json or .yaml/.yml")
    
    @classmethod
    def load_config(cls, filepath: str) -> Dict[str, Any]:
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.suffix in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError("Config file must be .json or .yaml/.yml")


class TrainingMetrics:
    """Utilities for tracking and analyzing training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def update(self, epoch: int, train_loss: float, train_acc: float, 
               val_loss: float, val_acc: float, lr: float):
        """Update metrics for current epoch."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rate'].append(lr)
    
    def get_best_epoch(self) -> Tuple[int, float]:
        """Get epoch with best validation accuracy."""
        best_idx = np.argmax(self.history['val_acc'])
        return self.history['epoch'][best_idx], self.history['val_acc'][best_idx]
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        epochs = self.history['epoch']
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.history['train_acc'], label='Train Acc', color='blue')
        axes[0, 1].plot(epochs, self.history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.history['learning_rate'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss vs Accuracy scatter
        axes[1, 1].scatter(self.history['val_loss'], self.history['val_acc'], 
                          c=epochs, cmap='viridis', alpha=0.7)
        axes[1, 1].set_title('Val Loss vs Val Accuracy')
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_history(self, filepath: str):
        """Save training history to CSV."""
        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)
        print(f"Training history saved to: {filepath}")
    
    def load_history(self, filepath: str):
        """Load training history from CSV."""
        df = pd.read_csv(filepath)
        self.history = df.to_dict('list')
        print(f"Training history loaded from: {filepath}")


class EvaluationTools:
    """Tools for model evaluation and analysis."""
    
    @staticmethod
    def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_prob: np.ndarray, class_names: List[str] = None) -> Dict[str, Any]:
        """Compute comprehensive classification metrics."""
        
        if class_names is None:
            class_names = ['NORMAL', 'PNEUMONIA']
        
        # Basic metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # AUC-ROC
        auc_roc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'per_class_metrics': {
                'precision': dict(zip(class_names, precision_per_class)),
                'recall': dict(zip(class_names, recall_per_class)),
                'f1_score': dict(zip(class_names, f1_per_class))
            }
        }
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str] = None, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        
        if class_names is None:
            class_names = ['NORMAL', 'PNEUMONIA']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                      save_path: Optional[str] = None):
        """Plot ROC curve."""
        
        # Get probabilities for positive class
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
        auc = roc_auc_score(y_true, y_prob_pos)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fpr, tpr, auc


class ExperimentManager:
    """Manage training experiments and results."""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / 'configs').mkdir(exist_ok=True)
        (self.experiment_dir / 'models').mkdir(exist_ok=True)
        (self.experiment_dir / 'logs').mkdir(exist_ok=True)
        (self.experiment_dir / 'plots').mkdir(exist_ok=True)
        (self.experiment_dir / 'results').mkdir(exist_ok=True)
    
    def create_experiment(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """Create a new experiment."""
        exp_dir = self.experiment_dir / experiment_name
        exp_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_path = exp_dir / 'config.yaml'
        ConfigManager.save_config(config, str(config_path))
        
        print(f"Created experiment: {experiment_name}")
        print(f"Experiment directory: {exp_dir}")
        
        return str(exp_dir)
    
    def list_experiments(self) -> List[str]:
        """List all experiments."""
        experiments = []
        for item in self.experiment_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                experiments.append(item.name)
        return sorted(experiments)
    
    def get_experiment_summary(self) -> pd.DataFrame:
        """Get summary of all experiments."""
        experiments = []
        
        for exp_name in self.list_experiments():
            exp_dir = self.experiment_dir / exp_name
            config_path = exp_dir / 'config.yaml'
            results_path = exp_dir / 'results.json'
            
            if config_path.exists():
                config = ConfigManager.load_config(str(config_path))
                exp_info = {
                    'experiment': exp_name,
                    'model': config.get('model_name', 'unknown'),
                    'image_size': f"{config.get('image_size', [0, 0])[0]}x{config.get('image_size', [0, 0])[1]}",
                    'batch_size': config.get('batch_size', 0),
                    'learning_rate': config.get('learning_rate', 0),
                    'epochs': config.get('epochs', 0),
                    'completed': results_path.exists()
                }
                
                # Add results if available
                if results_path.exists():
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    exp_info.update({
                        'best_val_acc': results.get('best_val_acc', 0),
                        'test_acc': results.get('test_acc', 0),
                        'auc_roc': results.get('auc_roc', 0)
                    })
                
                experiments.append(exp_info)
        
        return pd.DataFrame(experiments)


def create_sample_configs():
    """Create sample configuration files for different models."""
    
    output_dir = Path("/Users/pujasaha/Documents/cxp/xray-classifier/configs")
    output_dir.mkdir(exist_ok=True)
    
    print("CREATING SAMPLE CONFIGURATION FILES")
    print("=" * 50)
    
    for model_name in ConfigManager.DEFAULT_CONFIGS.keys():
        config = ConfigManager.get_config(model_name)
        
        # Add common paths
        config.update({
            'data_dir': '/Users/pujasaha/Documents/cxp/chest_xray',
            'experiment_dir': '/Users/pujasaha/Documents/cxp/xray-classifier/experiments',
            'device': 'cuda',
            'seed': 42,
            'num_workers': 4,
            'pin_memory': True
        })
        
        # Save configuration
        config_path = output_dir / f'{model_name}_config.yaml'
        ConfigManager.save_config(config, str(config_path))
        
        print(f"Created config: {config_path}")
    
    print(f"\nAll configuration files saved to: {output_dir}")


if __name__ == "__main__":
    # Demonstrate functionality
    print("PREPROCESSING UTILITIES DEMONSTRATION")
    print("=" * 50)
    
    # Create sample configs
    create_sample_configs()
    
    # Demonstrate metrics tracking
    print("\nDemonstrating metrics tracking...")
    metrics = TrainingMetrics()
    
    # Simulate some training data
    for epoch in range(1, 11):
        train_loss = 1.0 - epoch * 0.08 + np.random.normal(0, 0.05)
        train_acc = 0.5 + epoch * 0.04 + np.random.normal(0, 0.02)
        val_loss = 1.2 - epoch * 0.07 + np.random.normal(0, 0.08)
        val_acc = 0.45 + epoch * 0.042 + np.random.normal(0, 0.03)
        lr = 0.001 * (0.9 ** (epoch // 3))
        
        metrics.update(epoch, train_loss, train_acc, val_loss, val_acc, lr)
    
    best_epoch, best_acc = metrics.get_best_epoch()
    print(f"Best validation accuracy: {best_acc:.3f} at epoch {best_epoch}")
    
    print("\nUtilities ready for use!")