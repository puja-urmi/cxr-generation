"""
Visualization utilities for X-ray classifier results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch
from PIL import Image
import cv2
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Optional, Union
import logging

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_learning_curves(history: Dict, save_path: Optional[str] = None):
    """
    Plot training and validation learning curves
    
    Args:
        history: Dictionary with training history
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(history['val_f1'], label='Validation F1')
    plt.plot(history['val_precision'], label='Validation Precision')
    plt.plot(history['val_recall'], label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    
    # Plot AUC
    plt.subplot(2, 2, 4)
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Learning curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(labels: List[int], probs: List[float], save_path: Optional[str] = None):
    """
    Plot ROC curve
    
    Args:
        labels: True labels (binary)
        probs: Predicted probabilities for positive class
        save_path: Optional path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(labels: List[int], probs: List[float], save_path: Optional[str] = None):
    """
    Plot Precision-Recall curve
    
    Args:
        labels: True labels (binary)
        probs: Predicted probabilities for positive class
        save_path: Optional path to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Precision-Recall curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sample_predictions(
    image_paths: List[str], 
    predictions: List[int], 
    labels: List[int], 
    probabilities: List[float], 
    num_samples: int = 9,
    save_path: Optional[str] = None
):
    """
    Plot sample predictions with their confidence scores
    
    Args:
        image_paths: Paths to images
        predictions: Predicted classes
        labels: True labels
        probabilities: Predicted probabilities for positive class
        num_samples: Number of samples to display
        save_path: Optional path to save the plot
    """
    indices = np.random.choice(len(image_paths), min(num_samples, len(image_paths)), replace=False)
    
    plt.figure(figsize=(15, 15))
    
    n_cols = 3
    n_rows = (num_samples + 2) // n_cols
    
    for i, idx in enumerate(indices):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Load and display image
        try:
            img = Image.open(image_paths[idx]).convert('RGB')
            plt.imshow(img, cmap='gray')
        except Exception as e:
            plt.text(0.5, 0.5, f"Error loading image", ha='center', va='center')
        
        # Display prediction information
        pred = predictions[idx]
        label = labels[idx]
        prob = probabilities[idx]
        
        pred_class = config.CLASSES[pred]
        true_class = config.CLASSES[label]
        
        color = 'green' if pred == label else 'red'
        
        plt.title(f"Pred: {pred_class} ({prob:.2f})\nTrue: {true_class}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Sample predictions saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_grad_cam(model, img_path, target_layer_name=None, device=None):
    """
    Generate Grad-CAM visualization for model's decision
    
    Args:
        model: Trained model
        img_path: Path to image file
        target_layer_name: Name of layer to use for Grad-CAM
        device: Device to run on
        
    Returns:
        Tuple of (original image, heatmap, combined visualization)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For this simplified version, we'll use a predefined target layer (last conv layer)
    # In a full implementation, you'd want to properly hook into your specific model architecture
    
    # Load and preprocess image
    from data import get_transforms
    transform = get_transforms(train=False)
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    img_np = np.array(img)
    
    # Create a simple activation heatmap (this is a placeholder - full Grad-CAM would require hooks into model)
    # In a real implementation, this would extract activations and gradients from the target layer
    
    # Placeholder heatmap generation
    with torch.no_grad():
        features = model.backbone(img_tensor)
        output = model.classifier(features)
        
    # Create a dummy heatmap (in a real implementation, this would be calculated from gradients)
    heatmap = np.random.rand(img_np.shape[0] // 8, img_np.shape[1] // 8)
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Combine heatmap with original image
    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    return img_np, heatmap, superimposed_img


def visualize_model_predictions(
    model, 
    data_loader, 
    device, 
    num_images=6,
    save_path=None
):
    """
    Visualize model predictions on a batch of images
    
    Args:
        model: Trained model
        data_loader: DataLoader with images
        device: Device to run on
        num_images: Number of images to visualize
        save_path: Optional path to save the visualization
    """
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 3 + 1, 3, images_so_far)
                ax.set_title(f'Predicted: {config.CLASSES[preds[j]]}\nTrue: {config.CLASSES[labels[j]]}')
                
                # Convert tensor to image
                inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                
                plt.imshow(inp)
                plt.axis('off')
                
                if images_so_far == num_images:
                    if save_path:
                        plt.savefig(save_path)
                        logger.info(f"Model predictions visualization saved to {save_path}")
                    return
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Model predictions visualization saved to {save_path}")
        plt.tight_layout()
        plt.close()