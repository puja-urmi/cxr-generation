"""
Evaluation script for X-ray binary classifier
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
from matplotlib import pyplot as plt
import seaborn as sns

import config
from data import get_dataloaders
from model import get_model, load_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(img_dir, checkpoint_path=None):
    """
    Evaluate trained model on test data
    
    Args:
        img_dir: Directory with images
        checkpoint_path: Path to model checkpoint (default: config.CHECKPOINT_PATH)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get data loaders (we only need the test loader)
    _, _, test_loader = get_dataloaders(img_dir)
    
    # Initialize model
    model = get_model(device)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINT_PATH
    
    try:
        model, checkpoint = load_checkpoint(model, checkpoint_path)
        logger.info(f"Loaded model from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    all_image_paths = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            # Store results
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob for positive class
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"Confusion Matrix:")
    logger.info(f"{cm}")
    
    # Print classification report
    class_report = classification_report(all_labels, all_preds, target_names=config.CLASSES)
    logger.info(f"Classification Report:\n{class_report}")
    
    # Create results directory if it doesn't exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
        'Value': [acc, precision, recall, f1, auc]
    })
    results_df.to_csv(os.path.join(config.RESULTS_DIR, 'test_results.csv'), index=False)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'))
    
    # Plot model performance
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    values = [acc, precision, recall, f1, auc]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color='skyblue')
    plt.ylim(0, 1.0)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    
    # Add value labels above bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'performance_metrics.png'))
    
    logger.info(f"Results and plots saved to {config.RESULTS_DIR}")
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate X-ray classifier")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    evaluate_model(img_dir=args.img_dir, checkpoint_path=args.checkpoint)