"""
Test script for the trained DenseNet classifier on chest X-Ray data
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Add the classifier directory to the path
sys.path.append('/home/psaha03/scratch/cxr-generation/classifier')

import config
from data_loaders import ModelSpecificDataLoaders, ChestXRayDataset, MedicalImageTransforms
from model import XRayClassifier, load_checkpoint
from visualize_results import plot_roc_curve, plot_precision_recall_curve, plot_sample_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_model_on_dataset(model_path, test_data_dir, results_dir=None):
    """
    Test the trained model on the test dataset
    
    Args:
        model_path: Path to the trained model (.pt file)
        test_data_dir: Path to the test data directory
        results_dir: Directory to save results (optional)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set up results directory
    if results_dir is None:
        results_dir = '/home/psaha03/scratch/classifier/test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create test data loader
    logger.info("Setting up test data loader...")
    val_test_transform = MedicalImageTransforms.get_val_test_transforms(image_size=(224, 224))
    
    test_dataset = ChestXRayDataset(
        data_dir=test_data_dir,
        split='test',
        transform=val_test_transform,
        image_size=(224, 224)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Test dataset: {len(test_dataset)} images, {len(test_loader)} batches")
    
    # Initialize and load model
    logger.info("Loading trained model...")
    model = XRayClassifier(backbone='densenet121', pretrained=False)
    model = model.to(device)
    
    try:
        model, checkpoint = load_checkpoint(model, model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        if 'epoch' in checkpoint:
            logger.info(f"Model was trained for {checkpoint['epoch']} epochs")
        if 'val_acc' in checkpoint:
            logger.info(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference
    logger.info("Running inference on test set...")
    all_preds = []
    all_labels = []
    all_probs = []
    all_image_paths = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 10 == 0:
                logger.info(f"Processing batch {i+1}/{len(test_loader)}")
            
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            # Store results
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob for pneumonia class
            
            # Store image paths if available
            if 'image_path' in batch:
                all_image_paths.extend(batch['image_path'])
    
    # Convert to numpy arrays for easier manipulation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    logger.info("Calculating performance metrics...")
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"AUC-ROC:   {auc:.4f}")
    logger.info("=" * 60)
    logger.info("Confusion Matrix:")
    logger.info(f"                 Predicted")
    logger.info(f"              Normal  Pneumonia")
    logger.info(f"Actual Normal    {cm[0,0]:4d}     {cm[0,1]:4d}")  
    logger.info(f"    Pneumonia    {cm[1,0]:4d}     {cm[1,1]:4d}")
    logger.info("=" * 60)
    
    # Print detailed classification report
    class_report = classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia'])
    logger.info("Classification Report:")
    logger.info(f"\n{class_report}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
        'Value': [acc, precision, recall, f1, auc]
    })
    results_path = os.path.join(results_dir, 'test_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    # Plot and save visualizations
    logger.info("Creating visualizations...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # 2. Performance Metrics Bar Chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    values = [acc, precision, recall, f1, auc]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'gold'])
    plt.ylim(0, 1.0)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics - Test Set')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels above bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    metrics_path = os.path.join(results_dir, 'performance_metrics.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Performance metrics chart saved to {metrics_path}")
    
    # 3. ROC Curve
    roc_path = os.path.join(results_dir, 'roc_curve.png')
    plot_roc_curve(all_labels.tolist(), all_probs.tolist(), save_path=roc_path)
    
    # 4. Precision-Recall Curve  
    pr_path = os.path.join(results_dir, 'precision_recall_curve.png')
    plot_precision_recall_curve(all_labels.tolist(), all_probs.tolist(), save_path=pr_path)
    
    # 5. Sample predictions (if image paths are available)
    if all_image_paths:
        sample_pred_path = os.path.join(results_dir, 'sample_predictions.png')
        plot_sample_predictions(
            all_image_paths, 
            all_preds.tolist(), 
            all_labels.tolist(), 
            all_probs.tolist(),
            num_samples=9,
            save_path=sample_pred_path
        )
    
    logger.info(f"All results and visualizations saved to {results_dir}")
    
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
    # Configuration
    MODEL_PATH = "/home/psaha03/scratch/classifier/models/densenet_classifier_best.pt"
    TEST_DATA_DIR = "/home/psaha03/scratch/chest_xray_data"
    RESULTS_DIR = "/home/psaha03/scratch/classifier/test_results"
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)
    
    # Check if test data directory exists
    if not os.path.exists(TEST_DATA_DIR):
        logger.error(f"Test data directory not found: {TEST_DATA_DIR}")
        sys.exit(1)
    
    # Run the test
    logger.info("Starting model testing...")
    results = test_model_on_dataset(MODEL_PATH, TEST_DATA_DIR, RESULTS_DIR)
    logger.info("Testing completed successfully!")