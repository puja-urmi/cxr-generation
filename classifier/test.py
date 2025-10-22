"""
Testing and evaluation script for X-ray classifier
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms, datasets
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Add the classifier directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from model import XRayClassifier, load_checkpoint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_transform():
    """Get transform for inference"""
    return transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.INFERENCE_MEAN, std=config.INFERENCE_STD)
    ])


def predict_single_image(image_path: str, model: torch.nn.Module, device: torch.device, threshold: float = 0.5) -> Dict:
    """Make prediction for a single image"""
    if not os.path.exists(image_path):
        return {'error': f'Image file not found: {image_path}'}
    
    if not 0 <= threshold <= 1:
        return {'error': f'Threshold must be between 0 and 1, got {threshold}'}
    
    transform = get_transform()
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return {'error': str(e)}
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pneumonia_prob = probs[0][1].item()
        pred_class = 1 if pneumonia_prob >= threshold else 0
        confidence = pneumonia_prob if pred_class == 1 else (1 - pneumonia_prob)
    
    return {
        'image_path': image_path,
        'predicted_class': config.CLASSES[pred_class],
        'predicted_label': pred_class,
        'confidence': confidence,
        'pneumonia_probability': pneumonia_prob,
        'threshold_used': threshold
    }


def predict_batch_images(image_paths: List[str], model: torch.nn.Module, device: torch.device, threshold: float = 0.5) -> List[Dict]:
    """Make predictions for a batch of images"""
    results = []
    for image_path in image_paths:
        result = predict_single_image(image_path, model, device, threshold)
        results.append(result)
        
        if 'error' not in result:
            logger.info(f"Image: {os.path.basename(image_path)}, "
                       f"Prediction: {result['predicted_class']}, "
                       f"Pneumonia Prob: {result['pneumonia_probability']:.4f}")
    
    return results


def plot_roc_curve(y_true: List, y_probs: List, save_path: str = None):
    """Create and save ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    plt.close()


def plot_precision_recall_curve(y_true: List, y_probs: List, save_path: str = None):
    """Create and save Precision-Recall curve"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {save_path}")
    
    plt.close()



def test_model_on_dataset(model_path, test_data_dir, results_dir=None, threshold=0.5):
    """Test the trained model on the test dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set up results directory
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f'./test_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create test data loader
    logger.info("Setting up test data loader...")
    
    if not os.path.exists(test_data_dir):
        logger.error(f"Test data directory not found: {test_data_dir}")
        return None
    
    test_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.INFERENCE_MEAN, std=config.INFERENCE_STD)
    ])
    
    try:
        test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
        )
        
        logger.info(f"Test dataset: {len(test_dataset)} images, {len(test_loader)} batches")
        
        if len(test_dataset) == 0:
            logger.error("No images found in test directory!")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        return None
    
    # Load model
    logger.info("Loading trained model...")
    model = XRayClassifier(backbone=config.MODEL_ARCHITECTURE, pretrained=False)
    model = model.to(device)
    
    try:
        model, checkpoint = load_checkpoint(model, model_path)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference
    logger.info(f"Running inference with threshold = {threshold:.2f}")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i % 10 == 0:
                logger.info(f"Processing batch {i+1}/{len(test_loader)}")
            
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            # Apply threshold for pneumonia classification
            pneumonia_probs = probs[:, 1].cpu().numpy()
            preds = (pneumonia_probs >= threshold).astype(int)
            
            # Store results
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(pneumonia_probs)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    if len(all_labels) == 0:
        logger.error("No test data processed!")
        return None
    
    # Calculate metrics
    logger.info("Calculating performance metrics...")
    logger.info(f"Total samples processed: {len(all_labels)}")
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate specificity (True Negatives / (True Negatives + False Positives))
    # Specificity measures the model's ability to correctly identify normal (healthy) cases
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results with threshold information
    logger.info("=" * 70)
    logger.info("TEST RESULTS WITH THRESHOLD OPTIMIZATION")
    logger.info("=" * 70)
    logger.info(f"THRESHOLD USED: {threshold:.2f} (Pneumonia if P(pneumonia) >= {threshold:.2f})")
    logger.info("-" * 70)
    logger.info(f"Accuracy:   {acc:.4f}")
    logger.info(f"Precision:  {precision:.4f}")
    logger.info(f"Recall:     {recall:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    logger.info(f"F1 Score:   {f1:.4f}")
    logger.info(f"AUC-ROC:    {auc:.4f}")
    logger.info("=" * 70)
    logger.info("Confusion Matrix:")
    logger.info(f"                 Predicted")
    logger.info(f"              Normal  Pneumonia")
    logger.info(f"Actual Normal    {cm[0,0]:4d}     {cm[0,1]:4d}")  
    logger.info(f"    Pneumonia    {cm[1,0]:4d}     {cm[1,1]:4d}")
    logger.info("-" * 70)
    
    # Calculate class-specific accuracies
    normal_acc = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    pneumonia_acc = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    
    logger.info(f"NORMAL Classification Accuracy: {normal_acc:.4f} ({cm[0,0]}/{cm[0,0] + cm[0,1]})")
    logger.info(f"PNEUMONIA Classification Accuracy: {pneumonia_acc:.4f} ({cm[1,1]}/{cm[1,0] + cm[1,1]})")
    logger.info("=" * 70)
    
    # Print detailed classification report
    class_report = classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia'])
    logger.info("Classification Report:")
    logger.info(f"\n{class_report}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'AUC'],
        'Value': [acc, precision, recall, specificity, f1, auc]
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
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'AUC']
    values = [acc, precision, recall, specificity, f1, auc]
    
    # Create figure with larger size and better spacing
    plt.figure(figsize=(14, 8))
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'coral', 'pink', 'gold'], 
                   edgecolor='black', linewidth=0.8, alpha=0.8)
    
    # Set y-axis limits with more space at the top for labels
    plt.ylim(0, 1.1)
    
    # Improve labels and title with larger fonts
    plt.xlabel('Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.title('Model Performance Metrics - Test Set', fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels above bars with better positioning and larger font
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Adjust layout with more padding
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95)
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
    
    logger.info(f"All results and visualizations saved to {results_dir}")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def load_model_for_inference(model_path: str, device: torch.device):
    """Load model for inference"""
    model = XRayClassifier(backbone=config.MODEL_ARCHITECTURE, pretrained=False)
    model = model.to(device)
    model, checkpoint = load_checkpoint(model, model_path)
    logger.info(f"Successfully loaded model from {model_path}")
    return model


def predict_images(image_paths: List[str], model_path: str, results_dir: str = None, threshold: float = 0.5):
    """Run inference on individual images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model_for_inference(model_path, device)
    
    # Make predictions
    logger.info(f"Making predictions on {len(image_paths)} images with threshold {threshold:.2f}...")
    results = predict_batch_images(image_paths, model, device, threshold)
    
    # Save results if directory provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        
        prediction_data = []
        for result in results:
            if 'error' not in result:
                prediction_data.append({
                    'image_path': result['image_path'],
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'pneumonia_probability': result['pneumonia_probability'],
                    'threshold_used': result['threshold_used']
                })
        
        if prediction_data:
            pred_df = pd.DataFrame(prediction_data)
            pred_path = os.path.join(results_dir, 'predictions.csv')
            pred_df.to_csv(pred_path, index=False)
            logger.info(f"Predictions saved to {pred_path}")
    
    return results


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="X-ray Classifier Testing Tool")
    parser.add_argument("--mode", type=str, choices=['test', 'predict'], required=True,
                        help="Mode: 'test' for dataset evaluation, 'predict' for individual images")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.pt file)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to test data directory or image file/directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for pneumonia classification")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Reduce logging output (only show key results)")
    
    args = parser.parse_args()
    
    # Set logging level based on quiet flag
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
        # Only show essential results
        logging.getLogger(__name__).setLevel(logging.INFO)
    
    # Set default output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"./test_results_{timestamp}"
    
    # Check if model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    logger.info(f"Starting {args.mode} mode...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Threshold: {args.threshold:.2f}")
    
    if args.mode == 'test':
        # Full dataset evaluation
        if not os.path.exists(args.data):
            logger.error(f"Test data directory not found: {args.data}")
            sys.exit(1)
        
        results = test_model_on_dataset(args.model, args.data, args.output, args.threshold)
        logger.info("Dataset testing completed successfully!")
        
    elif args.mode == 'predict':
        # Individual image prediction
        image_paths = []
        
        if os.path.isdir(args.data):
            # Directory of images
            for filename in os.listdir(args.data):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(args.data, filename))
        elif os.path.isfile(args.data):
            # Single image
            image_paths = [args.data]
        else:
            logger.error(f"Data path not found: {args.data}")
            sys.exit(1)
        
        if not image_paths:
            logger.error(f"No valid images found in {args.data}")
            sys.exit(1)
        
        results = predict_images(image_paths, args.model, args.output, args.threshold)
        logger.info("Image prediction completed successfully!")


if __name__ == "__main__":
    main()