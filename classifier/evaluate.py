"""
Comprehensive testing and evaluation script for X-ray classifier
Supports both full dataset evaluation and individual image prediction
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Add the classifier directory to the path
sys.path.append('/home/psaha03/scratch/cxr-generation/classifier')

import config
from data_loaders import ModelSpecificDataLoaders, ChestXRayDataset, MedicalImageTransforms
from model import XRayClassifier, load_checkpoint, get_model

# Set up comprehensive logging
def setup_logging(log_file: str = None):
    """Setup logging to both console and file"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"test_results_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

logger, _ = setup_logging()


def get_transform():
    """Get transform for inference"""
    return transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_single_image(image_path: str, model: torch.nn.Module, device: torch.device) -> Dict:
    """
    Make prediction for a single image
    
    Args:
        image_path: Path to image file
        model: Trained model
        device: Device to run inference on
        
    Returns:
        Dictionary with prediction results
    """
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
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    result = {
        'image_path': image_path,
        'predicted_class': config.CLASSES[pred_class],
        'predicted_label': pred_class,
        'confidence': confidence,
        'class_probabilities': {cls: float(prob) for cls, prob in zip(config.CLASSES, probs[0].cpu().numpy())}
    }
    
    return result


def predict_batch_images(image_paths: List[str], model: torch.nn.Module, device: torch.device) -> List[Dict]:
    """
    Make predictions for a batch of images
    
    Args:
        image_paths: List of paths to image files
        model: Trained model
        device: Device to run inference on
        
    Returns:
        List of dictionaries with prediction results
    """
    results = []
    for image_path in image_paths:
        result = predict_single_image(image_path, model, device)
        results.append(result)
        
        if 'error' not in result:
            logger.info(f"Image: {os.path.basename(image_path)}, "
                       f"Prediction: {result['predicted_class']}, "
                       f"Confidence: {result['confidence']:.4f}")
    
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


def plot_sample_predictions(image_paths: List[str], predictions: List, labels: List, 
                          probabilities: List, num_samples: int = 9, save_path: str = None):
    """Plot sample predictions"""
    if len(image_paths) == 0:
        return
        
    # Select random samples
    indices = np.random.choice(len(image_paths), min(num_samples, len(image_paths)), replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= 9:
            break
            
        try:
            image = Image.open(image_paths[idx]).convert('RGB')
            axes[i].imshow(image, cmap='gray')
            
            pred_class = 'Pneumonia' if predictions[idx] == 1 else 'Normal'
            true_class = 'Pneumonia' if labels[idx] == 1 else 'Normal'
            prob = probabilities[idx]
            
            color = 'green' if predictions[idx] == labels[idx] else 'red'
            title = f'True: {true_class}, Pred: {pred_class}\nConf: {prob:.3f}'
            axes[i].set_title(title, color=color, fontsize=10)
            axes[i].axis('off')
            
        except Exception as e:
            logger.warning(f"Could not load image {image_paths[idx]}: {e}")
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(indices), 9):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sample predictions saved to {save_path}")
    
    plt.close()


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
    
    # Create test data loader - using robust ImageFolder approach
    logger.info("Setting up test data loader...")
    
    # Check if test directory exists and has the right structure
    if not os.path.exists(test_data_dir):
        logger.error(f"Test data directory not found: {test_data_dir}")
        return None
    
    # Log directory contents for debugging
    logger.info(f"Checking test directory: {test_data_dir}")
    for item in os.listdir(test_data_dir):
        item_path = os.path.join(test_data_dir, item)
        if os.path.isdir(item_path):
            count = len([f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            logger.info(f"  Found {item} directory with {count} images")
    
    # Use simple and robust ImageFolder dataset
    from torchvision import datasets
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        test_dataset = datasets.ImageFolder(
            root=test_data_dir,
            transform=test_transform
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Test dataset: {len(test_dataset)} images, {len(test_loader)} batches")
        logger.info(f"Class mapping: {test_dataset.class_to_idx}")
        
        # Check if we actually have data
        if len(test_dataset) == 0:
            logger.error("No images found in test directory!")
            logger.error(f"Expected structure: {test_data_dir}/NORMAL/ and {test_data_dir}/PNEUMONIA/")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        return None
    
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
        for i, (images, labels) in enumerate(test_loader):
            if i % 10 == 0:
                logger.info(f"Processing batch {i+1}/{len(test_loader)}")
            
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            # Store results
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob for pneumonia class (class 1)
    
    # Convert to numpy arrays for easier manipulation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Check if we have any data before calculating metrics
    if len(all_labels) == 0 or len(all_probs) == 0:
        logger.error("No test data processed! Check your data directory structure.")
        logger.error(f"Expected: {test_data_dir}/NORMAL/ and {test_data_dir}/PNEUMONIA/")
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


def load_model_for_inference(model_path: str, device: torch.device):
    """Load model for inference"""
    try:
        # Try loading with XRayClassifier first
        model = XRayClassifier(backbone='densenet121', pretrained=False)
        model = model.to(device)
        model, checkpoint = load_checkpoint(model, model_path)
        logger.info(f"Successfully loaded XRayClassifier model from {model_path}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load with XRayClassifier: {e}")
        try:
            # Try with get_model function as fallback
            model = get_model(device)
            model, checkpoint = load_checkpoint(model, model_path)
            logger.info(f"Successfully loaded model using get_model from {model_path}")
            return model
        except Exception as e2:
            logger.error(f"Failed to load model with both methods: {e2}")
            raise e2


def predict_images(image_paths: List[str], model_path: str, results_dir: str = None):
    """
    Run inference on individual images
    
    Args:
        image_paths: List of image file paths
        model_path: Path to trained model
        results_dir: Directory to save results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model_for_inference(model_path, device)
    
    # Make predictions
    logger.info(f"Making predictions on {len(image_paths)} images...")
    results = predict_batch_images(image_paths, model, device)
    
    # Save results if directory provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        
        # Save predictions to CSV
        prediction_data = []
        for result in results:
            if 'error' not in result:
                prediction_data.append({
                    'image_path': result['image_path'],
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'normal_prob': result['class_probabilities'].get('normal', 0),
                    'abnormal_prob': result['class_probabilities'].get('abnormal', 0)
                })
        
        if prediction_data:
            pred_df = pd.DataFrame(prediction_data)
            pred_path = os.path.join(results_dir, 'predictions.csv')
            pred_df.to_csv(pred_path, index=False)
            logger.info(f"Predictions saved to {pred_path}")
    
    return results


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="X-ray Classifier Testing and Prediction Tool")
    parser.add_argument("--mode", type=str, choices=['test', 'predict'], required=True,
                        help="Mode: 'test' for full dataset evaluation, 'predict' for individual images")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.pt file)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to test data directory (for test mode) or image file/directory (for predict mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results (default: ./test_results)")
    parser.add_argument("--log", type=str, default=None,
                        help="Log file path (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Setup logging with custom log file if provided
    if args.log:
        global logger
        logger, log_file = setup_logging(args.log)
    
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
    
    if args.mode == 'test':
        # Full dataset evaluation
        if not os.path.exists(args.data):
            logger.error(f"Test data directory not found: {args.data}")
            sys.exit(1)
        
        results = test_model_on_dataset(args.model, args.data, args.output)
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
        
        results = predict_images(image_paths, args.model, args.output)
        logger.info("Image prediction completed successfully!")


if __name__ == "__main__":
    main()