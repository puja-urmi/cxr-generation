"""
Inference script for X-ray binary classifier
"""

import os
import logging
import argparse
from typing import List, Union, Dict
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import config
from model import get_model, load_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def predict_batch(image_paths: List[str], checkpoint_path: str = None) -> List[Dict]:
    """
    Make predictions for a batch of images
    
    Args:
        image_paths: List of paths to image files
        checkpoint_path: Path to model checkpoint
        
    Returns:
        List of dictionaries with prediction results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = get_model(device)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINT_PATH
    
    try:
        model, _ = load_checkpoint(model, checkpoint_path)
        logger.info(f"Loaded model from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return [{'error': f"Failed to load model: {str(e)}"}]
    
    # Make predictions
    results = []
    for image_path in image_paths:
        result = predict_single_image(image_path, model, device)
        results.append(result)
        
        if 'error' not in result:
            logger.info(f"Image: {os.path.basename(image_path)}, "
                       f"Prediction: {result['predicted_class']}, "
                       f"Confidence: {result['confidence']:.4f}")
    
    return results


def visualize_prediction(image_path: str, result: Dict, save_path: str = None):
    """
    Visualize prediction results
    
    Args:
        image_path: Path to image file
        result: Prediction result dictionary
        save_path: Path to save visualization (if None, display instead)
    """
    try:
        image = Image.open(image_path).convert('RGB')
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('X-ray Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        classes = list(result['class_probabilities'].keys())
        probs = list(result['class_probabilities'].values())
        
        colors = ['green' if prob == max(probs) else 'blue' for prob in probs]
        
        plt.barh(classes, probs, color=colors)
        plt.title('Prediction Probabilities')
        plt.xlabel('Probability')
        plt.xlim(0, 1)
        
        # Add text with prediction
        pred_class = result['predicted_class']
        confidence = result['confidence']
        plt.figtext(0.5, 0.01, f"Prediction: {pred_class} (Confidence: {confidence:.2f})", 
                   ha='center', fontsize=12, bbox={'facecolor':'lightgray', 'alpha':0.5})
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing prediction: {e}")


def main(args):
    """Main inference function"""
    # Check if input is a directory or file
    image_paths = []
    
    if os.path.isdir(args.input):
        # Get all image files in directory
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(args.input, filename))
    else:
        # Single file
        image_paths = [args.input]
    
    if not image_paths:
        logger.error(f"No images found in {args.input}")
        return
    
    # Make predictions
    results = predict_batch(image_paths, args.checkpoint)
    
    # Save or display visualizations
    if args.visualize:
        for result, image_path in zip(results, image_paths):
            if 'error' not in result:
                if args.output:
                    # Create filename for visualization
                    basename = os.path.basename(image_path)
                    vis_path = os.path.join(args.output, f"vis_{os.path.splitext(basename)[0]}.png")
                    visualize_prediction(image_path, result, vis_path)
                else:
                    visualize_prediction(image_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained X-ray classifier")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions")
    parser.add_argument("--output", type=str, default=None, help="Directory to save visualizations")
    
    args = parser.parse_args()
    main(args)