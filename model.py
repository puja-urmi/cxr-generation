"""
Model definitions for X-ray binary classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Tuple
import logging
import config

logger = logging.getLogger(__name__)

class XRayClassifier(nn.Module):
    """
    CNN model for binary classification of X-ray images
    Uses a pre-trained backbone with a custom classifier head
    """
    def __init__(self, 
                 backbone: str = 'densenet121',
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        """
        Args:
            backbone: Name of backbone model (resnet18, densenet121, etc.)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super(XRayClassifier, self).__init__()
        
        # Initialize backbone
        if backbone == 'densenet121':
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()  # Remove the classifier
            
        elif backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone model: {backbone}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, config.NUM_CLASSES)
        )
        
        logger.info(f"Initialized {backbone} model with {'pretrained' if pretrained else 'random'} weights")
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def predict(self, x):
        """
        Get class predictions and probabilities
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


def get_model(device: torch.device) -> XRayClassifier:
    """
    Create and initialize the model
    
    Args:
        device: Device to put the model on
        
    Returns:
        Initialized model on device
    """
    model = XRayClassifier(backbone='densenet121', pretrained=True)
    model = model.to(device)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model from checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (model with loaded weights, checkpoint dictionary)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise