"""
Centralized Configuration settings for X-ray classifier project
Control all parameters from this single file
"""

import os
from pathlib import Path

# ===== BASE PATHS =====
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('XRAY_DATA_DIR', os.path.join(os.path.dirname(BASE_DIR), 'data'))

# ===== MODEL CONFIGURATION =====
# Available models: 'densenet121', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b3', 'efficientnet_b4'
MODEL_ARCHITECTURE = 'densenet121'
MODEL_NAME = "densenet_classifier"
PRETRAINED = True  # Use pretrained weights

# Model-specific configurations
MODEL_CONFIGS = {
    'densenet121': {
        'image_size': (224, 224),
        'batch_size': 32,
        'augmentation': 'moderate'
    }
}

# Get current model config
CURRENT_MODEL_CONFIG = MODEL_CONFIGS.get(MODEL_ARCHITECTURE, MODEL_CONFIGS['densenet121'])

# ===== IMAGE SETTINGS =====
IMG_SIZE = CURRENT_MODEL_CONFIG['image_size']
CHANNELS = 3  # 3 for RGB, 1 for grayscale

# ===== DATA LOADER SETTINGS =====
BATCH_SIZE = CURRENT_MODEL_CONFIG['batch_size']  # Can be overridden
NUM_WORKERS = 4
PIN_MEMORY = True
AUGMENTATION_STRENGTH = CURRENT_MODEL_CONFIG['augmentation']  # 'light', 'moderate', 'strong'

# Data splits (if you want to modify default behavior)
TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
TEST_SHUFFLE = False
DROP_LAST_TRAIN = True
DROP_LAST_VAL = False
DROP_LAST_TEST = False

# ===== TRAINING SETTINGS =====
LEARNING_RATE = 0.0001
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Optimizer settings
OPTIMIZER = 'Adam'  # 'Adam', 'SGD', 'AdamW'
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9  # Only for SGD

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = 'ReduceLROnPlateau'  # 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR'
SCHEDULER_FACTOR = 0.5  # For ReduceLROnPlateau
SCHEDULER_PATIENCE = 3  # For ReduceLROnPlateau
SCHEDULER_STEP_SIZE = 10  # For StepLR
SCHEDULER_GAMMA = 0.1  # For StepLR

# Class weights for imbalanced data
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_METHOD = 'balanced'  # 'balanced', 'inverse_freq', 'custom'
CUSTOM_CLASS_WEIGHTS = [1.0, 1.0]  # Only used if method is 'custom'

# ===== EVALUATION SETTINGS =====
# Threshold for binary classification
CLASSIFICATION_THRESHOLD = 0.6  # Pneumonia if P(pneumonia) >= threshold
USE_CUSTOM_THRESHOLD = True  # If False, uses argmax (0.5 threshold)

# Metrics to compute
COMPUTE_AUC = True
COMPUTE_PRECISION_RECALL = True
SAVE_CONFUSION_MATRIX = True
SAVE_ROC_CURVE = True
SAVE_PR_CURVE = True

# Visualization settings
PLOT_SAMPLE_PREDICTIONS = True
NUM_SAMPLE_PREDICTIONS = 9
SAVE_PREDICTION_EXAMPLES = True

# ===== CLASS DEFINITIONS =====
CLASSES = ['normal', 'abnormal']  # Corresponds to NORMAL=0, PNEUMONIA=1
NUM_CLASSES = len(CLASSES)
CLASS_NAMES = ['Normal', 'Pneumonia']  # For display purposes

# ===== FILE PATHS =====
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TENSORBOARD_DIR = os.path.join(BASE_DIR, 'runs')

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# Model checkpoint path
CHECKPOINT_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.pt")

# ===== LOGGING SETTINGS =====
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
SAVE_LOGS = True

# Training logging frequency
LOG_TRAIN_EVERY = 20  # Log every N batches during training
LOG_VAL_EVERY = 1    # Log validation every N epochs
SAVE_MODEL_EVERY = 5  # Save model checkpoint every N epochs (in addition to best)

# TensorBoard logging
USE_TENSORBOARD = True
LOG_IMAGES_EVERY = 5  # Log example images every N epochs
LOG_HISTOGRAMS = False  # Log model parameter histograms

# ===== SYSTEM SETTINGS =====
DEVICE = 'auto'  # 'auto', 'cuda', 'cpu'
MIXED_PRECISION = False  # Use automatic mixed precision training
DETERMINISTIC = False  # Set seeds for reproducibility
RANDOM_SEED = 42

# ===== INFERENCE SETTINGS =====
# Batch size for inference (can be larger than training batch size)
INFERENCE_BATCH_SIZE = 64
INFERENCE_NUM_WORKERS = 4

# Image preprocessing for inference
INFERENCE_NORMALIZE = True
INFERENCE_MEAN = [0.485, 0.456, 0.406]  # ImageNet means
INFERENCE_STD = [0.229, 0.224, 0.225]   # ImageNet stds