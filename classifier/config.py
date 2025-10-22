"""
Configuration settings for X-ray classifier project
"""

import os
from pathlib import Path

# ===== BASE PATHS =====
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ===== MODEL CONFIGURATION =====
MODEL_ARCHITECTURE = 'densenet121'
MODEL_NAME = "densenet_classifier"
PRETRAINED = True

# Model architecture parameters
HIDDEN_SIZE_1 = 512
HIDDEN_SIZE_2 = 128
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2

# ===== IMAGE SETTINGS =====
IMG_SIZE = (224, 224)
CHANNELS = 3

# ===== DATA LOADER SETTINGS =====
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True

# ===== TRAINING SETTINGS =====
LEARNING_RATE = 0.0001
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Optimizer settings
OPTIMIZER = 'Adam'
WEIGHT_DECAY = 0.0001

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = 'ReduceLROnPlateau'
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

# ===== EVALUATION SETTINGS =====
CLASSIFICATION_THRESHOLD = 0.5
USE_CUSTOM_THRESHOLD = True

# ===== CLASS DEFINITIONS =====
CLASSES = ['normal', 'abnormal']
NUM_CLASSES = len(CLASSES)
CLASS_NAMES = ['Normal', 'Abnormal']

# ===== FILE PATHS =====
MODEL_DIR = os.path.join(BASE_DIR, 'models')
TENSORBOARD_DIR = os.path.join(BASE_DIR, 'runs')

# Data paths (default data directory structure)
DATA_ROOT = '/home/psaha03/scratch/chest_xray'
TRAIN_PATH = os.path.join(DATA_ROOT, 'train')
VAL_PATH = os.path.join(DATA_ROOT, 'val')
TEST_PATH = os.path.join(DATA_ROOT, 'test')

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# Model checkpoint path
CHECKPOINT_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.pt")

# ===== LOGGING SETTINGS =====
LOG_LEVEL = 'INFO'
USE_TENSORBOARD = True

# ===== SYSTEM SETTINGS =====
DEVICE = 'auto'
RANDOM_SEED = 42

# ===== INFERENCE SETTINGS =====
INFERENCE_MEAN = [0.485, 0.456, 0.406]
INFERENCE_STD = [0.229, 0.224, 0.225]