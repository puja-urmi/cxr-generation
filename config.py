"""
Configuration settings for X-ray classifier project
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('XRAY_DATA_DIR', os.path.join(os.path.dirname(BASE_DIR), 'data'))

# Image settings
IMG_SIZE = (224, 224)  # Standard size for many pre-trained models
CHANNELS = 3  # 3 for RGB, 1 for grayscale

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 4  

# Model settings
MODEL_NAME = "densenet_classifier"
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CLASSES = ['normal', 'abnormal']
NUM_CLASSES = len(CLASSES)

os.makedirs(MODEL_DIR, exist_ok=True)

# Paths for saving/loading models
CHECKPOINT_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.pt")
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# TensorBoard settings
TENSORBOARD_DIR = os.path.join(BASE_DIR, 'runs')
os.makedirs(TENSORBOARD_DIR, exist_ok=True)