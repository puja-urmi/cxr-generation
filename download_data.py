"""
Script to download Chest X-ray Pneumonia dataset from Kaggle using kagglehub.

Requirements:
- kagglehub (install with: pip install kagglehub)
- You must have Kaggle API credentials set up (kaggle.json in ~/.kaggle/)
"""

import kagglehub

# Download latest version of the Chest X-ray Pneumonia dataset
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)
print("This dataset contains chest X-ray images categorized into 'normal' and 'pneumonia' classes.")
