"""
Script to download NIH Chest X-rays data using kagglehub.

Requirements:
- kagglehub (install with: pip install kagglehub)
- You must have Kaggle API credentials set up (kaggle.json in ~/.kaggle/)
"""

import kagglehub

# Download latest version of the dataset
path = kagglehub.dataset_download("nih-chest-xrays/data")

print("Path to dataset files:", path)
