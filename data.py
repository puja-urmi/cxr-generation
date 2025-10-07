"""
Data loading and preprocessing for X-ray images
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from typing import Tuple, Dict, List, Optional
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XRayDataset(Dataset):
    """
    Dataset for loading X-ray images with binary labels (normal/abnormal)
    """
    def __init__(self, 
                 csv_file: str, 
                 img_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 label_col: str = 'label',
                 img_col: str = 'image_path'):
        """
        Args:
            csv_file: Path to CSV with image paths and labels
            img_dir: Directory with images
            transform: Optional transforms to apply to images
            label_col: Column name for label in CSV
            img_col: Column name for image path in CSV
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col
        self.img_col = img_col
        
        # Verify data
        self._verify_data()
        
    def _verify_data(self):
        """Verify that images exist and labels are valid"""
        # Check if required columns exist
        for col in [self.img_col, self.label_col]:
            if col not in self.df.columns:
                raise ValueError(f"Column {col} not found in dataset CSV")
        
        # Check label values (assuming binary 0/1 labels)
        labels = self.df[self.label_col].unique()
        if not all(label in [0, 1] for label in labels):
            logger.warning("Labels are not all binary (0/1). Converting to int.")
            self.df[self.label_col] = self.df[self.label_col].astype(int)
            
        # Verify sample images
        sample_idx = np.random.choice(len(self.df), min(5, len(self.df)), replace=False)
        for idx in sample_idx:
            img_path = os.path.join(self.img_dir, self.df.iloc[idx][self.img_col])
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                
        logger.info(f"Dataset initialized with {len(self.df)} samples")
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = os.path.join(self.img_dir, self.df.iloc[idx][self.img_col])
        
        try:
            # Open as RGB to ensure 3 channels
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            label = self.df.iloc[idx][self.label_col]
            
            return {'image': image, 'label': torch.tensor(label, dtype=torch.long)}
        
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder if image can't be loaded
            placeholder = torch.zeros((3, *config.IMG_SIZE))
            return {'image': placeholder, 'label': torch.tensor(0)}
        

def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Get image transformations for training or validation/testing
    
    Args:
        train: Whether to include training augmentations
        
    Returns:
        A composition of transforms
    """
    if train:
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

def get_dataloaders(img_dir: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training, validation, and test data loaders
    
    Args:
        img_dir: Directory containing images
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = XRayDataset(
        csv_file=config.TRAIN_CSV,
        img_dir=img_dir,
        transform=get_transforms(train=True)
    )
    
    val_dataset = XRayDataset(
        csv_file=config.VAL_CSV,
        img_dir=img_dir,
        transform=get_transforms(train=False)
    )
    
    test_dataset = XRayDataset(
        csv_file=config.TEST_CSV,
        img_dir=img_dir,
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def prepare_data_csv(
    img_dir: str, 
    train_split: float = 0.7, 
    val_split: float = 0.15,
    normal_label: int = 0,
    abnormal_label: int = 1,
    patterns: Dict[str, int] = None
) -> None:
    """
    Create CSV files for training, validation, and testing
    
    Args:
        img_dir: Directory with images
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        normal_label: Label value for normal X-rays
        abnormal_label: Label value for abnormal X-rays
        patterns: Dictionary mapping string patterns to labels
                  e.g., {'normal': 0, 'pneumonia': 1}
    """
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    if patterns is None:
        patterns = {'normal': normal_label, 'abnormal': abnormal_label}
    
    image_paths = []
    labels = []
    
    # Walk through directory
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(os.path.relpath(root, img_dir), file)
                
                # Determine label based on file path patterns
                label = None
                for pattern, label_value in patterns.items():
                    if pattern.lower() in img_path.lower():
                        label = label_value
                        break
                
                # Skip if no label assigned
                if label is None:
                    logger.warning(f"Could not determine label for {img_path}")
                    continue
                
                image_paths.append(img_path)
                labels.append(label)
    
    if not image_paths:
        logger.error(f"No images found in {img_dir} matching patterns")
        return
    
    # Create dataframe and shuffle
    data = pd.DataFrame({'image_path': image_paths, 'label': labels})
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate splits
    n = len(data)
    train_end = int(train_split * n)
    val_end = train_end + int(val_split * n)
    
    # Split data
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    # Save CSVs
    train_data.to_csv(config.TRAIN_CSV, index=False)
    val_data.to_csv(config.VAL_CSV, index=False)
    test_data.to_csv(config.TEST_CSV, index=False)
    
    logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    logger.info(f"Class distribution in Train: {train_data['label'].value_counts().to_dict()}")
    

if __name__ == "__main__":
    # Example usage
    print("To prepare data CSV files, call:")
    print("python -c \"import data; data.prepare_data_csv('/path/to/images')\"")