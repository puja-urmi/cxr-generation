"""
Data Loaders for Chest X-Ray Dataset
=====================================

This module provides custom PyTorch DataLoader classes and preprocessing
transformations for the chest X-ray pneumonia detection dataset.

Key Features:
- On-the-fly preprocessing and augmentation
- Model-specific input size handling
- Medical imaging appropriate transformations
- Balanced sampling options
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import random
from typing import Tuple, Optional, Dict, List

import config


class ChestXRayDataset(Dataset):
    """Custom Dataset for Chest X-Ray images."""
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the chest_xray directory
            split: One of 'train', 'val', 'test'
            transform: Optional transform to be applied on images
            image_size: Target image size (height, width)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Class to index mapping
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        self.idx_to_class = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} images for {split} split")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image paths and their corresponding labels."""
        samples = []
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = self.data_dir / self.split / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpeg'):
                    label = self.class_to_idx[class_name]
                    samples.append((img_path, label))
        
        return samples
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {'NORMAL': 0, 'PNEUMONIA': 0}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1
        return distribution
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path)
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: resize and convert to tensor
            image = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])(image)
        
        return image, label


class MedicalImageTransforms:
    """Medical imaging appropriate transformations using config settings."""
    
    @staticmethod
    def get_train_transforms(
        image_size: Tuple[int, int] = None,
        augmentation_strength: str = None
    ) -> transforms.Compose:
        """
        Get training transformations with augmentation using config settings.
        
        Args:
            image_size: Target image size (uses config.IMG_SIZE if None)
            augmentation_strength: Augmentation level (uses config.AUGMENTATION_STRENGTH if None)
            
        Returns:
            Composed transforms for training
        """
        # Use config values if not provided
        if image_size is None:
            image_size = config.IMG_SIZE
        if augmentation_strength is None:
            augmentation_strength = config.AUGMENTATION_STRENGTH
        
        # Base transforms
        base_transforms = [
            transforms.Resize(image_size),
        ]
        
        # Augmentation based on strength
        if augmentation_strength == 'light':
            augment_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ]
        elif augmentation_strength == 'moderate':
            augment_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]
        elif augmentation_strength == 'strong':
            augment_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            ]
        else:
            augment_transforms = []
        
        # Final transforms
        final_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.INFERENCE_MEAN,  # Use config means
                std=config.INFERENCE_STD     # Use config stds
            )
        ]
        
        return transforms.Compose(base_transforms + augment_transforms + final_transforms)
    
    @staticmethod
    def get_val_test_transforms(
        image_size: Tuple[int, int] = None
    ) -> transforms.Compose:
        """
        Get validation/test transformations (no augmentation) using config settings.
        
        Args:
            image_size: Target image size (uses config.IMG_SIZE if None)
            
        Returns:
            Composed transforms for validation/testing
        """
        # Use config value if not provided
        if image_size is None:
            image_size = config.IMG_SIZE
            
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.INFERENCE_MEAN,  # Use config means
                std=config.INFERENCE_STD     # Use config stds
            )
        ])


class ModelSpecificDataLoaders:
    """Factory class for creating model-specific data loaders using config settings."""
    
    @classmethod
    def create_dataloaders(
        cls,
        data_dir: str,
        model_name: str = None,
        batch_size: Optional[int] = None,
        num_workers: int = None,
        pin_memory: bool = None
    ) -> Dict[str, DataLoader]:
        """
        Create data loaders using config settings.
        
        Args:
            data_dir: Path to the chest_xray directory
            model_name: Name of the model (uses config.MODEL_ARCHITECTURE if None)
            batch_size: Override config batch size
            num_workers: Override config num_workers
            pin_memory: Override config pin_memory
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        
        # Use config values if not provided
        if model_name is None:
            model_name = config.MODEL_ARCHITECTURE
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        if num_workers is None:
            num_workers = config.NUM_WORKERS
        if pin_memory is None:
            pin_memory = config.PIN_MEMORY
        
        # Get model config
        model_config = config.MODEL_CONFIGS.get(model_name, config.CURRENT_MODEL_CONFIG)
        
        print(f"Creating data loaders for {model_name}")
        print(f"Image size: {model_config['image_size']}")
        print(f"Batch size: {batch_size}")
        print(f"Augmentation: {model_config['augmentation']}")
        
        # Create transforms
        train_transform = MedicalImageTransforms.get_train_transforms(
            image_size=model_config['image_size'],
            augmentation_strength=model_config['augmentation']
        )
        val_test_transform = MedicalImageTransforms.get_val_test_transforms(
            image_size=model_config['image_size']
        )
        
        # Create datasets
        datasets = {}
        for split in ['train', 'val', 'test']:
            transform = train_transform if split == 'train' else val_test_transform
            datasets[split] = ChestXRayDataset(
                data_dir=data_dir,
                split=split,
                transform=transform,
                image_size=model_config['image_size']
            )
        
        # Create data loaders
        dataloaders = {}
        for split, dataset in datasets.items():
            # Use config settings for shuffle and drop_last
            if split == 'train':
                shuffle = config.TRAIN_SHUFFLE
                drop_last = config.DROP_LAST_TRAIN
            elif split == 'val':
                shuffle = config.VAL_SHUFFLE
                drop_last = config.DROP_LAST_VAL
            else:  # test
                shuffle = config.TEST_SHUFFLE
                drop_last = config.DROP_LAST_TEST
            
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last
            )
        
        # Print dataset info
        print(f"\nDataset sizes:")
        for split, loader in dataloaders.items():
            print(f"  {split}: {len(loader.dataset)} images, {len(loader)} batches")
        
        return dataloaders
    
    @classmethod
    def get_sample_batch(
        cls,
        dataloader: DataLoader,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample batch from dataloader for testing.
        
        Args:
            dataloader: DataLoader to sample from
            device: Device to move tensors to
            
        Returns:
            Tuple of (images, labels) tensors
        """
        batch = next(iter(dataloader))
        images, labels = batch
        return images.to(device), labels.to(device)


def demonstrate_data_loaders():
    """Demonstrate usage of the data loaders."""
    
    data_dir = "/Users/pujasaha/Documents/cxp/chest_xray"
    
    print("CHEST X-RAY DATA LOADERS DEMONSTRATION")
    print("=" * 50)
    
    # Test different model configurations
    models_to_test = ['densenet121', 'efficientnet_b0', 'efficientnet_b3']
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name.upper()}:")
        print("-" * 30)
        
        try:
            # Create data loaders
            dataloaders = ModelSpecificDataLoaders.create_dataloaders(
                data_dir=data_dir,
                model_name=model_name,
                num_workers=2  # Reduced for demo
            )
            
            # Get sample batch
            sample_images, sample_labels = ModelSpecificDataLoaders.get_sample_batch(
                dataloaders['train']
            )
            
            print(f"Sample batch shape: {sample_images.shape}")
            print(f"Sample labels: {sample_labels[:5].tolist()}")
            print(f"Pixel value range: [{sample_images.min():.3f}, {sample_images.max():.3f}]")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")


if __name__ == "__main__":
    demonstrate_data_loaders()