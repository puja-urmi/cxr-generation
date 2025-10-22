"""
Data Loaders for Chest X-Ray Dataset
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import config

logger = logging.getLogger(__name__)


class ChestXRayDataset(Dataset):
    """Custom Dataset for Chest X-Ray images."""
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = None
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
        self.image_size = image_size if image_size is not None else config.IMG_SIZE
        
        # Class to index mapping
        self.class_to_idx = {'HEALTHY': 0, 'UNHEALTHY': 1}
        self.idx_to_class = {0: 'HEALTHY', 1: 'UNHEALTHY'}

        # Load image paths and labels
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} images for {split} split")
        logger.info(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image paths and their corresponding labels."""
        samples = []
        
        for class_name in ['HEALTHY', 'UNHEALTHY']:
            class_dir = self.data_dir / self.split / class_name
            if class_dir.exists():
                # Load all common image formats
                class_count = 0
                for ext in ['*.jpeg', '*.jpg', '*.png']:
                    for img_path in class_dir.glob(ext):
                        label = self.class_to_idx[class_name]
                        samples.append((img_path, label))
                        class_count += 1
                
                logger.info(f"Found {class_count} {class_name} images in {class_dir}")
                if class_count == 0:
                    logger.warning(f"No images found in {class_dir} for supported formats (jpeg, jpg, png)")
            else:
                logger.error(f"Class directory not found: {class_dir}")
        
        if len(samples) == 0:
            raise ValueError(f"No images found in {self.data_dir}/{self.split}. Check directory structure and image formats.")
        
        return samples
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {'HEALTHY': 0, 'UNHEALTHY': 0}
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
    """Medical imaging transformations."""
    
    @staticmethod
    def get_train_transforms() -> transforms.Compose:
        """Get training transformations with augmentation."""
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.INFERENCE_MEAN,
                std=config.INFERENCE_STD
            )
        ])
    
    @staticmethod
    def get_val_test_transforms() -> transforms.Compose:
        """Get validation/test transformations (no augmentation)."""
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.INFERENCE_MEAN,
                std=config.INFERENCE_STD
            )
        ])


class ModelSpecificDataLoaders:
    """Factory class for creating data loaders."""
    
    @classmethod
    def create_dataloaders(cls, data_dir: str) -> Dict[str, DataLoader]:
        """
        Create data loaders using config settings.
        
        Args:
            data_dir: Path to the chest_xray directory
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        logger.info(f"Creating data loaders for {config.MODEL_ARCHITECTURE}")
        logger.info(f"Image size: {config.IMG_SIZE}")
        logger.info(f"Batch size: {config.BATCH_SIZE}")
        
        # Create transforms
        train_transform = MedicalImageTransforms.get_train_transforms()
        val_test_transform = MedicalImageTransforms.get_val_test_transforms()
        
        # Create datasets
        datasets = {}
        for split in ['train', 'val', 'test']:
            transform = train_transform if split == 'train' else val_test_transform
            datasets[split] = ChestXRayDataset(
                data_dir=data_dir,
                split=split,
                transform=transform,
                image_size=config.IMG_SIZE
            )
        
        # Create data loaders
        dataloaders = {}
        for split, dataset in datasets.items():
            shuffle = split == 'train'  # Only shuffle training data
            
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=shuffle,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                drop_last=False
            )
        
        # Print dataset info
        logger.info("Dataset sizes:")
        for split, loader in dataloaders.items():
            logger.info(f"  {split}: {len(loader.dataset)} images, {len(loader)} batches")
        
        return dataloaders
