import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

def check_dataset_structure(data_dir):
    """Check the actual structure of your dataset"""
    data_path = Path(data_dir)
    print(f"Checking structure of: {data_path}")
    print("-" * 50)
    
    # Check if direct NORMAL/PNEUMONIA folders exist
    normal_dir = data_path / "NORMAL"
    pneumonia_dir = data_path / "PNEUMONIA"
    
    if normal_dir.exists() and pneumonia_dir.exists():
        print("✓ Found direct NORMAL/PNEUMONIA structure")
        normal_count = len(list(normal_dir.glob("*.*")))
        pneumonia_count = len(list(pneumonia_dir.glob("*.*")))
        print(f"NORMAL: {normal_count} files")
        print(f"PNEUMONIA: {pneumonia_count} files")
        return "direct"
    
    # Check for train/test/val structure
    for split in ["train", "test", "val"]:
        split_dir = data_path / split
        if split_dir.exists():
            print(f"✓ Found {split}/ directory")
            normal_split = split_dir / "NORMAL"
            pneumonia_split = split_dir / "PNEUMONIA"
            if normal_split.exists() and pneumonia_split.exists():
                normal_count = len(list(normal_split.glob("*.*")))
                pneumonia_count = len(list(pneumonia_split.glob("*.*")))
                print(f"  {split}/NORMAL: {normal_count} files")
                print(f"  {split}/PNEUMONIA: {pneumonia_count} files")
    
    return "split"

def oversample_with_augmentation(data_dir, split_name=None, target_ratio=0.5):
    """
    Oversample minority class using image augmentation
    """
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    base_path = Path(data_dir)
    
    # Determine the correct path structure
    if split_name:
        normal_dir = base_path / split_name / "NORMAL"
        pneumonia_dir = base_path / split_name / "PNEUMONIA"
    else:
        normal_dir = base_path / "NORMAL"
        pneumonia_dir = base_path / "PNEUMONIA"
    
    # Check if directories exist
    if not normal_dir.exists():
        print(f"❌ NORMAL directory not found: {normal_dir}")
        return False
    if not pneumonia_dir.exists():
        print(f"❌ PNEUMONIA directory not found: {pneumonia_dir}")
        return False
    
    # Image augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Count existing images
    normal_images = list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.jpg")) + list(normal_dir.glob("*.png"))
    pneumonia_images = list(pneumonia_dir.glob("*.jpeg")) + list(pneumonia_dir.glob("*.jpg")) + list(pneumonia_dir.glob("*.png"))
    
    normal_count = len(normal_images)
    pneumonia_count = len(pneumonia_images)
    
    print(f"Current counts - Normal: {normal_count}, Pneumonia: {pneumonia_count}")
    
    if normal_count == 0:
        print("❌ No normal images found!")
        return False
    if pneumonia_count == 0:
        print("❌ No pneumonia images found!")
        return False
    
    # Calculate target
    total_target = pneumonia_count / (1 - target_ratio)
    normal_target = int(total_target * target_ratio)
    images_to_add = normal_target - normal_count
    
    if images_to_add <= 0:
        print("✓ No oversampling needed - normal class already balanced or larger")
        return True
    
    print(f"Need to generate {images_to_add} augmented normal images...")
    
    # Generate augmented images
    for i in range(images_to_add):
        if i % 100 == 0:  # Progress indicator
            print(f"Progress: {i}/{images_to_add}")
            
        # Select random source image
        source_img_path = random.choice(normal_images)
        
        # Load and preprocess image
        img = cv2.imread(str(source_img_path))
        if img is None:
            print(f"❌ Could not load image: {source_img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        
        # Generate augmented image
        aug_iter = datagen.flow(img, batch_size=1)
        aug_img = next(aug_iter)[0].astype(np.uint8)
        
        # Save augmented image
        new_name = f"{source_img_path.stem}_aug_{i:04d}.jpg"
        new_path = normal_dir / new_name
        cv2.imwrite(str(new_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
    
    print(f"✓ Augmentation complete! New normal count: {normal_target}")
    return True

# Usage
data_path = "/home/psaha03/scratch/chest_xray_data_osn"

# First, check the structure
structure_type = check_dataset_structure(data_path)

# Then run oversampling based on structure
if structure_type == "direct":
    oversample_with_augmentation(data_path)
else:
    # For train/test/val structure, specify which split to oversample
    print("\nWhich split do you want to oversample?")
    print("Options: train, test, val")
    
    # For now, let's try train split
    success = oversample_with_augmentation(data_path, split_name="train")
    if not success:
        print("Trying without split...")
        oversample_with_augmentation(data_path)