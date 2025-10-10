from pathlib import Path

def count_images_in_dataset(base_path):
    """
    Count images in all subdirectories of the dataset
    """
    base = Path(base_path)
    
    if not base.exists():
        print(f"Path doesn't exist: {base_path}")
        return
    
    print(f"Dataset structure for: {base_path}")
    print("-" * 50)
    
    # Check all subdirectories
    for split_dir in base.iterdir():
        if split_dir.is_dir():
            print(f"\n{split_dir.name}/")
            
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    # Count image files
                    image_files = list(class_dir.glob("*.jpg")) + \
                                 list(class_dir.glob("*.jpeg")) + \
                                 list(class_dir.glob("*.png"))
                    
                    print(f"  {class_dir.name}: {len(image_files)} images")

# Run the analysis
dataset_path = "/home/psaha03/scratch/chest_xray_data_osn"
count_images_in_dataset(dataset_path)