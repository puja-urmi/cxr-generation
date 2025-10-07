# X-ray Binary Classifier

A deep learning project for binary classification of chest X-ray images into normal and abnormal categories.

## Project Overview

This project implements a convolutional neural network (CNN) based classifier for medical X-ray images. The system can identify whether an X-ray shows normal anatomy or abnormalities, which can assist radiologists and healthcare providers in their diagnostic workflow.

The classifier uses deep learning techniques with pre-trained models fine-tuned for X-ray image analysis. It incorporates data augmentation strategies to enhance model generalization and provides comprehensive evaluation metrics to assess performance.

## Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8 or higher
- CUDA-enabled GPU (recommended for training)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/puja-urmi/xray-classifier.git
   cd xray-classifier
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Data Preparation

The project supports multiple chest X-ray datasets:

#### NIH Chest X-ray Dataset
```
python download_data.py
```

#### Chest X-ray Pneumonia Dataset
```
python download_data.py
```
   
After downloading, prepare the dataset CSV files:
```
python -c "import data; data.prepare_data_csv('/path/to/images', patterns={'normal': 0, 'abnormal': 1})"
```

This will:
- Scan the image directory
- Split data into train/validation/test sets
- Create CSV files with image paths and labels## Usage

### Training

Train the model with the following command:

```
python train.py --img_dir /path/to/images
```

Optional arguments:
- `--resume`: Resume training from the last checkpoint

The training process includes:
- Data augmentation (rotations, flips, etc.)
- Automatic learning rate adjustment
- Early stopping to prevent overfitting
- Regular validation to monitor progress
- Checkpoint saving for best-performing models

### Evaluation

Evaluate the model on test data:

```
python evaluate.py --img_dir /path/to/images
```

Optional arguments:
- `--checkpoint`: Path to a specific model checkpoint

This will generate:
- Accuracy, precision, recall, F1-score metrics
- ROC curve and AUC score
- Confusion matrix
- Classification report

### Prediction

Make predictions on new X-ray images:

```
python predict.py --input /path/to/image_or_directory --visualize --output /path/to/save/results
```

Options:
- `--input`: Path to a single image or directory of images
- `--visualize`: Generate visual explanations of predictions
- `--output`: Directory to save results
- `--checkpoint`: Specific model checkpoint to use

## Model Architecture

The classifier supports multiple architectures:

- **DenseNet121** (default): Excellent feature extraction with fewer parameters
- **ResNet18**: Good balance of performance and speed
- **EfficientNet-B0**: State-of-the-art performance with parameter efficiency

The model consists of:
1. Pre-trained backbone (feature extractor)
2. Custom classifier head with dropout for regularization
3. Binary output for normal/abnormal classification

## Results Visualization

The project includes comprehensive visualization tools:

- **Learning curves**: Training and validation metrics over time
- **ROC curves**: Performance at different classification thresholds
- **Precision-recall curves**: Alternative view of model performance
- **Confusion matrix**: Visualize true/false positives and negatives
- **Grad-CAM**: Highlight regions of the image that influenced the classification

Example usage:
```python
from visualize import plot_learning_curves, plot_roc_curve
import pickle

# Load training history
with open('results/training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Plot learning curves
plot_learning_curves(history, save_path='results/learning_curves.png')
```

## Project Structure

```
xray-classifier/
├── config.py           # Configuration settings
├── data.py             # Data loading and preprocessing
├── download_data.py    # NIH dataset downloader
├── download_cxp.py     # Pneumonia dataset downloader
├── model.py            # Model architecture definition
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── predict.py          # Inference script
├── visualize.py        # Visualization utilities
├── requirements.txt    # Package dependencies
└── README.md           # Project documentation
```

## Performance Optimization

For improved training performance:
- Use GPU acceleration when available
- Adjust batch size based on available memory
- Consider mixed precision training for newer GPUs
- Experiment with different pre-trained backbones

## Troubleshooting

Common issues and solutions:

- **CUDA out of memory**: Reduce batch size in config.py
- **Slow training**: Check for proper GPU utilization
- **Poor accuracy**: Try different data augmentation or learning rate
- **Overfitting**: Increase dropout or add regularization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

To contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NIH for providing the Chest X-rays Dataset
- Paul Mooney for the Chest X-ray Pneumonia Dataset
- PyTorch team for the deep learning framework
- Medical professionals who helped with data annotation