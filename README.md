# ASL Alphabet Recognition

A deep learning model for recognizing American Sign Language (ASL) alphabet signs using transfer learning with ResNet18.

## Overview

This project implements a convolutional neural network for classifying ASL alphabet hand signs. The model uses transfer learning with a pre-trained ResNet18 architecture, fine-tuned on the ASL alphabet dataset to achieve high accuracy in sign language recognition.

## Features

- **Transfer Learning**: Utilizes ResNet18 pre-trained on ImageNet
- **Mixed Precision Training**: Automatic mixed precision (AMP) support for faster training on CUDA devices
- **Data Augmentation**: Comprehensive augmentation pipeline including rotation, flipping, and color jittering
- **GPU Acceleration**: Optimized for CUDA with fallback to CPU
- **Model Persistence**: Saves trained model with class labels for inference

## Requirements

```
numpy
pandas
torch
torchvision
matplotlib
```

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd asl
```

2. Install dependencies:

```bash
pip install numpy pandas torch torchvision matplotlib
```

3. Download the ASL Alphabet dataset and organize it as follows:

```
asl-alphabet/
├── asl_alphabet_train/
│   └── asl_alphabet_train/
│       ├── A/
│       ├── B/
│       ├── C/
│       └── ... (other letters)
└── asl_alphabet_test/
    ├── A/
    ├── B/
    └── ... (other letters)
```

## Model Architecture

- **Base Model**: ResNet18 (pre-trained on ImageNet)
- **Custom Classifier**:
  - Linear layer (512 units) with ReLU activation
  - Dropout (0.4)
  - Output layer (29 classes for ASL alphabet)
- **Training Strategy**: Only the final fully connected layers are trained; all ResNet18 layers remain frozen

## Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Image Size    | 224×224          |
| Batch Size    | 32               |
| Epochs        | 15               |
| Learning Rate | 1e-4             |
| Optimizer     | Adam             |
| Loss Function | CrossEntropyLoss |
| Workers       | 4                |

### Data Augmentation (Training)

- Random horizontal flip
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation ±30%)
- ImageNet normalization

## Usage

### Training

Run the training script:

```bash
python asl.py
```

The script will:

1. Load and preprocess the dataset
2. Initialize the ResNet18 model with custom classifier
3. Train for 15 epochs with validation
4. Save the trained model as `sign_language_resnet18.pth`
5. Display a training/validation accuracy plot

### Output

The trained model is saved with the following structure:

```python
{
    'model_state_dict': model.state_dict(),
    'class_names': train_data.classes
}
```

## Performance Monitoring

The training script provides:

- Real-time training and validation metrics per epoch
- Training loss and accuracy
- Validation loss and accuracy
- Post-training accuracy plot (matplotlib visualization)

## Hardware Recommendations

- **GPU**: NVIDIA GPU with CUDA support (recommended for faster training)
- **CPU**: Falls back to CPU if CUDA is unavailable
- **RAM**: 8GB+ recommended
- **Storage**: Depends on dataset size

## Code Structure

```
asl/
├── asl.py              # Main training script
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── asl-alphabet/       # Dataset directory (not included)
```

## Key Features in Code

- **Mixed Precision Training**: Uses `torch.amp.autocast()` and `GradScaler` for efficient GPU utilization
- **Pin Memory**: Enabled for faster data transfer to GPU
- **Non-blocking Transfer**: Asynchronous data transfer to device
- **Gradient Accumulation**: Proper weighted loss calculation across batches

## Future Improvements

- [ ] Add learning rate scheduling
- [ ] Implement early stopping
- [ ] Add confusion matrix visualization
- [ ] Support for real-time webcam inference
- [ ] Experiment with other architectures (ResNet50, EfficientNet)
- [ ] Add model evaluation metrics (precision, recall, F1-score)

## License

[Add your license here]

## Acknowledgments

- ASL Alphabet dataset
- PyTorch and torchvision teams
- ResNet architecture by He et al.

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
