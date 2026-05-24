# Face Mask Detection — CNN Binary Classifier

A deep learning project that detects whether a person 
is wearing a face mask using Convolutional Neural Networks (CNN)
built with TensorFlow/Keras.

## Project Overview
- **Task:** Binary image classification
- **Classes:** `with_mask` vs `without_mask`
- **Framework:** TensorFlow 2 / Keras
- **Best Accuracy:** 99% (Round 1, Model 2)

## Experiments
Three rounds of testing comparing different configurations:

| Round | Dense Layer | Dropout | Augmentation |
|-------|------------|---------|--------------|
| Round 1 | 128 | None / 0.3 | No / Yes |
| Round 2 | 64  | None / 0.3 | No / Yes |
| Round 3 | 128 | None / 0.2 | No / Yes |

## Model Architecture
- 3 × Conv2D + MaxPooling blocks
- Data Augmentation (Flip, Rotation, Zoom)
- Dropout regularisation
- Dense layer → Sigmoid output

## Dataset
- Source: Kaggle face mask dataset
- Image size: 128 × 128 pixels
- Split: 80% train / 20% validation

## Web App
Built with Flask — upload an image and get instant prediction.

## Results
Best model: Round 1 Model 2
- Accuracy: 99%
- Total errors: 25 / 2252 images
- Precision & F1: 0.99
