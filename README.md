# ECG Classification Models

This repository contains two ECG classification projects utilizing deep learning models for classifying ECG heartbeats into various categories. Below is an overview of the two models included in the repository.

## 1. MIT-BIH ECG Classification using CNN-LSTM Model
Classifies ECG heartbeats using the MIT-BIH Arrhythmia dataset. The model leverages CNN for feature extraction and LSTM for sequence modeling.

### Model Architecture:
- **CNN Layers**: 1D convolutional layers for feature extraction.
- **LSTM Layers**: Bidirectional LSTM layers to capture temporal dependencies.
- **Output**: Softmax activation for multi-class classification.

### Performance:
- **Accuracy**: 98.43%
- **Macro Avg**: 92.75%
- **Weighted Avg**: 98.40%

## 2. ECG Classifier using CNN + BiLSTM
Classifies ECG signals as normal or abnormal using the PTB Diagnostic ECG Database (PTBDB). The model combines CNN for feature extraction and BiLSTM for sequence modeling.

### Model Architecture:
- **CNN Layer**: 1D convolutional layer.
- **BiLSTM Layers**: Four Bidirectional LSTM layers.
- **Output**: Sigmoid activation for binary classification.

### Performance:
- **Accuracy**: 99.08%
- **Macro Avg**: 98.85%
- **Weighted Avg**: 99.08%

## Requirements:
- `tensorflow`, `keras`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`
