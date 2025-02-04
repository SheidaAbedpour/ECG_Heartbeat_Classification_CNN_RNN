# MIT-BIH ECG Classification using CNN-LSTM Model
This repository implements a deep learning model for ECG classification using the MIT-BIH Arrhythmia dataset. The model leverages a CNN-LSTM architecture for feature extraction and sequence modeling to classify ECG heartbeats into different categories. 

## Model Architecture

The model leverages a CNN-LSTM architecture to process ECG data for classification. The CNN layers extract features from the ECG signals, followed by a series of Bidirectional LSTM layers that capture temporal dependencies. Finally, fully connected layers are used for classification. Below is a detailed description of the model layers:

1. **Input Layer:**
   - The input data is reshaped to the format `(samples, time steps, features)`, where `samples` is the number of ECG instances, `time steps` is the length of each ECG signal sequence, and `features` is the number of features (usually 1 for raw ECG signal).
   - The shape of the input is `(X_train.shape[1], X_train.shape[2])`.

2. **CNN Layers:**
   - The model starts with 1D convolutional layers to automatically extract features from the ECG signals.
   - The first convolutional layer has 64 filters and a kernel size of 3, followed by MaxPooling to reduce the spatial dimensions and Dropout for regularization.
   - The second convolutional layer has 32 filters and a kernel size of 3, again followed by MaxPooling and Dropout to prevent overfitting.

3. **Bidirectional LSTM Layers:**
   - Two Bidirectional LSTM layers are used to model the sequential nature of ECG signals. The Bidirectional wrapper ensures the model considers both past and future sequences.
   - Each LSTM layer has 64 units and is followed by Dropout to prevent overfitting.
   - The first two LSTM layers return sequences to preserve the time-step dimensionality, while the final LSTM layer outputs a single vector for the sequence.

4. **Dense Layers:**
   - A fully connected dense layer with 100 units and ReLU activation is added to capture higher-level features.
   - A Dropout layer follows to mitigate overfitting.

5. **Output Layer:**
   - The output layer consists of softmax activation, making it suitable for multi-class classification, where the number of units corresponds to the number of classes.
  

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.9892    | 0.9951 | 0.9922   | 18118   |
| 1     | 0.8936    | 0.7554 | 0.8187   | 556     |
| 2     | 0.9611    | 0.9558 | 0.9584   | 1448    |
| 3     | 0.7988    | 0.8086 | 0.8037   | 162     |
| 4     | 0.9950    | 0.9857 | 0.9903   | 1608    |

### Overall Metrics

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 0.9843  |
| Macro Average | 0.9275  | 0.9001 | 0.9127 | 21892 |
| Weighted Avg  | 0.9840  | 0.9843 | 0.9840 | 21892 |

