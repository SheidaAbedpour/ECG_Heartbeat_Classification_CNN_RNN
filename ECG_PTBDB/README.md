# ECG Classifier using CNN + BiLSTM
This project implements a deep learning-based classifier for electrocardiogram (ECG) signals. The classifier utilizes a combination of Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) layers. The goal of the project is to classify ECG signals into two categories: normal and abnormal.
The classifier uses the PTB Diagnostic ECG Database (PTBDB), which contains ECG signals labeled as normal and abnormal.

## Model Architecture
The model is built using the Keras API:
- **Input Layer**: Takes the reshaped ECG signal with a shape of `(timesteps, features)`.
- **CNN Layer**: A 1D convolutional layer is used to extract important features from the ECG signals.
- **BiLSTM Layers**: Four Bidirectional LSTM layers capture the temporal dependencies in the ECG signals.
- **Fully Connected Layers**: Dense layers with ReLU activations followed by a final output layer with a sigmoid activation function for binary classification.

## Model Training
- The model is compiled using the Adam optimizer with a learning rate of `1e-3` and binary cross-entropy loss.
- The model is trained with early stopping (patience of 3 epochs) and learning rate reduction (by a factor of 0.5 after 3 epochs of no improvement).
- The training process will continue for a specified number of epochs (default 150) or until early stopping conditions are met.

## Classification Report:

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 0.0    | 0.9900    | 0.9769 | 0.9834   | 607     |
| 1.0    | 0.9912    | 0.9962 | 0.9937   | 1576    |

- **Accuracy**: 0.9908 (2183 samples)
- **Macro avg**: Precision 0.9906, Recall 0.9866, F1-Score 0.9885 (2183 samples)
- **Weighted avg**: Precision 0.9908, Recall 0.9908, F1-Score 0.9908 (2183 samples)
