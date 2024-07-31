## ProteinSequence.ipynb Notebook : Protein Classifier based on their sequences

The ProteinSequence.ipynb notebook includes a project that uses machine learning to classify proteins based on their sequences. The code processes data from protein files, extracts key characteristics, and then uses a neural network to predict which functional category each protein belongs to. It leverages modern computing techniques, including GPU acceleration if available, to enhance processing speed.
  
## Why It Matters

Proteins are crucial in biology and medicine, and understanding their functions can lead to breakthroughs in fields like drug discovery and disease treatment. This project automates a part of this understanding process, making it faster and potentially more accurate than manual analysis. It's a step towards leveraging technology to solve complex biological problems, making science more accessible and impactful.

## ProteinSolubility.ipynb Notebook:  Prediction Application using Neural Networks with PyTorch

This project demonstrates the application of a neural network for predicting protein solubility using PyTorch. The dataset used is from the UCI Machine Learning Repository, focusing on predicting protein solubility based on various features.

## Table of Contents

- [Biotech Application of Neural Networks using PyTorch](#biotech-application-of-neural-networks-using-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Importing Libraries](#importing-libraries)
  - [Loading the Dataset](#loading-the-dataset)
  - [Extracting Features and Labels](#extracting-features-and-labels)
  - [Encoding Labels](#encoding-labels)
  - [Handling Non-Binary Labels](#handling-non-binary-labels)
  - [Standardizing the Features](#standardizing-the-features)
  - [Balancing the Dataset](#balancing-the-dataset)
  - [Converting to PyTorch Tensors](#converting-to-pytorch-tensors)
  - [Splitting Data into Training and Validation Sets](#splitting-data-into-training-and-validation-sets)
  - [Defining the Neural Network Model](#defining-the-neural-network-model)
  - [Loss Function and Optimizer](#loss-function-and-optimizer)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Testing with New Data](#testing-with-new-data)
 
## Importing Libraries

First, we import the necessary libraries:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler```
 
## Loading the Dataset:
We load the dataset from the UCI Machine Learning Repository:

python
```dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
data = pd.read_csv(dataset_url)```


# Extracting Features and Labels
# We separate the dataset into features and labels:

```label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)```


Handling Non-Binary Labels

Ensure labels are binary:

python

if encoded_labels.max() > 1:
    encoded_labels = (encoded_labels == encoded_labels.max()).astype(float)

Standardizing the Features

Standardization scales the features:

python

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

Balancing the Dataset

We balance the dataset using RandomOverSampler:

python

ros = RandomOverSampler()
features_balanced, labels_balanced = ros.fit_resample(features_standardized, encoded_labels)

Converting to PyTorch Tensors

Convert data to PyTorch tensors:

python

features_tensor = torch.tensor(features_balanced, dtype=torch.float32)
labels_tensor = torch.tensor(labels_balanced, dtype=torch.float32).view(-1, 1)

Splitting Data into Training and Validation Sets

Split the data into training and validation sets:

python

features_train, features_val, labels_train, labels_val = train_test_split(features_tensor, labels_tensor, test_size=0.2, random_state=42)

Defining the Neural Network Model

Define the neural network:

python

class SolubilityPredictor(nn.Module):
    def __init__(self, input_size):
        super(SolubilityPredictor, self).__init__()
        self.hidden1 = nn.Linear(input_size, 100)
        self.hidden2 = nn.Linear(100, 50)
        self.output = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return torch.sigmoid(x)

input_size = features_tensor.shape[1]
model = SolubilityPredictor(input_size)

Loss Function and Optimizer

Define the Focal Loss function and optimizer:

python

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss

criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

Training the Model

Train the model for 100 epochs:

python

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(features_train)
    loss = criterion(outputs, labels_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

Evaluating the Model

Evaluate the model using the validation set:

python

model.eval()
with torch.no_grad():
    predictions = model(features_val)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == labels_val).sum() / len(labels_val)
    print(f'Validation Accuracy: {accuracy.item():.4f}')

    # Print classification report
    print(classification_report(labels_val.numpy(), predicted_classes.numpy(), target_names=['Class 0', 'Class 1']))

# Testing with New Data
# Predict on new data:

python

new_data = [
    [10.5, 9000.0, 2500.0, 0.3, 100.0, 1.5e6, 150.0, 3000.0, 60],
    [5.0, 3000.0, 1000.0, 0.2, 50.0, 0.8e6, 80.0, 2000.0, 40]
]

# Standardize new data using the same scaler as the training data
new_data_standardized = scaler.transform(new_data)

# Convert to PyTorch tensor
new_data_tensor = torch.tensor(new_data_standardized, dtype=torch.float32)

# Predict using the trained model
model.eval()
with torch.no_grad():
    predictions = model(new_data_tensor)
    predicted_classes = (predictions > 0.5).float()

# Print the predictions
print(f"Predictions: {predictions.numpy()}")
print(f"Predicted Classes: {predicted_classes.numpy()}")

This complete process involves loading and preprocessing data, building and training a neural network model, and finally evaluating and predicting using the trained model. This demonstrates a practical application of neural networks in the biotech field using PyTorch.

