# waterbirds-training

## Overview
This project aims to perform classification tasks using two distinct datasets and three different models. The datasets include the Waterbirds dataset, loaded via the WILDS package, and the ResNet50 representation dataset. The three models used for classification are CNN, Linear Model, and ResNet50 Model.


## Project Goals

The primary goals of this project are as follows:

-  Classification: Perform classification tasks using the given datasets and models.
-  Performance Evaluation: Measure the accuracy of the classification models on both the training and test sets.
-  Group Accuracy: Evaluate the accuracy of the models on four specific groups formed by the background spurious attribute and bird type label.


## Datasets
-  Waterbirds Images Dataset
-  ResNet50 Representation Dataset


## Models
-  CNN Model
-  LinearClassifier Model
-  ResNet50 Model


## Combining Datasets and Models

In this project, we discuss and evaluate the following combinations:

-  Waterbirds Images Dataset with CNN Model
-  Waterbirds Images Dataset with ResNet50 Model
-  ResNet50 Representation Dataset with LinearClassifier Model

# Results

## Waterbirds Images Dataset with CNN Model

-  Training Set Accuracy**: 99.65%
-  Test Set Accuracy**: 62.81%
### Group-based Accuracy
-  Accuracy on waterbirds with water background**: 65.89%
-  Accuracy on waterbirds with land background**: 4.67%
-  Accuracy on landbirds with land background**: 96.72%
-  Accuracy on landbirds with water background**: 44.61%


## Waterbirds Images Dataset with ResNet50 Model

-  Training Set Accuracy**: 99.62%
-  Test Set Accuracy**: 66.86%
### Group-based Accuracy
-  Accuracy on waterbirds with water background**: 59.03%
-  Accuracy on waterbirds with land background**: 2.18%
-  Accuracy on landbirds with land background**: 98.00%
-  Accuracy on landbirds with water background**: 56.36%


## ResNet50 Representation Dataset with LinearClassifier Model

-  Training Set Accuracy**: 99.58%
-  Test Set Accuracy**: 86.38%
### Group-based Accuracy
-  Accuracy on waterbirds with water background**: 93.93%
-  Accuracy on waterbirds with land background**: 64.17%
-  Accuracy on landbirds with land background**: 99.20%
-  Accuracy on landbirds with water background**: 77.74%
