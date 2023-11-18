# Waterbirds Training Project

## Implemented Research Papers

This project implements the following research papers:

1. [Just Train Twice: Improving Group Robustness without Training Group Information](https://arxiv.org/pdf/2107.09044.pdf)

2. [Simple and Fast Group Robustness by Automatic Feature Reweighting](https://arxiv.org/pdf/2306.11074.pdf)

These papers serve as the basis for the training strategies incorporated into our models to enhance their robustness and improve out-of-distribution generalization.

## Introduction

One of the major challenges in machine learning is achieving robust out-of-distribution (OOD) generalization. This challenge often arises due to the reliance on spurious features—patterns that are predictive of the class label in the training data distribution but not causally related to the target.

![Dataset Example](https://drive.google.com/uc?id=1s4GXAyarhwQ9ewByHNJqYrh9BcsWMWmO)

The above figure provides a visual representation of our dataset, showcasing the images and their corresponding labels. This ongoing project, titled "Waterbirds Training," is aimed at addressing the challenge of OOD generalization through classification tasks on various distinct datasets. We employ three different models: CNN, Linear Model, and ResNet50 Model, to perform these classification tasks.

## Datasets

### Waterbirds Dataset (WILDS Package)

The Waterbirds dataset, formed by combining bird photographs from the [Caltech-UCSD Birds 200 (CUB) dataset](http://www.vision.caltech.edu/visipedia/CUB-200.html) with image backgrounds from the [Places dataset](http://places2.csail.mit.edu/download.html) (Zhou et al., 2017), is loaded using the [WILDS package](https://github.com/p-lambda/wilds). It serves as one of our primary datasets for classification.

### Representation Datasets

In addition to the Waterbirds dataset, we utilize three distinct representation datasets tailored to the task, all based on the Waterbirds images dataset:

- **ResNet50 Representation Dataset**
- **RegNet Representation Dataset**
- **DINOv2 Representation Dataset**

## Models

We employ the following three models for classification:

1. **CNN (Convolutional Neural Network)** - This is a custom CNN model designed for the task.

2. **Linear Model** - This model consists of a single linear layer and is tailored for representation datasets.

3. **ResNet50 Model** - A ResNet50-based model used for classification tasks.


## Model Representations and Results (Ongoing)

Here, we present the performance of our models on various representations. Please note that this section is a work in progress, and we will be adding more results soon.

### ResNet50 Representation

|   |Accuracy (%)|WGA (%)|
| :---: | :---: | :---: |
| **Standard** | 86.35 | 63.71 |
| **JTT** | 85.31 | 68.54 |

|   |Accuracy (%)|WGA (%)|
| :---: | :---: | :---: |
| **Standard** | 86.88 | 59.81 |
| **AFR** | 85.54 | 65.42 |


### RegNet Representation

|   |Accuracy (%)|WGA (%)|
| :---: | :---: | :---: |
| **Standard** | 93.25 | 75.39 |
| **JTT** | 93.82 | 87.38 |

|   |Accuracy (%)|WGA (%)|
| :---: | :---: | :---: |
| **Standard** | 95.20 | 81.00 |
| **AFR** | 94.53 | 83.49 |

### DINOv2 Representation 

|   |Accuracy (%)|WGA (%)|
| :---: | :---: | :---: |
| **Standard** | 96.13 | 90.03 |
| **JTT** | 96.19 | 91.28 |


## Contributions

Contributions to this ongoing project are welcome. If you have ideas for improvements or would like to collaborate, please feel free to submit issues or pull requests.

## Contact

For any questions, suggestions, or collaboration inquiries, you can reach out to Anna Khosrovyan at anna.khosrovyan.1806@gmail.com.