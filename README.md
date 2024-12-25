# Melanoma Detection Using CNN
> This project implements a Convolutional Neural Network (CNN) to detect melanoma, a type of skin cancer. The solution processes images of nine different skin disease categories, balances class distribution, and trains a deep learning model for accurate detection.


## Table of Contents
* [Project Overview]
* [Dataset]
* [Project Pipeline]
* [Prerequisites]
* [Results]
* [Limitations]
* [Future Improvements]


## Project Overview
Melanoma accounts for 75% of skin cancer deaths. Early detection can save lives. This project leverages deep learning to:

- Analyze images from the International Skin Imaging Collaboration (ISIC) dataset.

- Detect melanoma among nine skin diseases.

- Balance imbalanced class distributions using data augmentation.


## Dataset
The dataset contains images classified into the following categories:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

Dataset Preprocessing:

- Images are resized to 180x180 pixels.
- The dataset is divided into training and validation subsets.
- Class distribution is examined and balanced using the Augmentor library.

## Project Pipeline
1. Data Reading and Understanding
   - Load and preprocess train and validation datasets.

2. Class Distribution Analysis
   - Examine class distributions and handle imbalances.

3. Data Augmentation
   - Enhance training data using rotation, zoom, and flipping.

4. Model Development
   - Build a custom CNN model without transfer learning.
   - Normalize image pixel values to the range [0,1].

5. Model Training
   - Train the model on the augmented dataset for 30 epochs.

6. Evaluation
   - Analyze training and validation accuracy and loss.

7. Findings
   - Summarize results and assess overfitting/underfitting.


## Prerequisites
- Python 3.8 or above
- TensorFlow 2.x
- Matplotlib
- NumPy
- Pandas
- Augmentor


## Results
After training for 30 epochs:

- The model achieves balanced training and validation accuracy.
- Overfitting and underfitting are addressed using data augmentation and dropout layers.

Visualizations:
- Training and validation accuracy/loss plots are generated to track progress.

## Limitations
- Requires a GPU for efficient training.
- Results depend on dataset quality and class balance.

## Future Improvements
- Implement transfer learning for faster convergence.
- Experiment with additional data augmentation techniques.
- Extend the model to detect additional skin diseases.
