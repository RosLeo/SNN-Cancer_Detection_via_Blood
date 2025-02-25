# SNN-Cancer-Detection-via-Blood

Welcome! Here you will find a set of Python scripts for:
1. Data preprocessing and splitting,
2. Training and evaluating a **Convolutional Neural Network (CNN)**,
3. Building and training a **Siamese network** for one-shot learning tasks.

The code is designed to handle a multi-class cancer classification problem, showcasing both traditional classification (with CNN) and a prototype/one-shot approach (with Siamese networks).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Data Preprocessing (data_preprocessing.py)](#data-preprocessing-datapreprocessingpy)
- [CNN Model (cnn_model.py)](#cnn-model-cnn_modelpy)
- [Siamese Model (siamese_model.py)](#siamese-model-siamese_modelpy)
- [Main Script (main.py)](#main-script-mainpy)
- [How to Run](#how-to-run)
- [Contact](#contact)

---

## Overview

The main goal of this repository is to demonstrate two approaches for a multi-class cancer classification problem using gene expression data (or similar features):



The dataset used in the example is located in an Excel file with multiple sheets (`Tables_S1_to_S11.xlsx`). You can adapt or replace this dataset with your own, provided you adjust the code accordingly.

---

## Project Structure

```plaintext
.
├── data_preprocessing.py     # Data loading, cleaning, SMOTE, splitting, and scaling.
├── cnn_model.py              # CNN architecture, training routine, and visualization.
├── siamese_model.py          # Siamese architecture, training loop, one-shot tasks, and confusion matrix logic.
├── main.py                   # Main script that orchestrates data processing, model training, and evaluation.
├── README.md                 # This file.
```

---

## Dependencies

The primary dependencies are:

- Python 3.9+  
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [tensorflow](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [imbalanced-learn](https://imbalanced-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [openpyxl](https://pypi.org/project/openpyxl/)

---

## Data Preprocessing (`data_preprocessing.py`)

### Key functionality:
- **load_and_preprocess_data**(`excel_path: str, sheet_name: int = 5`):
  - Reads the Excel file and performs data cleaning.
  - Applies SMOTE for class balancing.
  - Performs label encoding on the target variable.
  - Scales features using StandardScaler.
  - Splits the dataset into **train**, **validation**, and **test** sets.
  - Returns processed data arrays ready for training.

---

## CNN Model (`cnn_model.py`)

### Key functionality:
- **Creates a 1D CNN model** with convolutional layers, max-pooling, dropout, and dense layers.
- **Handles class imbalance** using weighted categorical cross-entropy.
- **Includes early stopping** and **model checkpointing**.
- **Plots training metrics**.

---

## Siamese Model (`siamese_model.py`)

### Key functionality:
- **Constructs a Siamese network** with a shared CNN branch.
- **Supports different similarity metrics** (L1, L2, Cosine similarity).
- **Trains iteratively** with on-the-fly batch creation.
- **Tests performance** using one-shot learning and confusion matrix evaluation.

---

## Main Script (`main.py`)

### Functionality:
- Loads and preprocesses the dataset.
- Trains the CNN model and evaluates it.
- Trains the Siamese network and evaluates its one-shot learning capability.
- Computes and plots the confusion matrix for the Siamese network.

---

## How to Run

1. **Clone** or download this repository.
2. **Install** the required libraries.
3. Place your **Excel dataset** in the specified path (or update the path in `main.py`).
4. **Execute** the main script:
   ```bash
   python main.py
   ```

---

## Contact

For any questions, issues, or suggestions, feel free to reach out via GitHub issues.

Happy coding!
