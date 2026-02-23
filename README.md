# Hyperspectral Image Classification using PCA + ANN Ensemble

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Research](https://img.shields.io/badge/Focus-Research--Oriented-green)

---

## 📌 Project Overview

Hyperspectral imagery (HSI) contains rich spectral information across hundreds of contiguous bands. While this high dimensionality enables precise material discrimination, it also introduces:

- Spectral redundancy  
- Curse of dimensionality  
- Computational complexity  
- Risk of overfitting  

This project builds a **research-aligned experimental pipeline** for hyperspectral image classification using:

- Principal Component Analysis (PCA)
- Artificial Neural Network (ANN)
- Ensemble Learning
- Cross-Validation
- Detailed Performance Analysis

Dataset used: **Indian Pines (AVIRIS sensor)**

---

## 🎯 Research Motivation

Unlike RGB images, hyperspectral pixels contain spectral signatures across ~200 bands. Direct modeling can lead to poor generalization.

This project investigates:

- How PCA reduces spectral dimensionality
- How ANN performs on compressed spectral features
- Whether ensemble models improve robustness
- How per-class performance varies
- Cross-validation reliability

This mirrors research workflows used in remote sensing studies.

---

## 📂 Dataset Information

**Dataset:** Indian Pines  
**Spatial Size:** 145 × 145  
**Spectral Bands:** 200  
**Classes:** 16 land-cover categories  

Download from:

- https://www.ece.purdue.edu/~biehl/MultiSpec/hyperspectral.html  
- http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes  

Required files:

```
Indian_pines_corrected.mat
Indian_pines_gt.mat
```

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Flatten hyperspectral cube to pixel vectors
- Remove unlabeled pixels
- Standardize spectral features

### 2️⃣ Dimensionality Reduction
- PCA: 200 → 30 components
- Explained variance analysis
- Cumulative variance plot

### 3️⃣ ANN Classifier
- Multi-layer Perceptron (MLP)
- ReLU activation
- Batch Normalization
- Dropout Regularization
- CrossEntropy Loss

### 4️⃣ Hyperparameter Tuning
Grid search over:
- Hidden layer configurations
- Learning rates

### 5️⃣ Ensemble Learning
- Train 3 independent ANN models
- Soft voting aggregation
- Improved generalization

### 6️⃣ Evaluation Metrics
- Overall Accuracy
- Per-Class Accuracy
- Confusion Matrix
- Classification Report
- 5-Fold Cross Validation
- Training Loss Curve

---

## 📊 Experimental Outputs

The pipeline generates:

- PCA Explained Variance Plot
- Training Loss Curve
- Confusion Matrix
- Per-Class Accuracy
- Cross-Validation Accuracy

---

## 🛠 Tech Stack

- Python
- PyTorch
- NumPy
- SciPy
- Scikit-learn
- Matplotlib

---

## 📁 Project Structure

```
hyperspectral-ann/
│
├── Indian_pines_corrected.mat
├── Indian_pines_gt.mat
├── train.py
├── requirements.txt
├── results/
│   ├── pca_variance.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│
└── README.md
```

---

## ⚙ Installation

Create virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate      # windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch numpy scipy scikit-learn matplotlib
```

---

## ▶ How to Run

1. Download dataset (.mat files)
2. Place them in project root directory
3. Run:

```bash
python train.py
```

---

## 📈 Key Observations

- PCA significantly reduces dimensionality while retaining spectral variance.
- ANN performs effectively on compressed spectral signatures.
- Ensemble improves classification stability.
- Per-class analysis highlights imbalance effects.
- Cross-validation ensures robustness.

---

## 🔬 Future Improvements

- Spectral attention mechanisms
- Spatial-spectral hybrid modeling
- Class-weighted loss
- Bayesian uncertainty estimation
- Comparison with CNN/3D-CNN models

---

## 🎓 Research Relevance

This project demonstrates:

- Handling of high-dimensional spectral data
- Dimensionality reduction expertise
- Deep learning model design
- Ensemble strategy implementation
- Research-style evaluation
- Reproducible experimentation

Directly aligned with research in:

- Enhanced Neural Networks
- Hyperspectral Image Classification
- Ensemble Modeling
- Spectral Feature Analysis

---

## 👤 Author

**Shubham Chaudhary**  
B.Tech – Artificial Intelligence & Machine Learning  
BIT Mesra  

---
