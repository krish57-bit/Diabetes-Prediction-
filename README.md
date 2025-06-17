# 🧠 Diabetes Prediction using Machine Learning

This project demonstrates a full machine learning workflow to predict diabetes using the **PIMA Indian Diabetes Dataset**. It includes detailed data preprocessing, visualization, outlier handling, and classification using various ML models.

---

## 📁 Dataset

- **Source**: [PIMA Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Size**: 768 samples, 8 features + 1 binary target (`Outcome`)
- **Target**: `Outcome` (1 = Diabetic, 0 = Non-Diabetic)

---

## ⚙️ Workflow Overview

### 1. 📥 Data Loading & Exploration
- Checked for missing/zero values
- Descriptive statistics using `.describe()`
- Correlation heatmap (`sns.heatmap`)

### 2. 🛠️ Data Imputation
- Replaced 0s in critical columns (Insulin, BMI, etc.) with median/mean values based on distribution

### 3. 📦 Outlier Detection
- Used IQR method
- Boxplot visualization for identifying outliers

### 4. 🧼 Feature Scaling
- Applied `StandardScaler` to normalize all features

### 5. 🧪 Train-Test Split
- Used 67% training and 33% testing with `train_test_split()`

### 6. ⚖️ Imbalanced Data Handling
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the target classes

### 7. 🔍 Model Training
#### ✅ Logistic Regression
#### ✅ Gaussian Naive Bayes
#### ✅ K-Nearest Neighbors (KNN)

### 8. 📈 Model Evaluation
- Accuracy, Confusion Matrix, and Classification Report (Precision, Recall, F1-score)

---

## 📊 Visualizations

- ✅ Correlation Matrix Heatmap
- ✅ Feature Distribution using `sns.distplot`
- ✅ Boxplots before and after standardization

All saved under `images/`.

---

## 🧠 Models

| Model                | Evaluation Metric     | Balanced with SMOTE |
|---------------------|------------------------|----------------------|
| Logistic Regression | Accuracy + Recall      | ✅                   |
| Gaussian NB         | Confusion Matrix + F1  | ✅                   |
| KNN Classifier      | Accuracy + Classification Report | ✅         |

---

## 🧾 Requirements

Install with:

```bash
pip install -r requirements.txt
