# UIC Data Mining Project

## Overview

This is a course project for the Data Mining class at Beijing Normal University - Hong Kong Baptist University United International College (UIC). Using the **Car Evaluation Dataset**, it applies three machine learning algorithms — K-Nearest Neighbors (KNN), Logistic Regression, and Random Forest — to perform multi-class classification on overall car acceptability.

---

## Dataset

The dataset is based on the classic Car Evaluation dataset and contains 6 categorical features along with 1 target variable:

| Feature | Description | Values |
|---------|-------------|--------|
| `buying` | Buying price | low / med / high / vhigh |
| `maint` | Maintenance cost | low / med / high / vhigh |
| `doors` | Number of doors | 2 / 3 / 4 / 5more |
| `persons` | Passenger capacity | 2 / 4 / more |
| `lug_boot` | Luggage boot size | small / med / big |
| `safety` | Estimated safety | low / med / high |
| `evaluation` | **Target variable** | unacc / acc / good / vgood |

- Training set: `training.csv` — **1,330** samples
- Test set: `test.csv` — **333** samples

All features are categorical and are encoded numerically using `LabelEncoder` during preprocessing.

---

## Project Structure

```
.
├── training.csv              # Training dataset
├── test.csv                  # Test dataset
├── KNN.ipynb                 # KNN implementation (including from-scratch version)
├── Logistic Regression.ipynb # Logistic Regression training and tuning
├── Random_Forest.ipynb       # Random Forest training and tuning
└── README.md
```

---

## Models

### 1. K-Nearest Neighbors (KNN)
- Includes a from-scratch KNN implementation in pure Python (no sklearn dependency)
- Covers core components: dataset loading, Euclidean distance calculation, and majority-vote classification

### 2. Logistic Regression
- Uses `sklearn.linear_model.LogisticRegression` with `solver='newton-cg'` and `multi_class='multinomial'`
- Plots **learning curves** and **validation curves** to analyze model behavior
- Hyperparameter tuning via `GridSearchCV` (5-fold CV) over `C` and `solver`

### 3. Random Forest
- Uses `sklearn.ensemble.RandomForestClassifier`
- Validation curves plotted for both `n_estimators` and `max_features`
- Hyperparameter tuning via `GridSearchCV` (10-fold CV)
- Evaluated using **Accuracy** and **Macro F1-Score**

---

## Exploratory Data Analysis

The EDA phase includes the following visualizations:

- Class distribution of the target variable `evaluation` (`sns.countplot`)
- Grouped bar charts of each feature broken down by `evaluation` label
- Feature correlation heatmap (`sns.heatmap`)

---

## Requirements

```bash
Python >= 3.6
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Usage

Open and run the notebooks using Jupyter:

```bash
jupyter notebook
```

Recommended order:
1. `Logistic Regression.ipynb` — Data exploration and baseline modeling
2. `Random_Forest.ipynb` — Ensemble method with hyperparameter tuning
3. `KNN.ipynb` — From-scratch algorithm implementation
