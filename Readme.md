# Exploring Mental Health Data

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/playground-series-s4e11)

This project explores machine learning approaches to predict depression in students and working professionals using various behavioral and environmental factors.

## Overview

The project implements a comprehensive data analysis and machine learning pipeline to predict depression, with separate models optimized for students and working professionals. The analysis demonstrates the importance of handling class imbalance and selecting appropriate evaluation metrics for mental health predictions.

## Key Features

- Separate prediction pipelines for students and working professionals
- Advanced handling of class imbalance using SMOTE and NUNS (Non-Uniform Negative Sampling)
- Thorough feature engineering and dimensionality reduction using PCA
- Optimization of multiple model architectures:
  - Logistic Regression variants
  - XGBoost
  - Random Forest
  - Model stacking
- Hyperparameter optimization using Tree-structured Parzen Estimators
- Comprehensive error analysis and model evaluation

## Technical Implementation

### Data Processing
- Intelligent handling of missing values through pattern analysis
- Feature engineering
- Separate preprocessing pipelines for student and professional datasets

### Model Development
- Implementation of various classification algorithms
- Advanced hyperparameter optimization
- Class imbalance handling through multiple techniques
- Custom implementation of the NUNS algorithm with scikit-learn integration

### Evaluation
- Multi-metric evaluation (MCC, F1-Score, Precision, Recall)
- Detailed error analysis across different feature categories
- Precision-Recall threshold optimization for deployment

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- optuna
- imbalanced-learn
- seaborn
- matplotlib

## Authors

- [Ilan Aliouchouche](https://github.com/ilanaliouchouche)
- [Karim Rochd](https://github.com/karimrochd)
- [Feddy Immoula](https://github.com/feddy321)

