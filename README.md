# Breast Cancer Classification using Machine Learning

## Overview
This project builds and compares multiple machine learning models to classify breast tumors as malignant or benign using diagnostic features.

## Models Used
- Logistic Regression
- Support Vector Machine (RBF)
- Random Forest

## Workflow
1. Exploratory Data Analysis
2. Feature Scaling
3. Model Training & Comparison
4. Cross-Validation
5. Evaluation using Accuracy and Confusion Matrix

## Results

### Why Logistic Regression and SVM performed better than Random Forest

Logistic Regression and SVM achieved the highest test accuracy (98.24%), indicating that the Breast Cancer dataset is largely linearly separable after feature scaling. Since most features are continuous and well-behaved, linear decision boundaries with proper regularization are sufficient to capture the underlying patterns. Random Forest underperformed slightly because tree-based models can struggle when the signal is mostly linear and the dataset size is relatively small, leading to higher variance and less stable splits.

### Summary

This result suggests that simpler, well-regularized models can outperform more complex ensembles when the data is clean, structured, and close to linearly separable.


## Tech Stack
- Python
- scikit-learn
- pandas
- matplotlib
- seaborn
