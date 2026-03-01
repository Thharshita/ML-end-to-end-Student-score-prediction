# Student Performance Prediction - ML End-to-End Project

## Overview
A complete machine learning pipeline that predicts student exam performance based on demographic and educational factors. The project demonstrates end-to-end ML workflow from data ingestion to model deployment.

## Problem Statement
Analyzes how student test scores are influenced by:
- Gender
- Ethnicity
- Parental education level
- Lunch type
- Test preparation course completion

## Dataset
- **Source**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Size**: 1000 rows, 8 columns
- **Features**: 5 categorical, 3 numerical (math, reading, writing scores)

## Project Structure
```
├── artifacts/          # Trained models and processed data
├── notebook/          # EDA and model training notebooks
├── src/
│   ├── components/    # Data ingestion, transformation, model training
│   ├── pipeline/      # Training and prediction pipelines
│   ├── exception.py   # Custom exception handling
│   ├── logger.py      # Logging configuration
│   └── utils.py       # Helper functions
├── logs/              # Application logs
└── setup.py           # Package configuration
```

## Key Features
- **Modular Design**: Separate components for data processing and model training
- **Multiple ML Models**: Tests Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, XGBoost, CatBoost
- **Automated Pipeline**: End-to-end workflow from raw data to predictions
- **Logging & Exception Handling**: Comprehensive error tracking
- **Reproducible**: Packaged as installable Python module

## Installation
```bash
pip install -r requirement.txt
```

## Dependencies
- pandas, numpy, scikit-learn
- xgboost, catboost
- matplotlib, seaborn
- dill (model serialization)

## Usage
Run the complete pipeline:
```python
from src.components.data_ingestion import DataIngestion
obj = DataIngestion()
train_data, test_data = obj.initiate_data_ingestion()
```

## Model Performance
Best model selected based on R² score across multiple algorithms with hyperparameter tuning.

## Author
Harshita (starlastyle@gmail.com)
