# 🏡 House Price Prediction - Assignment

This project contains the data preprocessing and feature engineering pipeline for the Kaggle competition:  
**[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)**

---

## 📂 Files

- `data_preprocessing.py`: Main Python script to preprocess and engineer features.
- `X_train_processed.csv`: Processed training features (numerical and scaled).
- `X_test_processed.csv`: Processed test features.
- `y_train.csv`: Target values from original training data.

---

## 🔧 Features Implemented

- Handles missing values contextually.
- Adds new features like:
  - `TotalSF` (Total Square Footage)
  - `TotalBath` (Total Bathrooms)
  - `HouseAge`, `RemodelAge`
  - `HasPool`
- One-hot encodes categorical variables.
- Applies feature scaling (StandardScaler).

---

## ▶️ How to Run

1. Place `train.csv` and `test.csv` in the same folder.
2. Run the preprocessing script:

```bash
python data_preprocessing.py
