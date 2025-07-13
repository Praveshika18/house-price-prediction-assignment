# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Store SalePrice separately
y = train['SalePrice']

# Combine train and test (excluding SalePrice) for preprocessing
full_data = pd.concat([train.drop(['SalePrice'], axis=1), test], axis=0, ignore_index=True)

# Step 1: Fill missing categorical values with 'None'
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']
for col in none_cols:
    full_data[col] = full_data[col].fillna("None")

# Step 2: Fill missing numerical values with median
median_cols = ['GarageYrBlt', 'MasVnrArea']
for col in median_cols:
    full_data[col] = full_data[col].fillna(full_data[col].median())

# Step 3: Fill LotFrontage using Neighborhood median
full_data['LotFrontage'] = full_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# Drop 'Utilities' due to low variance
full_data.drop(['Utilities'], axis=1, inplace=True)

# Fill remaining numerical missing values with 0
num_cols = full_data.select_dtypes(include=[np.number]).columns
for col in num_cols:
    full_data[col] = full_data[col].fillna(0)

# Step 4: Feature Engineering
full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']
full_data['TotalBath'] = (full_data['FullBath'] + 0.5 * full_data['HalfBath'] +
                          full_data['BsmtFullBath'] + 0.5 * full_data['BsmtHalfBath'])
full_data['HouseAge'] = 2025 - full_data['YearBuilt']
full_data['RemodelAge'] = 2025 - full_data['YearRemodAdd']
full_data['HasPool'] = full_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

# Step 5: One-hot encoding
full_data = pd.get_dummies(full_data)

# Step 6: Scaling
X = full_data.iloc[:len(train), :]
X_test = full_data.iloc[len(train):, :]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data
pd.DataFrame(X_scaled, columns=X.columns).to_csv("X_train_processed.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("X_test_processed.csv", index=False)
y.to_csv("y_train.csv", index=False)

print("✅ Preprocessing complete. Files saved:")
print("- X_train_processed.csv")
print("- X_test_processed.csv")
print("- y_train.csv")
