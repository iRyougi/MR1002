# An example of K-Fold Cross-Validation split using the provided dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# 读取CSV数据
data = pd.read_csv('2.1-Exercise.csv')

# 准备特征和目标变量
X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
y = data['Target'].values

# Configurable constants for K-Fold
NUM_SPLITS = 5

# Perform a K-Fold split and print results
kfold = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=42)

print("""
The K-Fold method works by splitting off 'folds' of test data until every point has been used for testing.

The following output shows the result of splitting the data into {} folds.
A bar displaying the current train-test split is displayed for each split.
In the bar, "-" is a training point and "T" is a test point.
""")

for fold, (train_index, test_index) in enumerate(kfold.split(X)):
    print(f"Fold {fold + 1}")
    print(f"Train indices: {train_index}")
    print(f"Test indices: {test_index}\n")

    # If you'd like to see actual data points, uncomment the following lines
    # print("Train data: {}\n".format(X[train_index]))
    # print("Test data: {}\n".format(X[test_index]))
