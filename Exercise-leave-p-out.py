
# Example of LOOCV and Leave-P-Out splitting using the provided dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, LeavePOut

# Configurable constants for Leave-P-Out
P_VAL = 2

# 读取CSV数据
data = pd.read_csv('2.1-Exercise.csv')

# 准备特征和目标变量
X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
y = data['Target'].values

# Our two methods
loocv = LeaveOneOut()
lpocv = LeavePOut(p=P_VAL)

split_loocv = loocv.split(X)
split_lpocv = lpocv.split(X)

def print_result(split_data):
    """
    Prints the result of either a LPOCV or LOOCV operation

    Args:
        split_data: The resulting (train, test) split data
    """
    for train, test in split_data:
        print(f"Train indices: {train}")
        print(f"Test indices: {test}\n")

print("""
The Leave-P-Out method works by using every combination of P points as test data.

The following output shows the result of splitting the data by Leave-One-Out and Leave-P-Out methods.
""")

print("Leave-One-Out:")
print_result(split_loocv)

print("Leave-P-Out (where p = {}):\n".format(P_VAL))
print_result(split_lpocv)
