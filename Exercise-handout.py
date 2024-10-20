# An example of the Holdout Cross-Validation split using the provided dataset

import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV数据
data = pd.read_csv('2.1-Exercise.csv')

# 准备特征和目标变量
X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
y = data['Target'].values

# 将数据集拆分为训练集和测试集，80%训练，20%测试
TRAIN_SPLIT = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SPLIT, test_size=1-TRAIN_SPLIT, random_state=42)

print("""
The holdout method removes a certain portion of the training data and uses it as test data.
Ideally, the data points removed are random on each run.

The following output shows the dataset split into test and training data:
""")

# Print total data points, training, and test data points
print("Total data points: {}".format(len(data)))
print("# of training data points: {} (~{}%)".format(len(X_train), TRAIN_SPLIT*100))
print("# of test data points: {} (~{}%)\n".format(len(X_test), (1-TRAIN_SPLIT)*100))

# Uncomment the lines below to print actual data
# print("Training data:\n{}\n".format(X_train))
# print("Test data:\n{}".format(X_test))
