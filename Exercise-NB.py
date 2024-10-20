import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn import preprocessing

# Load the dataset
data = pd.read_csv('basketball.csv')

# Convert categorical data to numerical using label encoding
le = preprocessing.LabelEncoder()

data['outlook'] = le.fit_transform(data['outlook'])
data['temp'] = le.fit_transform(data['temp'])
data['humidity'] = le.fit_transform(data['humidity'])
data['windy'] = le.fit_transform(data['windy'])
data['play'] = le.fit_transform(data['play'])

# Prepare features and target variable
X = data[['outlook', 'temp', 'humidity', 'windy']]
y = data['play']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naive Bayes classifier model
nb = CategoricalNB()

# Train the model
nb.fit(X_train, y_train)

# Make predictions
y_pred = nb.predict(X_test)

# Output the results
print(f"Test set actual values: {y_test.values}")
print(f"Test set predicted values: {y_pred}")
