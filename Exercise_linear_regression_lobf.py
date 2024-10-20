
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the CSV data
data = pd.read_csv('2.2-Exercise.csv')

# Use 'High' as the independent variable and 'Target' as the dependent variable
X = data[['High']].values
y = data['Target'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
regression = LinearRegression()

# Train the model using the training data
regression.fit(X_train, y_train)

# Make predictions using the testing set
y_predictions = regression.predict(X_test)

# Plot the results
sns.set_style("darkgrid")
sns.regplot(x=X_test, y=y_test, fit_reg=False)

# Plot the regression line
plt.plot(X_test, y_predictions, color='black')

# Remove ticks from the plot for a cleaner look
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
