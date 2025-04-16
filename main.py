import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# This is linear regression
# file_path = './5yu3.xlsx'
# file_path = './7vtk.xlsx'
file_path = './4d05.xlsx'
data = pd.read_excel(file_path)

# orbits and B factor
X = data.iloc[:, data.columns != 'BF']
Y = data.iloc[:, 0]

# create training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# create linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# predictions using test set
Y_pred = model.predict(X_test)

# evaluation
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")

# The coefficient of determination R2: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))

# coefficients of the model
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Plot outputs
plt.scatter(X_test.iloc[:, 1], Y_test, color="black")
plt.plot(X_test.iloc[:, 1], Y_pred, color="blue", linewidth=1)
# plt.show()