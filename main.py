import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# This is linear regression
file_path = 'D:/Documents/Bioinformatics/Final project paper/5yu3.csv'
data = pd.read_csv(file_path)
# print(data.head())

# orbits and B factor
X = data.iloc[:, 2:16]
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

# coefficients of the model
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)



