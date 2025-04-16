import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# This is linear regression
file_path = './5yu3.xlsx'
# file_path = './7vtk.xlsx'
# file_path = './4d05.xlsx'
data = pd.read_excel(file_path)

# orbits and B factor
X = data.iloc[:, data.columns != 'BF'].values
Y = data.iloc[:, 0].values

# Normalization and standartization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# define kfold cross-validation method
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# create linear regression model
def create_model():
    model = LinearRegression()
    return model

# Array for predictions
predictions = np.zeros_like(Y)

# Cross-validation
fold_no = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    model = create_model()

    # Fit data to model
    model.fit(X_train, Y_train)

    # prediction
    Y_pred = model.predict(X_test).flatten()
    predictions[test_index] = Y_pred

    # print the fold number
    print(f'Fold {fold_no} completed.')
    fold_no += 1

    # Save our predictions
    Y_pred = model.predict(X_test).flatten()
    predictions[test_index] = Y_pred


Y_pred = model.predict(X_test).flatten()
predictions[test_index] = Y_pred

# evaluation
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")

# The coefficient of determination R2: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))

# coefficients of the model
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

prediction_df = pd.DataFrame({'Actual': Y, 'Predicted_1': predictions})
print(prediction_df.head())

spearman_corr = prediction_df.corr(method='spearman')

# Display the Spearman correlation matrix
print(spearman_corr)

# # Plot outputs
# plt.figure(figsize=(5,5))
# plt.scatter(Y_test, Y_pred, c='black')
# p1 = max(max(Y_pred), max(Y_test))
# p2 = min(min(Y_pred), min(Y_test))
# plt.plot([p1, p2], [p1, p2], 'b-')
# plt.title("4d05")
# plt.xlabel('True Values', fontsize=15)
# plt.ylabel('Predictions', fontsize=15)
# plt.axis('equal')
# plt. savefig("LR_4d05")
# plt.show()