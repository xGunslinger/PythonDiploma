import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD

# here the best performed model for each protein is taken,
# trained on the data it was performed the best
# and tested on the two other datasets
# read file with data
file_path = './5yu3.xlsx'
# file_path_2 = './7vtk.xlsx'
file_path_3 = './4d05.xlsx'

# first file
data = pd.read_excel(file_path)
df = pd.DataFrame(data)

# input data (orbits and B factor)
X = data.iloc[:, data.columns != 'BF'].values # orbits
Y = data.iloc[:, 0].values # B factor

# second file
# data2 = pd.read_excel(file_path_2)
# df2 = pd.DataFrame(data2)
# X2 = data2.iloc[:, data2.columns != 'BF'].values # orbits
# Y2 = data2.iloc[:, 0].values # B factor

# third file
data3 = pd.read_excel(file_path_3)
df3 = pd.DataFrame(data3)
X3 = data3.iloc[:, data3.columns != 'BF'].values # orbits
Y3 = data3.iloc[:, 0].values # B factor

# create training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X3, Y3, test_size=0.2, random_state=42)

# Normalization and standartization
scaler = StandardScaler()

# building a model
model = Sequential([
        Dense(128, activation='relu'), # layers with neurons
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True), loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=100, batch_size=20, validation_split=0.2, verbose=0)

# predictions for the second protein
Y_pred = model.predict(X).flatten()
predictions = Y_pred.flatten()

# evaluation for the second protein
mse = mean_squared_error(Y, Y_pred)
print(f"Mean Squared Error: {mse}")

# # Save each prediction into separate excel file
prediction_df = pd.DataFrame({'Actual': Y, 'Predicted': predictions})
final_file = r'P3-P1.xlsx'
prediction_df.to_excel(final_file, index=False)
print('All folds completed and predictions are saved.')
