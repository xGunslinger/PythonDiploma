import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# This is DLM
file_path = 'D:/Documents/Bioinformatics/Final project paper/5yu3.csv'
data = pd.read_csv(file_path)

# orbits and B factor
X = data.iloc[:, 2:16].values
Y = data.iloc[:, 0].values

# create training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# data normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert the array back to a dataframe
dataset = DataFrame(data)
# summarize
# print(dataset.describe())

# building the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # 1 neuron

# compilation
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# training
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)

# evaluation
loss = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")

# predictions using test set
Y_pred = model.predict(X_test)

# Real values vs predicted
print("Real values:", Y_test[:5])
print("Predicted values:", Y_pred[:5])