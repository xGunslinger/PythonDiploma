import pandas as pd
from pandas import DataFrame
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import mean_squared_error

# This is DLM
file_path = 'D:/Documents/Bioinformatics/Final project paper/5yu3.xlsx'
data = pd.read_excel(file_path)
# print(data.head())

print("Dataset has: " + str(len(data)) + " rows.")

# orbits and B factor
X = data.iloc[:, 1:16].values # orbits
Y = data.iloc[:, 0].values # B factor

# data normalization
dl_min_max_scaled_X = X.copy()
# scaler = MinMaxScaler()
scaler = StandardScaler()
dl_min_max_scaled_X = scaler.fit_transform(X)

# create training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(dl_min_max_scaled_X, Y, test_size=0.2, train_size=0.8, random_state=83)

# debug
# print(len(dl_min_max_scaled_X))
# print(dl_min_max_scaled_X)
# print(Y)

# building the model
model = Sequential()
model.add(Dense(15, input_dim=15, activation='relu'))
model.add(Dense(30, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))  # output layer with default linear activation function

def rmse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def mae(y_true, y_pred):
    return K.mean(abs(y_true - y_pred))

# metric accuracy is for classification model not regression as we have at the output
# model.compile(optimizer=Adam(learning_rate=0.001), loss=[rmse])
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True), loss='mean_squared_error', metrics=[rmse])
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)

# evaluation
loss = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")

# predictions using test set
Y_predicted = model.predict(X_test)
num_predicted = int(len(Y_predicted))

# Exp vs predicted
# print("Experimental values:", Y_test)
# print("Predicted values:", Y_predicted)

# Save result into excel file
df = pd.DataFrame(Y, columns=['Y_true'])
df = df.assign(Y_predicted_1=pd.DataFrame(Y_predicted))
final_file = r'C:\Users\ashle\PycharmProjects\diploma\final1.xlsx'
df.to_excel(final_file, index=False)
print('SUCCESS')

# convert pandas frame to list
dl_Y_test_list = Y_test.tolist()

testing = []
for element in range(num_predicted):
    naive = (Y_predicted[element] / dl_Y_test_list[element]) * 100
    testing.append(naive)

print(sum(testing) / len(testing))