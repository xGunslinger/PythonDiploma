import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import KFold

# read file with data
# file_path = './5yu3.xlsx'
# file_path = './7vtk.xlsx'
file_path = './4d05.xlsx'
data = pd.read_excel(file_path)
df = pd.DataFrame(data)

# input data (orbits and B factor)
X = data.iloc[:, data.columns != 'BF'].values # orbits
Y = data.iloc[:, 0].values # B factor

# Normalization and standartization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validation parameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# model creation
def create_model():
    model = Sequential([
        Dense(128, input_dim=15, activation='relu'), # layers with neurons
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    # model compiler
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True), loss='mean_squared_error')
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
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
    model.fit(X_train, Y_train, epochs=100, batch_size=20, validation_split=0.2, verbose=0)

    # prediction
    Y_pred = model.predict(X_test).flatten()
    predictions[test_index] = Y_pred

    # print the fold number and model loss for this fold
    print(f'Fold {fold_no} completed. Model loss: {model.evaluate(X_test, Y_test)}')
    fold_no += 1

    # Save our predictions
    Y_pred = model.predict(X_test).flatten()
    predictions[test_index] = Y_pred
    print(type(predictions[test_index]))
#
# Save each prediction into separate excel file
prediction_df = pd.DataFrame({'Actual': Y, 'Predicted_1': predictions})
final_file = r'res32424324234.xlsx'
prediction_df.to_excel(final_file, index=False)
print('All folds completed and predictions are saved.')
