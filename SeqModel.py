import pandas as pd
import numpy as np
from keras import backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

file_path = 'D:/Documents/Bioinformatics/Final project paper/5yu3.xlsx'
data = pd.read_excel(file_path)

# orbits and B factor
X = data.iloc[:, 1:16].values # orbits
Y = data.iloc[:, 0].values # B factor

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Normalization
scaler = MinMaxScaler()
# scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validation parameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# RMSE loss function
def rmse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# def mae(y_true, y_pred):
    #     return K.mean(abs(y_true - y_pred))

def create_model():
    model = Sequential([
        Dense(15, input_dim=15, activation='relu'),
        Dense(30, activation='relu'),
        Dense(1, activation='linear')
    ])

    # metric accuracy is for classification model not regression as we have at the output
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True), loss='mean_squared_error', metrics=[rmse])
    # model.compile(optimizer=Adam(learning_rate=0.001), loss=[rmse])
    return model

# Array for predictions
predictions = np.zeros_like(Y)

# Callback to stop before
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Cross-validation
fold_no = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    model = create_model()

    # Fit data to model
    model.fit(X_train, Y_train, epochs=100, batch_size=20, validation_split=0.2, verbose=0)

    Y_pred = model.predict(X_test).flatten()
    predictions[test_index] = Y_pred

    print(f'Fold {fold_no} completed. Model loss: {model.evaluate(X_test, Y_test)}')
    fold_no += 1

    # Save our predictions
    Y_pred = model.predict(X_test).flatten()
    predictions[test_index] = Y_pred

    print(f'Fold {fold_no} completed. Model loss: {model.evaluate(X_test, Y_test)}')
    fold_no += 1

# Save into excel file
prediction_df = pd.DataFrame({'Actual': Y, 'Predicted': predictions})
final_file = r'C:\Users\ashle\PycharmProjects\diploma\final1.xlsx'
prediction_df.to_excel(final_file, index=False)
print('All folds completed and predictions are saved.')
