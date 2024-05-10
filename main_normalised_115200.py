import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential, layers, models, optimizers
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Get N (1 x 50 x 14) arrays of data
def preprocess_import_data(filename, num_time_steps):
    # Load data from CSV
    gesture_data = np.loadtxt(filename, delimiter=",", skiprows=1)
    
    # Restructure data for LSTM
    num_samples = gesture_data.shape[0] // num_time_steps
    num_columns = gesture_data.shape[1]
    reshaped_data = np.reshape(gesture_data, (num_samples, num_time_steps, num_columns))
    
    # Extract x and y_labels
    x = reshaped_data[:,:,1:12]  # Remove index, iteration, and gesture ID
    y_labels = reshaped_data[:,:,-1][:,-1] - 1  # Subtract 1 to normalize to 0-2
    
    return x, y_labels

# Moving average filter function
def moving_average(data, window_size):
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return np.concatenate(([np.nan] * (len(data) - len(moving_avg)), moving_avg))

num_time_steps = 150
num_features = 11 # 6 values from MPU6050 + 5 flex sensors
num_gestures = 4 # Number of gesture classes
filename = "combined_gesture_data_115200.csv" # change to new file name
x, y_labels = preprocess_import_data(filename, num_time_steps)

# The stratify parameter will ensure that the train and test split has the same class distribution ratio as the original dataset. 
# It is crucial in the case of imbalanced datasets. Otherwise, it might happen that the training data only consists of the majority class.
x_train, x_test, y_train, y_test = train_test_split(x, y_labels, test_size=0.1, stratify=y_labels, random_state=42)


# Reshape the 3D arrays to 2D for normalization
x_train_flat = x_train.reshape(-1, num_features)
x_test_flat = x_test.reshape(-1, num_features)

scaler = MinMaxScaler()
x_train_normalized = scaler.fit_transform(x_train_flat)
x_test_normalized = scaler.transform(x_test_flat)

# Reshape back to 3D
x_train_normalized = x_train_normalized.reshape(-1, num_time_steps, num_features)
x_test_normalized = x_test_normalized.reshape(-1, num_time_steps, num_features)

# Access the scaling factors to use in live data update.
data_min = scaler.data_min_
data_max = scaler.data_max_

print("Minimum values for each feature:", data_min)
print("Maximum values for each feature:", data_max)

# Save the scaling parameters to use in the live data prediction.
np.savez("scaling_params_115200.npz", data_min=scaler.data_min_, data_max=scaler.data_max_)

model = Sequential()
model.add(LSTM(units=num_time_steps, input_shape=(num_time_steps, num_features)))
#model.add(Dropout(0.2))
model.add(Dense(num_gestures, activation="softmax")) # No. of classes

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=250, verbose=1, min_lr = 0.00001)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train_normalized, y_train, epochs=1000, batch_size=32, validation_data=(x_test_normalized, y_test), callbacks=[reduce_lr, early_stop])

# Make predictions on the test set
predictions = model.predict(x_test_normalized)

# Since predictions are probabilities, get the class with the highest probability
predicted_classes = np.argmax(predictions, axis=1)

# Now compare the predicted classes with the actual classes (y_test)
for idx in range(len(x_test)):
    print(f'Predicted gesture: {predicted_classes[idx]}, Actual gesture: {y_test[idx]}')

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test_normalized, y_test)
print("Test loss: ", loss)
print("Test accuracy: ", accuracy)

model.save("gesture_model_115200_2.h5")