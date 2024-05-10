# Import libraries
import serial
import time
import sys
import scipy.io as spio
import keyboard
import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.model_selection import train_test_split
# Functions for MiGlove operation:
import miGlove_functions_115200 as miglove
from datetime import datetime
import sim as vrep

# Timer class used to control duration of live data input
class Timer:
    def __init__(self, t):
        self.t = t
        self.start = time.time()

    def still_going(self):
        return time.time() - self.start < self.t
    
# Function to pad the data arrays
def pad_data_arrays(length):
    final_index = len(accelX_values) - 1
    while len(accelX_values) < length:
        accelX_values.append(accelX_values[final_index])
        accelY_values.append(accelY_values[final_index])
        accelZ_values.append(accelZ_values[final_index])
        gyroX_values.append(gyroX_values[final_index])
        gyroY_values.append(gyroY_values[final_index])
        gyroZ_values.append(gyroZ_values[final_index])
        finger1_values.append(finger1_values[final_index])
        finger2_values.append(finger2_values[final_index])
        finger3_values.append(finger3_values[final_index])
        finger4_values.append(finger4_values[final_index])
        finger5_values.append(finger5_values[final_index])
        iteration_label.append(iteration_label[final_index])

# Setup code
arduino = None

# Initialise empty lists for each sensor data
accelX_values = []
accelY_values = []
accelZ_values = []
gyroX_values = []
gyroY_values = []
gyroZ_values = []
finger1_values = []
finger2_values = []
finger3_values = []
finger4_values = []
finger5_values = []
iteration_label = []
window_length = 150
window_begin  = 0
iteration_count = 0

# Set up an interrupt handler
interrupt_occurred = False  # Flag to indicate interrupt occurrence
recording_started = False # Flag to indicate data recording has started

# Setting up an interrupt handler
def on_interrupt(q):
    global interrupt_occurred
    if recording_started == True:
        print("User input detected...")
        interrupt_occurred = True
        # Close the serial connection
        arduino.close()

def connect_to_simulator():
    # Connect to the remote API server
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

    # Check if the connection was successful
    if clientID != -1:
        print("Connected to remote API server")
    else:
        print("Failed to connect to remote API server")
        sys.exit("Could not connect")
    
    return clientID

def start_simulation(clientID):
    # Start the simulation
    return_code = vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
    if return_code == vrep.simx_return_ok:
        print("Simulation started successfully")
    else:
        print("Failed to start the simulation")
        sys.exit("Error")

def stop_simulation(clientID):
    # Stop the simulation
    return_code = vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    if return_code == vrep.simx_return_ok:
        print("Simulation stopped successfully")
    else:
        print("Failed to stop the simulation")
        sys.exit("Error")

# def set_joint_positions(clientID, joint_handles, target_positions):
#     # Set target positions for each joint
#     for i, joint_handle in enumerate(joint_handles):
#         if i < len(target_positions):
#             return_code = vrep.simxSetJointTargetPosition(clientID, joint_handle, target_positions[i], vrep.simx_opmode_streaming)
#     print("Target positions set successfully")

def set_joint_positions(clientID, joint_handles, target_positions):
    # Set target positions for each joint
    for i, joint_handle in enumerate(joint_handles):
        if i < len(target_positions):
            return_code = vrep.simxSetJointTargetPosition(clientID, joint_handle, target_positions[i], vrep.simx_opmode_streaming)
            # Wait until the joint position has been reached
            while True:
                # Get the current joint position
                _, current_position = vrep.simxGetJointPosition(clientID, joint_handle, vrep.simx_opmode_blocking)
                if abs(current_position - target_positions[i]) < 0.5:  # Adjust tolerance as needed
                    break  # Exit the loop if the joint position is close enough
                time.sleep(0.1)  # Adjust sleep time as needed
    #print("Target positions set successfully")

def set_joint_velocities(clientID, joint_handles, target_velocities):
    # Set target velocities for each joint
    for i, joint_handle in enumerate(joint_handles):
        if i < len(target_velocities):
            return_code = vrep.simxSetJointTargetVelocity(clientID, joint_handle, target_velocities[i], vrep.simx_opmode_streaming)
    print("Target velocities set successfully")

def disconnect_from_simulator(clientID):
    # Wait for the velocities to take effect
    vrep.simxGetPingTime(clientID)
    print("Disconnecting from simulator...")

    # Disconnect from the remote API server
    vrep.simxFinish(clientID)

def perform_hammering():
    # Hammering gesture
    print("Hammering...")
    target_positions = [0, 0, -1, 0, 0, 0]
    set_joint_positions(clientID, joint_handles, target_positions)
    for i in range(4):
        target_positions = [0, 0, -1, -1, 0, 0]
        set_joint_positions(clientID, joint_handles, target_positions)
        target_positions = [0, 0, -1, +1, 0, 0]
        set_joint_positions(clientID, joint_handles, target_positions)
    time.sleep(1)

def perform_sawing():
    # Sawing gesture
    print("Sawing...")
    target_positions = [0, 0, -2, -1, 0, 0]
    set_joint_positions(clientID, joint_handles, target_positions)
    for i in range(4):
        target_positions = [0, -0.5, -1.25, -1, 0, 0]
        set_joint_positions(clientID, joint_handles, target_positions)
        target_positions = [0, 0, -2, -1, 0, 0]
        set_joint_positions(clientID, joint_handles, target_positions)

def perform_screwdriving():
    # Screwdriving gesture
    print("Screwdriving...")
    target_positions = [1.5, 0, -1, -0.35, 0, 0]
    set_joint_positions(clientID, joint_handles, target_positions)
    for i in range(4):
        target_positions = [1.5, 0, -1, -2, 0, 5]
        set_joint_positions(clientID, joint_handles, target_positions)
        target_positions = [1.5, 0,-1, -2, 0, 5]
        set_joint_positions(clientID, joint_handles, target_positions)

def perform_no_gesture():
        # No gesture
    print("No gesture...")
    target_positions = [0, 0, -2, 0, 0, 0]
    set_joint_positions(clientID, joint_handles, target_positions)
    for i in range(4):
        target_positions = [0.5, 0, -2, 0, 0, 0]
        set_joint_positions(clientID, joint_handles, target_positions)
        target_positions = [-0.5, 0, -2, 0, 0, 0]
        set_joint_positions(clientID, joint_handles, target_positions)

def return_to_initial_position():
    # Return to initial position
    print("Returning to initial position...")
    target_positions = [0, 0, 0, 0, 0, 0]
    set_joint_positions(clientID, joint_handles, target_positions)

    # Stop the simulation before disconnecting
    stop_simulation(clientID)
    # Disconnect from simulator
    disconnect_from_simulator(clientID)

# Register the interrupt handler
keyboard.hook(on_interrupt)

# # Import and preprocess data
# num_time_steps = window_length
# num_features = 11 # 6 values from MPU6050 + 5 flex sensors
# num_gestures = 3 # Number of gesture classes
# filename = "combined_gesture_data.csv"
# x_train, y_train = preprocess_import_data(filename, num_time_steps)

# # Reshape the 3D arrays to 2D for normalization
# x_train_flat = x_train.reshape(-1, num_features)
# scaler = MinMaxScaler()
# x_train_normalized = scaler.fit_transform(x_train_flat)
# # Reshape back to 3D
# x_train_normalized = x_train_normalized.reshape(-1, num_time_steps, num_features)

# Load model
model = load_model("gesture_model_115200_2.h5")
# Load the scaling parameters from training sample
scaling_params = np.load("scaling_params_115200.npz")
data_min = scaling_params['data_min']
data_max = scaling_params['data_max']

# Main loop:
print("Starting main loop...")
while(1):          
    # Period for collecting live data input from glove - 5 seconds
    miglove.clear_input_buffer()
    arduino = miglove.connect_to_arduino(arduino)
    # wait_for_input()  # Wait for user input to start recording
    time.sleep(0.75)
    recording_started = True
    timer = Timer(10) # 10 second timer started
    miglove.clear_arduino_input_buffer(arduino)
    while timer.still_going(): # 10 second timer for live data input
        # Read data from Arduino
        data = arduino.readline()
        decoded_data = data.decode('utf-8').strip() #removes leading/trailing whitespaces

        # Check if the data is in the correct format
        if "AccelX" in decoded_data and "AccelY" in decoded_data and "AccelZ" in decoded_data \
                and "GyroX" in decoded_data and "GyroY" in decoded_data and "GyroZ" in decoded_data \
                    and "Finger1" in decoded_data and "Finger2" in decoded_data and "Finger3" in decoded_data \
                        and "Finger4" in decoded_data and "Finger5" in decoded_data:

            # Split the data into individual values
            values = decoded_data.split(',')
            
            # Extract and append values to their respective lists
            accelX_values.append(float(values[0].split(":")[1]))
            accelY_values.append(float(values[1].split(":")[1]))
            accelZ_values.append(float(values[2].split(":")[1]))
            gyroX_values.append(float(values[3].split(":")[1]))
            gyroY_values.append(float(values[4].split(":")[1]))
            gyroZ_values.append(float(values[5].split(":")[1]))
            finger1_values.append(float(values[6].split(":")[1]))
            finger2_values.append(float(values[7].split(":")[1]))
            finger3_values.append(float(values[8].split(":")[1]))
            finger4_values.append(float(values[9].split(":")[1]))
            finger5_values.append(float(values[10].split(":")[1]))
            iteration_label.append(iteration_count)


            # Print the individual values (optional)
            # print(f"Index: {len(accelX_values)}, AccelX: {accelX_values[-1]}, AccelY: {accelY_values[-1]}, AccelZ: {accelZ_values[-1]}, "
            #     f"GyroX: {gyroX_values[-1]}, GyroY: {gyroY_values[-1]}, GyroZ: {gyroZ_values[-1]}, "
            #     f"Finger1: {finger1_values[-1]}, Finger2: {finger2_values[-1]}, Finger3: {finger3_values[-1]}, "
            #     f"Finger4: {finger4_values[-1]}, Finger5: {finger5_values[-1]}, Iteration: {iteration_label[-1]}")

    # Close serial connection        
    arduino.close()

    # Holds last value to pad data arrays if below threshold number of points acquired
    if len(accelX_values) < window_length:
        pad_data_arrays(window_length)
                    # Print the individual values (optional)
    # print(f"Index: {len(accelX_values)}, AccelX: {accelX_values[-1]}, AccelY: {accelY_values[-1]}, AccelZ: {accelZ_values[-1]}, "
    #     f"GyroX: {gyroX_values[-1]}, GyroY: {gyroY_values[-1]}, GyroZ: {gyroZ_values[-1]}, "
    #     f"Finger1: {finger1_values[-1]}, Finger2: {finger2_values[-1]}, Finger3: {finger3_values[-1]}, "
    #     f"Finger4: {finger4_values[-1]}, Finger5: {finger5_values[-1]}, Iteration: {iteration_label[-1]}")
    print(f"Data points captured: {len(accelX_values)}")
    interrupt_occurred = False
    recording_started = False
    iteration_count += 1

    # Convert into numpy data array for model
    live_data_array = np.array([accelX_values, accelY_values, accelZ_values, gyroX_values, gyroY_values, gyroZ_values, finger1_values, finger2_values, finger3_values, finger4_values, finger5_values]).T
    print(live_data_array.shape)
    # Normalise input data
    #live_data_array_flat = live_data_array.reshape(-1, 11)     # Reshape the 3D arrays to 2D for normalization
    #print(live_data_array_flat.shape)
    # Creates matching scaler to that used on training data
    scaler = MinMaxScaler()
    scaler.data_min_ = data_min
    scaler.data_max_ = data_max
    live_data_array_normalised = scaler.fit_transform(live_data_array) # Normalise
    print(live_data_array_normalised.shape)
    #x_test = live_data_array_normalised.reshape(-1, len(accelX_values), 11)     # Reshape back to 3D
    #print(x_test.shape)

    # Period for running live data sequence through model for classification - ~5 seconds

    # Make predictions on the test set

    # Gestures:
    # 1: Hammering
    # 2: Sawing
    # 3: Screwdriving
    # 4: Inactive

    # Loop through the windows of data and make predictions
    num_preds = [0, 0, 0, 0] # Initialize the number of predictions for each class
    for window_start in range(window_begin, len(accelX_values) - window_length + 1):
        current_window = live_data_array_normalised[window_start:window_start + window_length]
        # Reshape the current window to 3D
        current_window_reshaped = current_window.reshape(-1, window_length, 11)
        # Make a prediction on the current window
        prediction = model.predict(current_window_reshaped)
        print(prediction)
        # Update the number of predictions for each class
        for i in range(4):
            num_preds[i] += np.count_nonzero(np.argmax(prediction, axis=1) == i)

    print("Hammering guesses:", num_preds[0])
    print("Sawing guesses:", num_preds[1])
    print("Screwdriving guesses:", num_preds[2])
    print("No gesture guesses:", num_preds[3])
        # Print the prediction result
        # print(f"From index {window_start} to index {window_start + window_length - 1}, Predicted gesture: {np.argmax(prediction)}")
    window_begin = len(accelX_values)


    # UR10 robot simulation
    clientID = connect_to_simulator()
    start_simulation(clientID)

    # Get handles for the UR10 joints
    joint_handles = []
    for i in range(1, 7):
        return_code, joint_handle = vrep.simxGetObjectHandle(clientID, 'UR10_joint' + str(i), vrep.simx_opmode_blocking)
        if return_code == vrep.simx_return_ok:
            joint_handles.append(joint_handle)
        else:
            print("Failed to get handle for UR10_joint" + str(i))
            sys.exit("Error")

    # # The predicted gesture is the class with the most guesses - winner-takes-all strategy
    predicted_class = np.argmax(num_preds)


    # Robot mimics predicted gesture
    match predicted_class:
        case 0:

            predicted_gesture = "Hammering"
            perform_hammering()
        case 1:
            predicted_gesture = "Sawing"
            perform_sawing()
        case 2:
            predicted_gesture = "Screwdriving"
            perform_screwdriving()
        case 3:
            predicted_gesture = "No Gesture"
            perform_no_gesture()

    return_to_initial_position()
    
    # print(f"{datetime.now().strftime('%H:%M:%S.%f')}: Predicted gesture: {predicted_gesture}") # Prints predicted gesture with timestamp


    # End of loop

