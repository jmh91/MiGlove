import serial
import time
import sys
import scipy.io as spio
import keyboard
import csv
import numpy as np 
import miGlove_functions_115200 as miglove

arduino = None

# Gestures:
# 1: Hammering
# 2: Sawing
# 3: Screwdriving
# 4: Inactive

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
gesture_label = []
iteration_label = []
gesture_number = 1 # Change to gesture UID
data_per_iteration = 150 # more data for higher baud rate
max_iterations = 30
max_data = data_per_iteration

# Initialise file save paths
file_save_path_csv = 'gesture_data_hammering_115200.csv' # .csv file path
file_save_path_mat = 'gesture_data_hammering_115200.mat' # .mat file path

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

# Register the interrupt handler
keyboard.hook(on_interrupt)

# Save data to .csv file
def save_to_csv():
    with open(file_save_path_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ", "Finger1", "Finger2", "Finger3", "Finger4", "Finger5","Iteration", "Gesture"])
        for i in range(len(accelX_values)):
            writer.writerow([i, accelX_values[i], accelY_values[i], accelZ_values[i], gyroX_values[i], gyroY_values[i],
                             gyroZ_values[i], finger1_values[i], finger2_values[i], finger3_values[i],
                             finger4_values[i], finger5_values[i], iteration_label[i], gesture_label[i]]) # change gesture_number depending on gesture UID

# Save data to .mat file
def save_to_mat():
    spio.savemat(file_save_path_mat, {'Index': np.arange(len(accelX_values)),
                              'AccelX': accelX_values,
                              'AccelY': accelY_values,
                              'AccelZ': accelZ_values,
                              'GyroX': gyroX_values,
                              'GyroY': gyroY_values,
                              'GyroZ': gyroZ_values,
                              'Finger1': finger1_values,
                              'Finger2': finger2_values,
                              'Finger3': finger3_values,
                              'Finger4': finger4_values,
                              'Finger5': finger5_values,
                              'Iteration': iteration_label,
                              'Gesture': gesture_label}) # change label number for gesture UID

# Function to pad the data arrays
def pad_data_arrays():
    final_index = len(accelX_values) - 1
    while len(accelX_values) < max_data:
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
        gesture_label.append(gesture_label[final_index])

# MAIN CODE
miglove.clear_input_buffer()
j = 1
while j <= 1: # deprecated - changed to only record one unique gesture per csv
    k = 1
    while k <= max_iterations:
        arduino = miglove.connect_to_arduino(arduino)
        try:
            miglove.wait_for_input(arduino)  # Wait for user input to start recording
            time.sleep(0.5)
            recording_started = True

            while not interrupt_occurred:
                # Read data from Arduino
                data = arduino.readline()
                # Break out of data recording if either interrupt occurs or maximum data capture is achieved
                if interrupt_occurred or len(accelX_values) >=  max_data:
                    break

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
                    gesture_label.append(gesture_number)
                    iteration_label.append(k)

                    # Print the individual values (optional)
                    print(f"Index: {len(accelX_values)}, AccelX: {accelX_values[-1]}, AccelY: {accelY_values[-1]}, AccelZ: {accelZ_values[-1]}, "
                        f"GyroX: {gyroX_values[-1]}, GyroY: {gyroY_values[-1]}, GyroZ: {gyroZ_values[-1]}, "
                        f"Finger1: {finger1_values[-1]}, Finger2: {finger2_values[-1]}, Finger3: {finger3_values[-1]}, "
                        f"Finger4: {finger4_values[-1]}, Finger5: {finger5_values[-1]}, Iteration: {k}, Gesture: {gesture_number}")
                    
        except KeyboardInterrupt:
            pass  # Catch the keyboard interrupt to gracefully exit the loop

        finally:

            # Holds last value to pad data arrays if below threshold number of points acquired
            if len(accelX_values) < max_data:
                pad_data_arrays()
                            # Print the individual values (optional)
            print(f"Index: {len(accelX_values)}, AccelX: {accelX_values[-1]}, AccelY: {accelY_values[-1]}, AccelZ: {accelZ_values[-1]}, "
                f"GyroX: {gyroX_values[-1]}, GyroY: {gyroY_values[-1]}, GyroZ: {gyroZ_values[-1]}, "
                f"Finger1: {finger1_values[-1]}, Finger2: {finger2_values[-1]}, Finger3: {finger3_values[-1]}, "
                f"Finger4: {finger4_values[-1]}, Finger5: {finger5_values[-1]}, Iteration: {k}, Gesture: {gesture_number}")
            print(f"Data points captured: {len(accelX_values)}")
            interrupt_occurred = False
            recording_started = False
        # Saves data (overwrites with every iteration)   
        save_to_csv()
        save_to_mat()
        print("Data has been saved.")    
        max_data += data_per_iteration
        k+=1
    j+=1

# Close the serial connection if not closed already
if arduino.is_open:
    arduino.close()
        