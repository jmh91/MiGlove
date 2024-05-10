import serial
import time
import numpy as np
import matplotlib.pyplot as plt
#import miGlove_functions as miglove
import miGlove_functions_115200 as miglove

# Timer class used to control duration of live data input
class Timer:
    def __init__(self, t):
        self.t = t
        self.start = time.time()

    def still_going(self):
        return time.time() - self.start < self.t

# Moving average filter function
def moving_average(data, window_size):
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return np.concatenate(([np.nan] * (len(data) - len(moving_avg)), moving_avg))

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
arduino = None


# initialize arrays to store data
t = []

arduino = miglove.connect_to_arduino(arduino)



# read data from arduino and store it in arrays
miglove.clear_arduino_input_buffer(arduino)
timer = Timer(10) # 10 second timer started
start_time = time.time()
while timer.still_going():
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
                    t.append(time.time() - start_time)  # Record time stamp
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

                    # Print the individual values (optional)
                    print(f"Index: {len(accelX_values)}, AccelX: {accelX_values[-1]}, AccelY: {accelY_values[-1]}, AccelZ: {accelZ_values[-1]}, "
                        f"GyroX: {gyroX_values[-1]}, GyroY: {gyroY_values[-1]}, GyroZ: {gyroZ_values[-1]}, "
                        f"Finger1: {finger1_values[-1]}, Finger2: {finger2_values[-1]}, Finger3: {finger3_values[-1]}, "
                        f"Finger4: {finger4_values[-1]}, Finger5: {finger5_values[-1]}")
# close serial port
arduino.close()


# Apply moving average filtering to sensor data
window_size = 10
accelX_values_filtered = moving_average(accelX_values, window_size)
accelY_values_filtered = moving_average(accelY_values, window_size)
accelZ_values_filtered = moving_average(accelZ_values, window_size)
gyroX_values_filtered = moving_average(gyroX_values, window_size)
gyroY_values_filtered = moving_average(gyroY_values, window_size)
gyroZ_values_filtered = moving_average(gyroZ_values, window_size)

# Plot the data
fig, axs = plt.subplots(6, 1, figsize=(15, 30), sharex=True)

variables = [
    (accelX_values, accelX_values_filtered, "AccelX"),
    (accelY_values, accelY_values_filtered, "AccelY"),
    (accelZ_values, accelZ_values_filtered, "AccelZ"),
    (gyroX_values, gyroX_values_filtered, "GyroX"),
    (gyroY_values, gyroY_values_filtered, "GyroY"),
    (gyroZ_values, gyroZ_values_filtered, "GyroZ")
] 
for ax, (data, filtered_data, title) in zip(axs, variables):
    ax.plot(t, data, label="Unfiltered")
    ax.plot(t[:len(filtered_data)], filtered_data, label="Filtered")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()


