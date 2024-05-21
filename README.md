# MiGlove README
This README provides the necessary steps to install, configure, and run the MiGlove project for gesture recognition, both with and without CoppeliaSim.
Can refer to the 'MiGlove_report.pdf' for further details and development decisions.

## Prerequisites
1. Ensure **Python 3.0** or later is installed.
2. Ensure **Arduino IDE** is installed.

## Installation

1. **Arduino Library Installation**:
   - Install the **Adafruit MPU6050** library using the Arduino Library Manager.

2. **Python Libraries Installation**:
   - To run the Python code, install the following libraries using `pip`:
     ```bash
     pip install pyserial
     pip install scipy
     pip install keyboard
     pip install numpy
     pip install tensorflow
     pip install scikit-learn
     pip install keras
     pip install matplotlib
     ```

3. **Download and Extract Files**:
   - Download the .zip file containing the project and extract the files.

4. **Arduino Setup**:
   - Connect the microcontroller (Arduino) to your device.
   - Upload the `.ino` file found in the Arduino directory to the microcontroller.
   - Ensure the Arduino serial monitor is closed before proceeding.

## Recording Gesture Datasets

1. **Run the Recording Script**:
   - Execute the `combined_sensor_code_v5_115200.py` script.
   - Follow the prompts in the Python terminal to record gesture data for a specified number of iterations.
   - Modify the iteration number as needed for data collection.
   - Change the `gesture_number` variable and the save file paths within the code to match the gesture being recorded.

2. **Combine CSV Files**:
   - Open the `csv_combiner.py` script.
   - Replace the CSV file paths with the newly recorded CSV file paths and run the code to create a new, combined dataset.

3. **Normalize and Train Model**:
   - Open the `main_normalised_115200.py` script.
   - Ensure the loaded CSV file name matches the combined dataset created earlier.
   - Tune the `epoch_number` and `batch_size` variables to optimize model training (batch_size should be no larger than about 10% of the number of gesture samples in the combined dataset).
   - Adjust the `test_size` variable in the `train_test_split` function accordingly (e.g., 0.1 means 90% data for training and 10% for testing).
   - Run the script. It should output a `.h5` model and a `.npz` file containing the normalizing scaler parameters.
   - Continue tuning and rerunning the model until 100% accuracy is achieved on the unseen test data.

## Live Gesture Recognition

### Without CoppeliaSim

1. **Run the Recognition Script**:
   - Open the `live_gesture_recognition_115200.py` script.
   - Ensure the correct `.npz` and `.h5` files from earlier are being loaded.
   - Run the script. When the terminal reads "Getting data...", perform the desired gesture for 10 seconds until the terminal changes.
   - The terminal will print prediction scores and return the most probable gesture label.
   - The program will loop. To cancel, enter `CTRL+C` into the terminal.

### With CoppeliaSim

1. **Install CoppeliaSim**:
   - Ensure CoppeliaSim is installed. The version used here is v4.1.0. Download it from [CoppeliaSim Previous Versions](https://www.coppeliarobotics.com/previousVersions).

2. **Setup CoppeliaSim**:
   - Open CoppeliaSim Application and create a new scene using the file menu.
   - Add a UR10 robot to the scene.
   - Remove the example script attached to the UR10 robot by selecting the paper icon in the hierarchy tree, right-clicking, and selecting remove.

3. **Configure CoppeliaSim for Remote API**:
   - In the CoppeliaSim terminal, type `simRemoteApi.start(19999)`. If using a different port, ensure the port is specified in the Python code.

4. **Check MiGlove Software Configuration**:
   - Ensure that the `.dll` and `sim.py` files exist in the same directory as the Python code to connect Python to CoppeliaSim Remote API.
   - Verify that the `connect_to_simulator` function uses your device's IP address (default is `127.0.0.1` if running CoppeliaSim on the same machine as Python) and that the same port is open in CoppeliaSim.

5. **Run the Recognition Script with CoppeliaSim**:
   - Execute the `live_gesture_recognition_115200_with_robot_sim.py` script.
   - Follow the same process as outlined in the **Without CoppeliaSim** section. The robot will perform a gesture once 10 seconds of gesture capture is complete.

6. **Change Robot Actions**:
   - To change the robot's actions in response to certain gestures, modify the target joint positions in the `perform_` functions within the `live_gesture_recognition_115200_with_robot_sim.py` script.
   - Adjust the joint positions to achieve the desired movements for each gesture.
