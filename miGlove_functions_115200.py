import serial
import time
import sys
import scipy.io as spio
import keyboard
import csv
import numpy as np

# Empties input buffer for controlled user interrupts
def clear_input_buffer():
    try:
        # For Python 3 (Windows)
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        # For Linux/Unix-based systems (For Raspberry Pi)
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)

def connect_to_arduino(arduino):
    if arduino is not None:
        arduino.close() # Closes any existing connection
    time.sleep(1)    
    # Declare a serial connection to the Arduino
    arduino = serial.Serial('COM3', 115200)  # Use the appropriate COM port
    # Wait for the connection to be established
    time.sleep(2)

    arduino.write(b'Hello, Arduino!')
    data = arduino.readline()
    print(f"Arduino says: {data.decode('utf-8')}")
    return arduino

# Function to clear any data in the Arduino input buffer before Enter key is pressed
def clear_arduino_input_buffer(arduino):
    while arduino.in_waiting:
        arduino.readline()

# Function to wait for user input to start recording
def wait_for_input(arduino):
    input("Press Enter to start recording gesture data...")
    print("Recording data now...")
    # Empty input buffer to remove unwanted sensor data accumulated during standby.
    clear_arduino_input_buffer(arduino)
    