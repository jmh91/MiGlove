#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

const int numSensors = 5;
const int sensorPins[numSensors] = { A15, A14, A13, A12, A11 };
float angles[numSensors];

// Measure the voltage at 5V and the actual resistance of your
// 47k resistor, and enter them below:
const float VCC = 4.98;       // Measured voltage of Ardunio 5V line
const float R_DIV = 47500.0;  // Measured resistance of 3.3k resistor

// Upload the code, then try to adjust these values to more
// accurately calculate bend degree.
const float STRAIGHT_RESISTANCE = 25000.0;  // resistance when straight
const float BEND_RESISTANCE = 52200.0;      // resistance at 90 deg

void setup(void) {
  // put your setup code here, to run once:

    // begin Serial - wait to initialise
    Serial.begin(115200);
    while (!Serial)
      delay(10);

    // MPU6050 setup
    if (!mpu.begin()) {
      Serial.println("Failed to find MPU6050 chip");
      while (1) {
        delay(10);
      }
    }
    
    Serial.println("MPU6050 Found!");
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);
    Serial.println("");
    // Flex sensor setup
    for (int j = 0; j < numSensors; j++) {
      pinMode(sensorPins[j], INPUT);
    }
    delay(100);
  }
// Either use raw sensor values or convert into angle - return voltage too if needed.
  float flex_read(int flexNum) {
    int sensVal = analogRead(sensorPins[flexNum]);
    float voltage = sensVal * VCC / 1023.0;
    float resistance = R_DIV * (VCC / voltage - 1.0);
    float angle = map(resistance, STRAIGHT_RESISTANCE, BEND_RESISTANCE, 0, 90.0);
    return sensVal;
  }

  void loop() {
    // put your main code here, to run repeatedly:
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    for (int i = 0; i < numSensors; i++) {
      angles[i] = flex_read(i);
    }

    /* Print out the values */
    Serial.print("AccelX:");
    Serial.print(a.acceleration.x);
    Serial.print(",");
    Serial.print("AccelY:");
    Serial.print(a.acceleration.y);
    Serial.print(",");
    Serial.print("AccelZ:");
    Serial.print(a.acceleration.z);
    Serial.print(",");
    Serial.print("GyroX:");
    Serial.print(g.gyro.x);
    Serial.print(",");
    Serial.print("GyroY:");
    Serial.print(g.gyro.y);
    Serial.print(",");
    Serial.print("GyroZ:");
    Serial.print(g.gyro.z);
    Serial.print(",");
    Serial.print("Finger1:");
    Serial.print(angles[0]);
    Serial.print(",");
    Serial.print("Finger2:");
    Serial.print (angles[1]);
    Serial.print(",");
    Serial.print("Finger3:");
    Serial.print(angles[2]);
    Serial.print(",");
    Serial.print("Finger4:");
    Serial.print(angles[3]);
    Serial.print(",");
    Serial.print("Finger5:");
    Serial.print(angles[4]);
    Serial.println("");
    // Delay for Serial Plotter
    delay(50);
  }
