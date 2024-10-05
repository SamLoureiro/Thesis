#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

// Settings
#define BTN_PIN             37         // Button pin
#define LED_R_PIN           13        // Red LED pin

// Constants
#define CONVERT_G_TO_MS2    9.80665f  // Used to convert G to m/s^2
#define SAMPLING_FREQ_HZ    100       // Sampling frequency (Hz)
#define SAMPLING_PERIOD_MS  1000 / SAMPLING_FREQ_HZ   // Sampling period (ms)
#define NUM_SAMPLES         100       // 100 samples at 100 Hz is 1 sec window

void setup() {

  // Enable button pin
  pinMode(BTN_PIN, INPUT_PULLUP);

  // Enable LED pin (RGB LEDs are active low)
  pinMode(LED_R_PIN, OUTPUT);
  digitalWrite(LED_R_PIN, HIGH);

  // Start serial
  Serial.begin(115200);

  // Try to initialize!
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  //setupt motion detection
  mpu.setHighPassFilter(MPU6050_HIGHPASS_0_63_HZ);
  mpu.setMotionDetectionThreshold(1);
  mpu.setMotionDetectionDuration(20);
  mpu.setInterruptPinLatch(true);	// Keep it latched.  Will turn off when reinitialized.
  mpu.setInterruptPinPolarity(true);
  mpu.setMotionInterrupt(true);

  Serial.println("");
  delay(100);
}

void loop() {

  float acc_x;
  float acc_y;
  float acc_z;
  float gyr_x;
  float gyr_y;
  float gyr_z;
  unsigned long timestamp;
  unsigned long start_timestamp;

  // Wait for button press
  while (digitalRead(BTN_PIN) == 1);

  // Turn on LED to show we're recording
  digitalWrite(LED_R_PIN, LOW);

  // Print header
  //Serial.println("timestamp,accX,accY,accZ,gyrX,gyrY,gyrZ");

  // Record samples in buffer
  start_timestamp = millis();
  for (int i = 0; i < NUM_SAMPLES; i++) {
    

    // Take timestamp so we can hit our target frequency
    timestamp = millis();
    
    // Read and convert accelerometer data to m/s^2
    if(/*mpu.getMotionInterruptStatus()*/1) {
      /* Get new sensor events with the readings */
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);
      acc_x = a.acceleration.x;
      acc_y = a.acceleration.y;
      acc_z = a.acceleration.z;
      gyr_x = g.gyro.x;
      gyr_y = g.gyro.y;
      gyr_z = g.gyro.z;
      acc_x *= CONVERT_G_TO_MS2;
      acc_y *= CONVERT_G_TO_MS2;
      acc_z *= CONVERT_G_TO_MS2;


      // Print CSV data with timestamp
      Serial.print(timestamp - start_timestamp);
      Serial.print(",");
      Serial.print(acc_x);
      Serial.print(",");
      Serial.print(acc_y);
      Serial.print(",");
      Serial.print(acc_z);
      Serial.print(",");
      Serial.print(gyr_x);
      Serial.print(",");
      Serial.print(gyr_y);
      Serial.print(",");
      Serial.println(gyr_z);

      // Wait just long enough for our sampling period
      while (millis() < timestamp + SAMPLING_PERIOD_MS);
  }

  }

  // Print empty line to transmit termination of recording
  Serial.println();

  // Turn off LED to show we're done
  digitalWrite(LED_R_PIN, HIGH);

  // Make sure the button has been released for a few milliseconds
  while (digitalRead(BTN_PIN) == 0);
  delay(100);
}
