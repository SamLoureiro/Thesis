#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>


Adafruit_MPU6050 mpu_l, mpu_r;

// Settings
#define BTN_PIN             37         // Button pin
#define LED_R_PIN           13        // Red LED pin

// Constants
#define CONVERT_G_TO_MS2    9.80665f  // Used to convert G to m/s^2
#define SAMPLING_FREQ_HZ    100       // Sampling frequency (Hz)
#define SAMPLING_PERIOD_MS  1000 / SAMPLING_FREQ_HZ   // Sampling period (ms)
#define NUM_SAMPLES         100       // 100 samples at 100 Hz is 1 sec window

void setup() {

  // Start serial
  Serial.begin(115200);

  // Enable button pin
  pinMode(BTN_PIN, INPUT_PULLUP);

  // Enable LED pin (RGB LEDs are active low)
  pinMode(LED_R_PIN, OUTPUT);
  digitalWrite(LED_R_PIN, HIGH);

  Wire.setSCL(19);  // SCL on first i2c bus on T4.1
  Wire.setSDA(18);  // SDA on first i2c bus on T4.1
  Wire1.setSCL(16); // SCL1 on second i2c bus on T4.1
  Wire1.setSDA(17); // SDA1 on second i2c bus on T4.1

  if (!mpu_l.begin(0x68, &Wire, 0)) {
    Serial.println("Failed to find MPU6050_l chip");
    while (1) {
      delay(10);
    }
  }
  if (!mpu_r.begin(0x68, &Wire1, 0)) {
    Serial.println("Failed to find MPU6050_r chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("2 MPU6050 Found!");

    //setupt motion detection
  mpu_l.setHighPassFilter(MPU6050_HIGHPASS_0_63_HZ);
  mpu_l.setMotionDetectionThreshold(1);
  mpu_l.setMotionDetectionDuration(20);
  mpu_l.setInterruptPinLatch(true);	// Keep it latched.  Will turn off when reinitialized.
  mpu_l.setInterruptPinPolarity(true);
  mpu_l.setMotionInterrupt(true);

  mpu_r.setHighPassFilter(MPU6050_HIGHPASS_0_63_HZ);
  mpu_r.setMotionDetectionThreshold(1);
  mpu_r.setMotionDetectionDuration(20);
  mpu_r.setInterruptPinLatch(true);	// Keep it latched.  Will turn off when reinitialized.
  mpu_r.setInterruptPinPolarity(true);
  mpu_r.setMotionInterrupt(true);

  Serial.println("");



}

void loop() {

  float acc_x_l, acc_x_r;
  float acc_y_l, acc_y_r;
  float acc_z_l, acc_z_r;;
  float gyr_x_l, gyr_x_r;
  float gyr_y_l, gyr_y_r;
  float gyr_z_l, gyr_z_r;
  unsigned long timestamp;
  unsigned long start_timestamp;

  // Wait for button press
  while (digitalRead(BTN_PIN) == 1);

  // Turn on LED to show we're recording
  digitalWrite(LED_R_PIN, LOW);

  // Print header
  Serial.println("timestamp,accX_l,accY_l,accZ_l,gyrX_l,gyrY_l,gyrZ_l,accX_r,accY_r,accZ_r,gyrX_r,gyrY_r,gyrZ_r");

  // Record samples in buffer
  start_timestamp = millis();
  for (int i = 0; i < NUM_SAMPLES; i++) {
    
    // Take timestamp so we can hit our target frequency
    timestamp = millis();
    
    // Read and convert accelerometer data to m/s^2
  
    /* Get new sensor events with the readings */
    sensors_event_t a_l, g_l, temp_l;   
    mpu_r.getEvent(&a_l, &g_l, &temp_l);
    if(mpu_l.getClock() != 1) {
      Serial.println("Accel Left is Disconnected");
      while(!mpu_l.begin(0x68, &Wire, 0)) {
        Serial.println("Failed to find MPU6050_l chip");
        delay(500);
      }        
      Serial.println("Accel Left Reconnected");        
    }
    
    sensors_event_t a_r, g_r, temp_r;
    mpu_r.getEvent(&a_r, &g_r, &temp_r);
    if(mpu_r.getClock() != 1) {
      Serial.println("Accel Right is Disconnected");
      while(!mpu_r.begin(0x68, &Wire1, 0)) {
        Serial.println("Failed to find MPU6050_r chip");
        delay(500);
      }        
      Serial.println("Accel Right Reconnected");        
    }

    // 7 - Desconectou
    // 0 - Reconectou
    // 1 - Está nice


    acc_x_l = a_l.acceleration.x;
    acc_y_l = a_l.acceleration.y;
    acc_z_l = a_l.acceleration.z;
    gyr_x_l = g_l.gyro.x;
    gyr_y_l = g_l.gyro.y;
    gyr_z_l = g_l.gyro.z;
    acc_x_l *= CONVERT_G_TO_MS2;
    acc_y_l *= CONVERT_G_TO_MS2;
    acc_z_l *= CONVERT_G_TO_MS2;

    Serial.println();
    Serial.println("Teste Direito");
    Serial.println(mpu_r.getClock());
    Serial.println();
    acc_x_r = a_r.acceleration.x;
    acc_y_r = a_r.acceleration.y;
    acc_z_r = a_r.acceleration.z;
    gyr_x_r = g_r.gyro.x;
    gyr_y_r = g_r.gyro.y;
    gyr_z_r = g_r.gyro.z;
    acc_x_r *= CONVERT_G_TO_MS2;
    acc_y_r *= CONVERT_G_TO_MS2;
    acc_z_r *= CONVERT_G_TO_MS2;


    // Print CSV data with timestamp
    Serial.print(timestamp - start_timestamp);
    Serial.print(",");
    Serial.print(acc_x_l);
    Serial.print(",");
    Serial.print(acc_y_l);
    Serial.print(",");
    Serial.print(acc_z_l);
    Serial.print(",");
    Serial.print(gyr_x_l);
    Serial.print(",");
    Serial.print(gyr_y_l);
    Serial.print(",");
    Serial.print(gyr_z_l);
    Serial.print(",");
    Serial.print(acc_x_r);
    Serial.print(",");
    Serial.print(acc_y_r);
    Serial.print(",");
    Serial.print(acc_z_r);
    Serial.print(",");
    Serial.print(gyr_x_r);
    Serial.print(",");
    Serial.print(gyr_y_r);
    Serial.print(",");
    Serial.println(gyr_z_r);


    // Take timestamp so we can hit our target frequency
    timestamp = millis();
    

    // Wait just long enough for our sampling period
    while (millis() < timestamp + SAMPLING_PERIOD_MS);
  

  }

  // Print empty line to transmit termination of recording
  Serial.println();

  // Turn off LED to show we're done
  digitalWrite(LED_R_PIN, HIGH);

  // Make sure the button has been released for a few milliseconds
  while (digitalRead(BTN_PIN) == 0);
  delay(100);
}
