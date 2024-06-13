#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <SD.h>
#include <SPI.h>
#include <Bounce2.h>

Adafruit_MPU6050 mpu_l, mpu_r;
File acelFile, errorFile;
Bounce2::Button button = Bounce2::Button();

// Settings
#define BTN_SAMPLE_ACCEL    37         // Button pin
#define LED_R_PIN           13         // Red LED pin

// Constants
#define CONVERT_G_TO_MS2    9.80665f                  // Used to convert G to m/s^2
#define SAMPLING_FREQ_HZ    100                       // Sampling frequency (Hz)
#define SAMPLING_PERIOD_MS  (1000 / SAMPLING_FREQ_HZ) // Sampling period (ms)
#define NUM_SAMPLES         100                       // 100 samples at 100 Hz is 1 sec window

const unsigned long DEBOUNCE_DELAY = 10;  // Debounce time in milliseconds
static bool buttonState = LOW;             // Keep track of the previous state of the button
static bool samplingState = false;         // Keep track of sampling state

const int chipSelect = BUILTIN_SDCARD;

void setup() {
  Serial.begin(115200);
  button.attach(BTN_SAMPLE_ACCEL, INPUT_PULLUP); // Use external pull-up
  button.interval(50);

  // Initialize SD card
  if (!SD.begin(chipSelect)) {
    Serial.println("Initialization Failed!");
    return;
  }

  // Initialize MPU6050 sensors
  Wire.setSCL(19);  // SCL on first i2c bus on T4.1
  Wire.setSDA(18);  // SDA on first i2c bus on T4.1
  Wire1.setSCL(16); // SCL1 on second i2c bus on T4.1
  Wire1.setSDA(17); // SDA1 on second i2c bus on T4.1

  if (!mpu_l.begin(0x68, &Wire, 0) || !mpu_r.begin(0x68, &Wire1, 0)) {
    Serial.println("Failed to find MPU6050 chips");
    while (1) {
      delay(10);
    }
  }

  // Setup motion detection for both sensors
  setupMotionDetection(&mpu_l);
  setupMotionDetection(&mpu_r);

  pinMode(LED_R_PIN, OUTPUT);
  digitalWrite(LED_R_PIN, LOW);
}

void loop() {
  button.update();

  if (button.fell()) { // Button pressed down
    if (!samplingState) { // Start sampling
      startSampling();
    } else { // Stop sampling
      stopSampling();
    }
  }

  delay(100);
}

void setupMotionDetection(Adafruit_MPU6050* mpu) {
  mpu->setHighPassFilter(MPU6050_HIGHPASS_0_63_HZ);
  mpu->setMotionDetectionThreshold(1);
  mpu->setMotionDetectionDuration(20);
  mpu->setInterruptPinLatch(true);
  mpu->setInterruptPinPolarity(true);
  mpu->setMotionInterrupt(true);
}

void startSampling() {
  samplingState = true;
  digitalWrite(LED_R_PIN, HIGH); // Turn on LED to indicate sampling

  const char* acel_csv = "acel_data.csv";
  if (SD.exists(acel_csv)) {
    SD.remove(acel_csv);
  }
  acelFile = SD.open(acel_csv, FILE_WRITE);
  if (acelFile) {
    Serial.println(String(acel_csv) + " open with success");
    errorFile.println(String(acel_csv) + " open with success");
    acelFile.println("timestamp,accX_l,accY_l,accZ_l,gyrX_l,gyrY_l,gyrZ_l,accX_r,accY_r,accZ_r,gyrX_r,gyrY_r,gyrZ_r");
  } else {
    Serial.println("Error opening file " + String(acel_csv));
    errorFile.println("Error opening file " + String(acel_csv));
    stopSampling(); // Stop sampling if file open failed
    return;
  }

  unsigned long start_timestamp = millis();
  unsigned long last_timestamp = start_timestamp;

  while (samplingState) {    
    unsigned long current_timestamp = millis();

    if(((SD.totalSize() - SD.usedSize())/SD.totalSize()) * 100 > 80) {

      digitalWrite(LED_R_PIN, LOW);
      Serial.println("Memoria a 80 de 100");
      errorFile.println("Memoria a 80 de 100");
      return;
    }
    if(current_timestamp - last_timestamp >= SAMPLING_PERIOD_MS) {
      button.update();
      last_timestamp = current_timestamp;

      sensors_event_t a_l, g_l, temp_l;
      mpu_l.getEvent(&a_l, &g_l, &temp_l);
      recordSensorData(&a_l, &g_l, acelFile, start_timestamp, current_timestamp);

      sensors_event_t a_r, g_r, temp_r;
      mpu_r.getEvent(&a_r, &g_r, &temp_r);
      recordSensorData(&a_r, &g_r, acelFile, start_timestamp, current_timestamp);

      acelFile.println();

      if (button.released()) { // Button pressed down again
        stopSampling(); // Stop sampling and break out of loop        
        Serial.println("Stoped Sampling");
        break;
      }
    }
  }

  return;
}

void recordSensorData(sensors_event_t* a, sensors_event_t* g, File& file,
                      unsigned long start_timestamp, unsigned long timestamp) {
  float acc_x = a->acceleration.x * CONVERT_G_TO_MS2;
  float acc_y = a->acceleration.y * CONVERT_G_TO_MS2;
  float acc_z = a->acceleration.z * CONVERT_G_TO_MS2;
  float gyr_x = g->gyro.x;
  float gyr_y = g->gyro.y;
  float gyr_z = g->gyro.z;

  file.print(timestamp - start_timestamp);
  file.print(",");
  file.print(acc_x);
  file.print(",");
  file.print(acc_y);
  file.print(",");
  file.print(acc_z);
  file.print(",");
  file.print(gyr_x);
  file.print(",");
  file.print(gyr_y);
  file.print(",");
  file.print(gyr_z);
  //Serial.println(acc_x);
}

void stopSampling() {
  samplingState = false;
  digitalWrite(LED_R_PIN, LOW); // Turn off LED to indicate sampling stopped
  acelFile.close();
}
