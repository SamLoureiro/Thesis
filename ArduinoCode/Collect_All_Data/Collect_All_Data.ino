#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <SD.h>
#include <SPI.h>
#include <Bounce2.h>
#include <Audio.h>
#include <SerialFlash.h>
#include "setI2SFreq.h"
#include <string.h>


Bounce2::Button Noisebutton = Bounce2::Button();
Bounce2::Button Damagedbutton = Bounce2::Button();
Bounce2::Button Healthybutton = Bounce2::Button();

// GUItool: begin automatically generated code
AudioInputI2S            i2s2;           //xy=105,63
AudioAnalyzePeak         peak1;          //xy=278,108
AudioRecordQueue         queue1;         //xy=281,63
AudioPlaySdRaw           playRaw1;       //xy=302,157
AudioOutputI2S           i2s1;           //xy=470,120
AudioConnection          patchCord1(i2s2, 0, queue1, 0);
AudioConnection          patchCord2(i2s2, 0, peak1, 0);
AudioConnection          patchCord3(playRaw1, 0, i2s1, 0);
AudioConnection          patchCord4(playRaw1, 0, i2s1, 1);
AudioControlSGTL5000     sgtl5000_1;     //xy=265,212


Adafruit_MPU6050 mpu_l, mpu_r;
File acelFile, errorFile, audio_data, wavFile;

// Settings
#define NoisePin            34         // Button pin
#define DamagedPin          35         // Button pin
#define HealthyPin          36         // Button pin
#define BUILTIN_PIN         13         // Built In LED pin
#define ERROR_PIN           33         // Built In LED pin
#define TEMP_PIN            A12        // Temperature Sensor Analog Pin
#define DUST_PIN            27
#define GAS_PIN             41 

// Constants
#define CONVERT_G_TO_MS2    9.80665f                  // Used to convert G to m/s^2
#define SAMPLING_FREQ_HZ    50                        // Sampling frequency (Hz)
#define SAMPLING_PERIOD_MS  (1000 / SAMPLING_FREQ_HZ) // Sampling period (ms)
#define BUFFER_SIZE 100 // Number of lines to accumulate before writing to SD

static bool samplingState = false;         // Keep track of sampling state

const int chipSelect = BUILTIN_SDCARD;

const char* error_txt = "error_data.csv";

const int myInput = AUDIO_INPUT_MIC;

const float referenceVoltage = 3.3;

String csvBuffer = "";

int bufferCount = 0;

int mode = 0;

void recordSensorData(sensors_event_t* a, sensors_event_t* g, sensors_event_t* t, File& file, unsigned long start_timestamp, unsigned long timestamp, bool writetime);
void stopSampling();
void startRecording(const char* wavName);
void continueRecording();
void stopRecording();
void writeWavHeader(File &file, int sampleRate, int bitDepth, int channels, int dataSize);
void updateDataSizeInHeader(File &file);
const char* getUniqueFilename(const char* baseName, const char* extension);
void flushBuffer(File& file);

void setup() {
  Serial.begin(115200);
  Noisebutton.attach(NoisePin, INPUT_PULLUP);
  Noisebutton.interval(50);

  Damagedbutton.attach(DamagedPin, INPUT_PULLUP);
  Damagedbutton.interval(50);

  Healthybutton.attach(HealthyPin, INPUT_PULLUP); 
  Healthybutton.interval(50);

  pinMode(BUILTIN_PIN, OUTPUT);
  digitalWrite(BUILTIN_PIN, LOW);
  pinMode(ERROR_PIN, OUTPUT);
  digitalWrite(ERROR_PIN, LOW);

  AudioMemory(60);
  sgtl5000_1.enable();
  sgtl5000_1.inputSelect(myInput);
  sgtl5000_1.volume(0.5);
  setI2SFreq(192000);

  // Initialize SD card
  while (!SD.begin(chipSelect)) {
    digitalWrite(ERROR_PIN, HIGH);
    Serial.println("Initialization Failed!");
    delay(100);
  }
  digitalWrite(ERROR_PIN, LOW);
  errorFile = SD.open(error_txt, FILE_WRITE);
  if (errorFile) {
    Serial.println(String(error_txt) + " open with success");
    errorFile.print("Time Stamp,");
    errorFile.println("Error Log");
    writeLog(errorFile, millis(), "Log File opened with success");
  } else {
    digitalWrite(ERROR_PIN, HIGH);
    return;
  }
  digitalWrite(ERROR_PIN, LOW);
  //errorFile.print("Time Stamp");
  //errorFile.println(",Error Log");
  // Initialize MPU6050 sensors
  Wire.setSCL(19);  // SCL on first i2c bus on T4.1
  Wire.setSDA(18);  // SDA on first i2c bus on T4.1
  Wire1.setSCL(16); // SCL1 on second i2c bus on T4.1
  Wire1.setSDA(17); // SDA1 on second i2c bus on T4.1

  while(!mpu_l.begin(0x68, &Wire, 0) || !mpu_r.begin(0x68, &Wire1, 0)) {
    digitalWrite(ERROR_PIN, HIGH);
    writeLog(errorFile, millis(), "Failed to find MPU6050 chips");
    Serial.println("Failed to find MPU6050 chips");    
    delay(1000);    
  }
  digitalWrite(ERROR_PIN, LOW);
  writeLog(errorFile, millis(), "MPU6050 chips connected successfully");
  Serial.println("MPU6050 chips connected successfully");
  // Setup motion detection for both sensors
  setupMotionDetection(&mpu_l);
  setupMotionDetection(&mpu_r);

}

void loop() {
  Noisebutton.update();
  Damagedbutton.update();
  Healthybutton.update();

  if (Noisebutton.fell()) { // Button pressed down
    if (!samplingState) { // Start sampling
      startSampling("noise_audio", "noise_accel");
    }
  }
  else if (Damagedbutton.fell()) { // Button pressed down
    if (!samplingState) { // Start sampling
      startSampling("damaged_audio", "damaged_accel");
    }
  }
  else if (Healthybutton.fell()) { // Button pressed down
    if (!samplingState) { // Start sampling
      startSampling("healthy_audio", "healthy_accel");
    }
  }
  delay(10);
}

void setupMotionDetection(Adafruit_MPU6050* mpu) {
  mpu->setHighPassFilter(MPU6050_HIGHPASS_0_63_HZ);
  mpu->setMotionDetectionThreshold(1);
  mpu->setMotionDetectionDuration(20);
  mpu->setInterruptPinLatch(true);
  mpu->setInterruptPinPolarity(true);
  mpu->setMotionInterrupt(true);
}

void startSampling(const char* wavName, const char* csvName) {
  samplingState = true;
  digitalWrite(BUILTIN_PIN, HIGH); // Turn on LED to indicate sampling

  const char* acel_csv = getUniqueFilename(csvName, "csv");
  acelFile = SD.open(acel_csv, FILE_WRITE);
  if (acelFile) {
    Serial.println(String(acel_csv) + " open with success");
    writeLog(errorFile, millis(), ".csv opened with sucess"); 
    acelFile.println("timestamp,accX_l,accY_l,accZ_l,gyrX_l,gyrY_l,gyrZ_l,temp_l,accX_r,accY_r,accZ_r,gyrX_r,gyrY_r,gyrZ_r,temp_r,temp_center,gas,dust");
  } else {
    digitalWrite(ERROR_PIN, HIGH);
    Serial.println("Error opening file " + String(acel_csv));
    writeLog(errorFile, millis(), "Error opening .csv file");   
    stopRecording();
    stopSampling(); // Stop sampling if file open failed    
    return;
  }

  startRecording(wavName);

  unsigned long start_timestamp = millis();
  unsigned long last_timestamp = start_timestamp;

  while (samplingState) {    
    continueRecording();
    unsigned long current_timestamp = millis();

    if(((SD.totalSize() - SD.usedSize())/SD.totalSize()) * 100 > 80) {

      digitalWrite(BUILTIN_PIN, LOW);
      Serial.println("Memoria a 80\%");
      writeLog(errorFile, millis(), "Memoria a 80\%");    
      digitalWrite(ERROR_PIN, HIGH);
      stopRecording();
      stopSampling();
      
      return;
    }
    if(current_timestamp - last_timestamp >= SAMPLING_PERIOD_MS) {      
      Noisebutton.update();
      Damagedbutton.update();
      Healthybutton.update();

      last_timestamp = current_timestamp;

      sensors_event_t a_l, g_l, temp_l;
      mpu_l.getEvent(&a_l, &g_l, &temp_l);
      if(mpu_l.getClock() != 1) {
        Serial.println("Accel Left is Disconnected");
        writeLog(errorFile, millis(), "Accel Left is Disconnected");    
        unsigned long start_error_accel = millis();
        unsigned long current_error_accel = millis();
        while(!mpu_l.begin(0x68, &Wire, 0)) {
          stopRecording();
          stopSampling();
          current_error_accel = millis();
          Serial.println("Failed to find MPU6050_l chip");
          writeLog(errorFile, millis(), "Failed to find MPU6050_l chip");
          digitalWrite(ERROR_PIN, HIGH);
          if(current_error_accel - start_error_accel > 10000) {
            writeLog(errorFile, millis(), "Failed to reconnect MPU6050_l");            
            return;
          }
          delay(100);
        }        
        Serial.println("Accel Left Reconnected");     
        writeLog(errorFile, millis(), "Accel Left Reconnected");
        return;
      }      
      recordSensorData(&a_l, &g_l, &temp_l, acelFile, start_timestamp, current_timestamp, true);

      sensors_event_t a_r, g_r, temp_r;
      mpu_r.getEvent(&a_r, &g_r, &temp_r);
      if(mpu_r.getClock() != 1) {
        Serial.println("Accel Right is Disconnected");
        writeLog(errorFile, millis(), "Accel Right is Disconnected");
        stopRecording();
        stopSampling();       
        unsigned long start_error_accel = millis();
        unsigned long current_error_accel = millis();
        while(!mpu_r.begin(0x68, &Wire1, 0)) {
          current_error_accel = millis();
          Serial.println("Failed to find MPU6050_r chip");
          writeLog(errorFile, millis(), "Failed to find MPU6050_r chip");
          digitalWrite(ERROR_PIN, HIGH);
          if(current_error_accel - start_error_accel > 10000) {
            writeLog(errorFile, millis(), "Failed to reconnect MPU6050_r");  
            return;
          }
          delay(100);
        }    
        Serial.println("Accel Right Reconnected");
        writeLog(errorFile, millis(), "Accel Right Reconnected");
        return;    
      }
      recordSensorData(&a_r, &g_r, &temp_r, acelFile, start_timestamp, current_timestamp, false);

      if (Noisebutton.released() || Damagedbutton.released() || Healthybutton.released()) { // Button pressed down again
        stopRecording(); 
        stopSampling(); // Stop sampling and break out of loop
        writeLog(errorFile, millis(), "Stoped Sampling");                   
        Serial.println("Stoped Sampling");
        break;
      }
    }
  }

  return;
}

void recordSensorData(sensors_event_t* a, sensors_event_t* g, sensors_event_t* t, File& file, unsigned long start_timestamp, unsigned long timestamp, bool writetime) {

  float acc_x = a->acceleration.x * CONVERT_G_TO_MS2;
  float acc_y = a->acceleration.y * CONVERT_G_TO_MS2;
  float acc_z = a->acceleration.z * CONVERT_G_TO_MS2;
  float gyr_x = g->gyro.x;
  float gyr_y = g->gyro.y;
  float gyr_z = g->gyro.z;
  float temp = t->temperature;

  String csv_line = "";

  if (writetime) {
    csv_line += String(timestamp - start_timestamp);
    csv_line += ",";
  }

  csv_line += String(acc_x) + ",";
  csv_line += String(acc_y) + ",";
  csv_line += String(acc_z) + ",";
  csv_line += String(gyr_x) + ",";
  csv_line += String(gyr_y) + ",";
  csv_line += String(gyr_z) + ",";
  csv_line += String(temp);

  if (writetime) {
    csv_line += ",";
  } else {
    float temp_center = Read_Temperature_Sensor(TEMP_PIN, referenceVoltage);
    csv_line += "," + String(temp_center) + "\n";
  }

  // Add the line to the buffer
  csvBuffer += csv_line;
  bufferCount++;

  // Write to SD card if buffer is full
  if (bufferCount >= BUFFER_SIZE) {
    file.print(csvBuffer);
    csvBuffer = "";
    bufferCount = 0;
  }
}


void stopSampling() {
  samplingState = false;  
  // Flush the remaining buffer content to the SD card
  flushBuffer(acelFile);
  acelFile.close();
  errorFile.close();
  digitalWrite(BUILTIN_PIN, LOW); // Turn off LED to indicate sampling stopped
}

void startRecording(const char* wavName) {  

  const char* wav_filename = getUniqueFilename(wavName, "WAV");
  audio_data = SD.open(wav_filename, FILE_WRITE);

  if (audio_data) {
    // Define WAV file parameters
    int sampleRate = 192000; // Change as needed
    int bitDepth = 16;       // Change as needed
    int channels = 1;        // Change as needed
    int dataSize = 0;
    writeWavHeader(audio_data, sampleRate, bitDepth, channels, dataSize);
    queue1.begin();
    Serial.println("startRecording");
    writeLog(errorFile, millis(), "Start recording");
  }
}

void continueRecording() {
  if (queue1.available() >= 2) {
    byte buffer[512];
    // Fetch 2 blocks from the audio library and copy
    // into a 512 byte buffer.  The Arduino SD library
    // is most efficient when full 512 byte sector size
    // writes are used.
    memcpy(buffer, queue1.readBuffer(), 256);
    queue1.freeBuffer();
    memcpy(buffer+256, queue1.readBuffer(), 256);
    queue1.freeBuffer();
    // write all 512 bytes to the SD card
    //elapsedMicros usec = 0;
    audio_data.write(buffer, 512);
    // Uncomment these lines to see how long SD writes
    // are taking.  A pair of audio blocks arrives every
    // 5802 microseconds, so hopefully most of the writes
    // take well under 5802 us.  Some will take more, as
    // the SD library also must write to the FAT tables
    // and the SD card controller manages media erase and
    // wear leveling.  The queue1 object can buffer
    // approximately 301700 us of audio, to allow time
    // for occasional high SD card latency, as long as
    // the average write time is under 5802 us.
    // Serial.print("SD write, us=");
    // Serial.println(usec);
  }
}

void stopRecording() {  
  queue1.end();
  while (queue1.available() > 0) {
    audio_data.write((byte*)queue1.readBuffer(), 256);
    queue1.freeBuffer();
  }

  Serial.println("stopRecording");
  writeLog(errorFile, millis(), "Stop recording");

  updateDataSizeInHeader(audio_data);

}


void writeWavHeader(File &file, int sampleRate, int bitDepth, int channels, int dataSize) {
  int byteRate = sampleRate * channels * (bitDepth / 8);
  int blockAlign = channels * (bitDepth / 8);

  // Write the WAV file header
  file.write("RIFF", 4);
  int chunkSize = 36 + dataSize;
  file.write((uint8_t*)&chunkSize, 4);
  file.write("WAVE", 4);
  file.write("fmt ", 4);
  int subchunk1Size = 16;
  file.write((uint8_t*)&subchunk1Size, 4);
  int16_t audioFormat = 1; // PCM
  file.write((uint8_t*)&audioFormat, 2);
  file.write((uint8_t*)&channels, 2);
  file.write((uint8_t*)&sampleRate, 4);
  file.write((uint8_t*)&byteRate, 4);
  file.write((uint8_t*)&blockAlign, 2);
  file.write((uint8_t*)&bitDepth, 2);
  file.write("data", 4);
  file.write((uint8_t*)&dataSize, 4);
}

void updateDataSizeInHeader(File &file) {

  int dataSize = file.size() - 44; // 44 bytes is the size of the WAV header

  // Update the chunk size
  int chunkSize = 36 + dataSize;
  file.seek(4);
  file.write((uint8_t*)&chunkSize, 4);

  // Update the data size
  file.seek(40);
  file.write((uint8_t*)&dataSize, 4);

  file.close();
}

const char* getUniqueFilename(const char* baseName, const char* extension) {
  static char uniqueName[32];
  int counter = 0;

  while (true) {
    snprintf(uniqueName, sizeof(uniqueName), "%s_%d.%s", baseName, counter, extension);
    if (!SD.exists(uniqueName)) {
      return uniqueName;
    }
    counter++;
  }
}

float Read_Temperature_Sensor(const int pin, const float refvoltage) {

  int sensorValue = analogRead(pin);

  // Convert the analog value to voltage:
  float voltage = sensorValue * (refvoltage / 1023.0);

  // Convert the voltage to temperature in Celsius using the provided formula:
  float temperatureC = 100 * voltage - 50;

  return temperatureC; 
}

void flushBuffer(File& file) {
  if (bufferCount > 0) {
    file.print(csvBuffer);
    csvBuffer = "";
    bufferCount = 0;
  }
}

void writeLog(File& file, const long time, const char* error) {
  file.print(time);
  file.println("," + String(error));
}


/*bool copyFileWithWavHeader(int sampleRate, int bitDepth, int channels) {

  // Open the destination file for writing
  wavFile = SD.open(wav, FILE_WRITE);
  if (!wavFile) {
    Serial.println("Failed to open destination file.");
    digitalWrite(ERROR_PIN, HIGH);
    stopSampling();  
    return false;
  }

  // Calculate the data size and write the WAV header
  audio_data.seek(0);
  int dataSize = audio_data.size();
  writeWavHeader(wavFile, sampleRate, bitDepth, channels, dataSize);

  // Copy data from source to destination
  const size_t bufferSize = 512; // You can adjust the buffer size
  uint8_t buffer[bufferSize];
  size_t bytesRead;

  while ((bytesRead = audio_data.read(buffer, bufferSize)) > 0) {
    wavFile.write(buffer, bytesRead);
  }

  // Close both files
  audio_data.close();
  wavFile.close();

  return true;
}*/