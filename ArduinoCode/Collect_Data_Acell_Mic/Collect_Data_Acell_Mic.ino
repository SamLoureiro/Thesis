#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <SD.h>
#include <SPI.h>
#include <Bounce2.h>
#include <Audio.h>
#include <SerialFlash.h>
#include "setI2SFreq.h"

Bounce2::Button button = Bounce2::Button();

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
#define BTN_SAMPLE_ACCEL    34         // Button pin
#define BUILTIN_PIN         13         // Built In LED pin
#define ERROR_PIN           33         // Built In LED pin

// Constants
#define CONVERT_G_TO_MS2    9.80665f                  // Used to convert G to m/s^2
#define SAMPLING_FREQ_HZ    100                       // Sampling frequency (Hz)
#define SAMPLING_PERIOD_MS  (1000 / SAMPLING_FREQ_HZ) // Sampling period (ms)
#define NUM_SAMPLES         100                       // 100 samples at 100 Hz is 1 sec window

static bool samplingState = false;         // Keep track of sampling state

const int chipSelect = BUILTIN_SDCARD;

const char* error_txt = "error_data.csv";

const char* wav = "record.wav";

const int myInput = AUDIO_INPUT_MIC;

int mode = 0;

void setup() {
  Serial.begin(115200);
  button.attach(BTN_SAMPLE_ACCEL, INPUT_PULLUP); // Use external pull-up
  button.interval(50);

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
  if (!SD.begin(chipSelect)) {
    digitalWrite(ERROR_PIN, HIGH);
    Serial.println("Initialization Failed!");
    return;
  }

  errorFile = SD.open(error_txt, FILE_WRITE);
  if (errorFile) {
    Serial.println(String(error_txt) + " open with success");
    errorFile.print("Time Stamp,");
    errorFile.println("Error Log");
    errorFile.print(millis());
    errorFile.println("," + String(error_txt) + " open with success");
  } else {
    digitalWrite(ERROR_PIN, HIGH);
    return;
  }
  digitalWrite(ERROR_PIN, LOW);
  errorFile.print("Time Stamp");
  errorFile.println("Error Log");
  // Initialize MPU6050 sensors
  Wire.setSCL(19);  // SCL on first i2c bus on T4.1
  Wire.setSDA(18);  // SDA on first i2c bus on T4.1
  Wire1.setSCL(16); // SCL1 on second i2c bus on T4.1
  Wire1.setSDA(17); // SDA1 on second i2c bus on T4.1

  while(!mpu_l.begin(0x68, &Wire, 0) || !mpu_r.begin(0x68, &Wire1, 0)) {
    digitalWrite(ERROR_PIN, HIGH);
    errorFile.print(millis());
    errorFile.println("Failed to find MPU6050 chips");  
    Serial.println("Failed to find MPU6050 chips");    
    delay(1000);    
  }
  digitalWrite(ERROR_PIN, LOW);
  errorFile.print(millis());
  errorFile.println("MPU6050 chips connected successfully"); 
  Serial.println("MPU6050 chips connected successfully");
  // Setup motion detection for both sensors
  setupMotionDetection(&mpu_l);
  setupMotionDetection(&mpu_r);

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
  digitalWrite(BUILTIN_PIN, HIGH); // Turn on LED to indicate sampling

  const char* acel_csv = "acel_data.csv";
  if (SD.exists(acel_csv)) {
    SD.remove(acel_csv);
  }
  acelFile = SD.open(acel_csv, FILE_WRITE);
  if (acelFile) {
    Serial.println(String(acel_csv) + " open with success");
    errorFile.print(millis());
    errorFile.println("," + String(acel_csv) + " open with success");
    acelFile.println("timestamp,accX_l,accY_l,accZ_l,gyrX_l,gyrY_l,gyrZ_l,accX_r,accY_r,accZ_r,gyrX_r,gyrY_r,gyrZ_r");
  } else {
    digitalWrite(ERROR_PIN, HIGH);
    Serial.println("Error opening file " + String(acel_csv));
    errorFile.print(millis());
    errorFile.println(",Error opening file " + String(acel_csv));
    stopRecording();
    stopSampling(); // Stop sampling if file open failed    
    return;
  }

  startRecording();

  unsigned long start_timestamp = millis();
  unsigned long last_timestamp = start_timestamp;

  while (samplingState) {    
    continueRecording();
    unsigned long current_timestamp = millis();

    if(((SD.totalSize() - SD.usedSize())/SD.totalSize()) * 100 > 80) {

      digitalWrite(BUILTIN_PIN, LOW);
      Serial.println("Memoria a 80\%");
      errorFile.print(millis());
      errorFile.println("Memoria a 80\%");
      digitalWrite(ERROR_PIN, HIGH);
      stopRecording();
      stopSampling();
      
      return;
    }
    if(current_timestamp - last_timestamp >= SAMPLING_PERIOD_MS) {
      button.update();
      last_timestamp = current_timestamp;

      sensors_event_t a_l, g_l, temp_l;
      mpu_l.getEvent(&a_l, &g_l, &temp_l);
      if(mpu_l.getClock() != 1) {
        Serial.println("Accel Left is Disconnected");
        errorFile.print(millis());
        errorFile.println(",Accel Left is Disconnected");
        unsigned long start_error_accel = millis();
        unsigned long current_error_accel = millis();
        while(!mpu_l.begin(0x68, &Wire, 0)) {
          current_error_accel = millis();
          Serial.println("Failed to find MPU6050_l chip");
          errorFile.print(millis());
          errorFile.println(",Failed to find MPU6050_l chip");
          digitalWrite(ERROR_PIN, HIGH);
          if(current_error_accel - start_error_accel > 10000) {
            digitalWrite(ERROR_PIN, HIGH);
            stopRecording();
            stopSampling();            
            return;
          }
          delay(100);
        }        
        Serial.println("Accel Left Reconnected");     
        errorFile.print(millis());
        errorFile.println(",Accel Left Reconnected"); 
        digitalWrite(ERROR_PIN, LOW);
      }      
      recordSensorData(&a_l, &g_l, acelFile, start_timestamp, current_timestamp);

      sensors_event_t a_r, g_r, temp_r;
      mpu_r.getEvent(&a_r, &g_r, &temp_r);
      if(mpu_r.getClock() != 1) {
        Serial.println("Accel Right is Disconnected");
        errorFile.print(millis());
        errorFile.println(",Accel Right is Disconnected");
        unsigned long start_error_accel = millis();
        unsigned long current_error_accel = millis();
        while(!mpu_r.begin(0x68, &Wire1, 0)) {
          current_error_accel = millis();
          Serial.println("Failed to find MPU6050_r chip");
          errorFile.print(millis());
          errorFile.println(",Failed to find MPU6050_r chip");
          digitalWrite(ERROR_PIN, HIGH);
          if(current_error_accel - start_error_accel > 10000) {
            digitalWrite(ERROR_PIN, HIGH);
            stopRecording();
            stopSampling();            
            return;
          }
          delay(100);
        }   
        digitalWrite(ERROR_PIN, LOW);     
        Serial.println("Accel Right Reconnected"); 
        errorFile.print(millis()); 
        errorFile.println(",Accel Right Reconnected");      
      }
      recordSensorData(&a_r, &g_r, acelFile, start_timestamp, current_timestamp);

      acelFile.println();

      if (button.released()) { // Button pressed down again
        stopRecording(); 
        stopSampling(); // Stop sampling and break out of loop            
        Serial.println("Stoped Sampling");
        errorFile.print(millis());
        errorFile.println(",Stoped Sampling");
        break;
      }
    }
  }

  return;
}

void recordSensorData(sensors_event_t* a, sensors_event_t* g, File& file, unsigned long start_timestamp, unsigned long timestamp) {

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
  digitalWrite(BUILTIN_PIN, LOW); // Turn off LED to indicate sampling stopped
  acelFile.close();
  errorFile.close();
}

void startRecording() {  
  if (SD.exists("RECORD.WAV")) {
    // The SD library writes new data to the end of the
    // file, so to start a new recording, the old file
    // must be deleted before new data is written.
    SD.remove("RECORD.WAV");
  }
  audio_data = SD.open("RECORD.WAV", FILE_WRITE);

  if (audio_data) {
    // Define WAV file parameters
    int sampleRate = 192000; // Change as needed
    int bitDepth = 16;       // Change as needed
    int channels = 1;        // Change as needed
    int dataSize = 0;
    writeWavHeader(audio_data, sampleRate, bitDepth, channels, dataSize);
    queue1.begin();
    Serial.println("startRecording");
    errorFile.print(millis());
    errorFile.println(",startRecording");
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
  errorFile.print(millis());
  errorFile.println(",stopRecording");

  updateDataSizeInHeader(audio_data);

  /*// Define WAV file parameters
  int sampleRate = 192000; // Change as needed
  int bitDepth = 16;       // Change as needed
  int channels = 1;        // Change as needed*/

  // Copy the file and add WAV header
  /*if (copyFileWithWavHeader(sampleRate, bitDepth, channels)) {
    Serial.println("File copied and WAV header added successfully.");
  } else {
    Serial.println("File copy or WAV header addition failed.");
    digitalWrite(ERROR_PIN, HIGH);
    stopSampling(); 
  }*/

}


bool copyFileWithWavHeader(int sampleRate, int bitDepth, int channels) {

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
  /*File file = SD.open(sourceFile, FILE_WRITE);
  if (!file) {
    Serial.println("Failed to open source file for writing.");
    return;
  }*/

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