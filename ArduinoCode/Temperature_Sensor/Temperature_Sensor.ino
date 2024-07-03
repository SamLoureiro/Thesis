// Define the pin where the TMP235 sensor is connected
const int analogPin = A12; // Analog input pin for the TMP235 sensor

// Define the reference voltage (3.3V for this case)
const float referenceVoltage = 3.3;

void setup() {
  // Initialize the serial communication:
  Serial.begin(9600);
}

void loop() {
  // Read the analog value from the TMP235 sensor:
  int sensorValue = analogRead(analogPin);

  // Convert the analog value to voltage:
  float voltage = sensorValue * (referenceVoltage / 1023.0);

  // Convert the voltage to temperature in Celsius using the provided formula:
  float temperatureC = 100 * voltage - 50;

  // Print the temperature to the Serial Monitor:
  Serial.print("Temperature: ");
  Serial.print(temperatureC);
  Serial.println(" °C");

  // Wait for a bit before taking another reading:
  delay(1000);
}
