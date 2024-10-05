// Version T4.1
//    This program (or code that looks like it)
//    can be found in many places.
//    For example on the Arduino.cc forum.
//    The original author is not know.

#include <Wire.h>

void setup()
{
  Wire.setSCL(19);  // SCL on first i2c bus on T4.1
  Wire.setSDA(18);  // SDA on first i2c bus on T4.1
  Wire1.setSCL(16); // SCL1 on second i2c bus on T4.1
  Wire1.setSDA(17); // SDA1 on second i2c bus on T4.1
  Wire2.setSCL(24); // SCL2 on second i2c bus on T4.1
  Wire2.setSDA(25); // SDA2 on second i2c bus on T4.1

  Wire.begin();
  Wire1.begin();
  Wire2.begin();

  Serial.begin(9600);
  while (!Serial);             // Leonardo: wait for serial monitor
  Serial.println("\nI2C Scanner");
}


void loop()
{
  byte error, address;
  int nDevices;

  Serial.println("Scanning...");

  nDevices = 0;
  for (address = 1; address < 127; address++ ) //I'm guessing I haven't updated this correctly
  {
    // The i2c_scanner uses the return value of
    // the Write.endTransmisstion to see if
    // a device did acknowledge to the address.

    //---------------------------------------------------------------------------------
    Wire.beginTransmission(address); // i2c on first bus starts here
    error = Wire.endTransmission();

    if (error == 0)
    {
      Serial.print("I2C device found at Wire 0x");
      if (address < 16)
        Serial.print("0");
      Serial.print(address, HEX);
      Serial.println("  !");

      nDevices++;
    }
    else if (error == 4)
    {
      Serial.print("Unknown error at Wire 0x");
      if (address < 16)
        Serial.print("0");
      Serial.println(address, HEX);
    } // i2c on first bus ends here

    //---------------------------------------------------------------------------------
    Wire1.beginTransmission(address); // i2c on second bus starts here
    error = Wire1.endTransmission();

    if (error == 0)
    {
      Serial.print("I2C device found at Wire1 0x");
      if (address < 16)
        Serial.print("0");
      Serial.print(address, HEX);
      Serial.println("  !");

      nDevices++;
    }
    else if (error == 4)
    {
      Serial.print("Unknown error at Wire1 0x");
      if (address < 16)
        Serial.print("0");
      Serial.println(address, HEX);
    }
                                   // i2c on second bus ends here

  //---------------------------------------------------------------------------------

  Wire2.beginTransmission(address); // i2c on third bus starts here
    error = Wire2.endTransmission();

    if (error == 0)
    {
      Serial.print("I2C device found at Wire2 0x");
      if (address < 16)
        Serial.print("0");
      Serial.print(address, HEX);
      Serial.println("  !");

      nDevices++;
    }
    else if (error == 4)
    {
      Serial.print("Unknown error at Wire2 0x");
      if (address < 16)
        Serial.print("0");
      Serial.println(address, HEX);
    }
  }                                 // i2c on third bus ends here

 //---------------------------------------------------------------------------------

  if (nDevices == 0)
    Serial.println("No I2C devices found\n");
  else
    Serial.print("number of devices found = ");
  Serial.println(nDevices);  // this prints the total number of i2c devices found
  Serial.println("done\n");

  delay(2000);           // wait 2 seconds for next scan
}