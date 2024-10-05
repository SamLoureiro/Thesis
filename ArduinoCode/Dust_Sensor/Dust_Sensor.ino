void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  float dust = 0;
  dust = analogRead(27);
  Serial.print("Dust Sensor: ");
  Serial.println(dust);

  float gas = 0;
  gas = analogRead(41);
  Serial.print("Gas sensor: ");
  Serial.println(gas);
  delay(500);
}
