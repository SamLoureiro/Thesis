// Define pin assignments
int STBY = 10; // Standby
int buttonPin = 2; // Button input

// Motor A
int PWMA = 3; // Speed control
int AIN1 = 9; // Direction
int AIN2 = 8; // Direction

// Variables to manage the motor cycle
bool motorRunning = false;
unsigned long lastChangeTime = 0;
int cycleState = 0;
unsigned long cycleInterval = 30000; // 30 seconds for each cycle step

void setup() {
  // Initialize the motor control pins as outputs
  pinMode(STBY, OUTPUT);
  pinMode(PWMA, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(buttonPin, INPUT); // Button input with internal pull-up resistor
  
  // Initialize Serial communication for debugging
  Serial.begin(9600);
  
  // Disable standby mode
  digitalWrite(STBY, HIGH);
  Serial.println("Motor driver initialized, standby mode disabled.");
  Serial.println("Press the button to start/stop the motor cycle.");
}

void loop() {
  // Check if the button is pressed
  if (digitalRead(buttonPin) == HIGH) {
    delay(50); // Debounce delay
    if (digitalRead(buttonPin) == HIGH) { // Check again to confirm button press
      while (digitalRead(buttonPin) == HIGH); // Wait for button release
      
      if (!motorRunning) {
        motorRunning = true;
        lastChangeTime = millis();
        cycleState = 0;
        updateMotorCycle(cycleState);
        Serial.println("Button pressed, starting motor cycle.");
      } else {
        motorRunning = false;
        stopMotor();
        Serial.println("Button pressed, stopping motor cycle.");
      }
    }
  }

  // If the motor is running, handle the cycle
  if (motorRunning) {
    unsigned long currentTime = millis();
    if (currentTime - lastChangeTime >= cycleInterval) {
      lastChangeTime = currentTime;
      /*cycleState++;
      if (cycleState > 3) {
        cycleState = 0;
      }*/
      randomSeed(analogRead(0)); 
      int randomNumber = random(0, 3);
      updateMotorCycle(cycleState);
    }
  }
}

// Function to set the motor speed and direction
void setMotorSpeed(int speed) {
  if (speed > 0) {
    moveMotor(map(speed, 0, 100, 0, 255), HIGH, LOW); // Forward
  } else if (speed < 0) {
    moveMotor(map(abs(speed), 0, 100, 0, 255), LOW, HIGH); // Backward
  } else {
    stopMotor();
  }
}

// Function to move the motor
void moveMotor(int pwmValue, int direction1, int direction2) {
  digitalWrite(AIN1, direction1);
  digitalWrite(AIN2, direction2);
  analogWrite(PWMA, pwmValue);
  Serial.print("Motor moving with PWM value: ");
  Serial.print(pwmValue);
  Serial.print(", Direction1: ");
  Serial.print(direction1);
  Serial.print(", Direction2: ");
  Serial.println(direction2);
}

// Function to stop the motor
void stopMotor() {
  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, 0);
  Serial.println("Motor stopped.");
}

// Function to brake the motor
void brakeMotor() {
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, HIGH);
  analogWrite(PWMA, 0);
  Serial.println("Motor braked.");
}

// Function to update the motor state based on the cycle state
void updateMotorCycle(int state) {
  switch (state) {
    case 0:
      Serial.println("Cycle state 0: Motor stopped.");
      setMotorSpeed(-80);
      break;
    case 1:
      Serial.println("Cycle state 1: Motor running forward at 50% speed.");
      setMotorSpeed(50);
      break;
    case 2:
      Serial.println("Cycle state 2: Motor running backward at 50% speed.");
      setMotorSpeed(-50);
      break;
    case 3:
      Serial.println("Cycle state 3: Motor running forward at 100% speed.");
      setMotorSpeed(80);
      break;
  }
}
