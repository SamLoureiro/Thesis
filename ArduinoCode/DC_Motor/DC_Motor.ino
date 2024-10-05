// Define pin assignments
int STBY = 10; // Standby
int buttonPin = 2; // Button input

// Motor A
int PWMA = 3; // Speed control
int AIN1 = 9; // Direction
int AIN2 = 8; // Direction

// Variables to manage the motor state
bool motorRunning = false;

void setup() {
  // Initialize the motor control pins as outputs
  pinMode(STBY, OUTPUT);
  pinMode(PWMA, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(buttonPin, INPUT); // Button input
  
  // Initialize Serial communication for debugging
  Serial.begin(9600);
  
  // Disable standby mode
  digitalWrite(STBY, HIGH);
  Serial.println("Motor driver initialized, standby mode disabled.");
  Serial.println("Press the button to start/stop the motor.");
}

void loop() {
  // Check if the button is pressed
  if (digitalRead(buttonPin) == HIGH) {
    delay(50); // Debounce delay
    if (digitalRead(buttonPin) == HIGH) { // Check again to confirm button press
      while (digitalRead(buttonPin) == HIGH); // Wait for button release

      // Toggle motor state
      motorRunning = !motorRunning;
      if (motorRunning) {
        setMotorSpeed(75); // Set speed to 50% of maximum speed
        Serial.println("Button pressed, starting motor.");
      } else {
        stopMotor();
        Serial.println("Button pressed, stopping motor.");
      }
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
