// Define pin assignments
int STBY = 10; // Standby

// Motor A
int PWMA = 3; // Speed control
int AIN1 = 9; // Direction
int AIN2 = 8; // Direction

void setup() {
  // Initialize the motor control pins as outputs
  pinMode(STBY, OUTPUT);
  pinMode(PWMA, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  
  // Initialize Serial communication for debugging
  Serial.begin(9600);
  
  // Disable standby mode
  digitalWrite(STBY, HIGH);
  Serial.println("Motor driver initialized, standby mode disabled.");
  Serial.println("Enter a speed value between -100 and 100 to control the motor.");
  Serial.println("Enter 's' to stop the motor.");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim(); // Remove any leading/trailing whitespace

    if (input.equalsIgnoreCase("s")) {
      // Stop the motor
      Serial.println("Stopping motor.");
      stopMotor();
    } else {
      int speed = input.toInt(); // Convert input to integer

      // Ensure the input is within the valid range
      if (speed >= -100 && speed <= 100) {
        Serial.print("Setting motor speed to: ");
        Serial.println(speed);
        setMotorSpeed(speed);
      } else {
        Serial.println("Invalid speed value. Please enter a value between -100 and 100.");
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
