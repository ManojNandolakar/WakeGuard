int relayPin = 9;
int ledPin = 13;

void setup() {
  Serial.begin(9600);

  pinMode(relayPin, OUTPUT);
  pinMode(ledPin, OUTPUT);

  // Assume ACTIVE HIGH relay (fix for your issue)
  digitalWrite(relayPin, LOW);  // MOTOR OFF
  digitalWrite(ledPin, LOW);    // LED OFF

  Serial.println("System Ready");
}

void loop() {

  if (Serial.available()) {

    char data = Serial.read();

    // ================= LED CONTROL =================
    if (data == '1') {
      Serial.println("LED ON");
      digitalWrite(ledPin, HIGH);
    }

    else if (data == '0') {
      Serial.println("LED OFF");
      digitalWrite(ledPin, LOW);
    }

    // ================= MOTOR CONTROL =================
    else if (data == 'M') {
      Serial.println("MOTOR ON");
      digitalWrite(relayPin, HIGH);   // ON (ACTIVE HIGH)
    }

    else if (data == 'S') {
      Serial.println("MOTOR OFF");
      digitalWrite(relayPin, LOW);    // OFF
    }
  }
}