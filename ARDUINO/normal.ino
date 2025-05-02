#define SAMPLE_INTERVAL_MS 10  // 100Hz sampling rate
#define ECG_LENGTH 100         // Simulate one full heartbeat
unsigned long lastSampleTime = 0;
int ecgIndex = 0;

float ecgWaveform[ECG_LENGTH] = {
  // Baseline
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

  // P wave (smooth bump)
  0.05, 0.1, 0.15, 0.1, 0.05,

  // PR segment (return to baseline)
  0.0, 0.0, 0.0,

  // QRS complex (sharp spike)
  -0.15, 0.3, 1.0, -0.4, -0.1,

  // ST segment (flat-ish)
  0.1, 0.1, 0.1, 0.1,

  // T wave (broader bump)
  0.1, 0.15, 0.25, 0.2, 0.15, 0.1,

  // Return to baseline
  0.0, 0.0, 0.0, 0.0,

  // Fill the rest with baseline to complete the beat
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
};

void setup() {
  Serial.begin(9600);
}

void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentTime;

    float value = ecgWaveform[ecgIndex];
    Serial.println(value);

    ecgIndex = (ecgIndex + 1) % ECG_LENGTH;
  }
}
