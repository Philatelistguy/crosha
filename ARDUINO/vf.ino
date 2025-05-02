#include <Arduino.h>
#include <avr/pgmspace.h>

#define SAMPLE_INTERVAL_MS 10  // 100Hz sampling
#define ECG_LENGTH 100         // Smaller dataset
unsigned long lastSampleTime = 0;
int ecgIndex = 0;

// Mini Ventricular Fibrillation waveform (PROGMEM)
const float ecgVF[ECG_LENGTH] PROGMEM = {
  0.15f, 0.30f, 0.50f, 0.60f, 0.50f, 0.25f, -0.05f, -0.30f, -0.45f, -0.50f,
 -0.35f, -0.10f, 0.10f, 0.25f, 0.35f, 0.30f, 0.10f, -0.10f, -0.30f, -0.40f,
 -0.45f, -0.35f, -0.10f, 0.15f, 0.40f, 0.55f, 0.60f, 0.50f, 0.25f, -0.05f,
 -0.30f, -0.45f, -0.50f, -0.40f, -0.20f, 0.00f, 0.20f, 0.35f, 0.40f, 0.30f,
  0.15f, -0.05f, -0.25f, -0.35f, -0.40f, -0.30f, -0.10f, 0.15f, 0.40f, 0.60f,
  0.70f, 0.60f, 0.40f, 0.15f, -0.10f, -0.35f, -0.45f, -0.40f, -0.25f, -0.05f,
  0.15f, 0.30f, 0.40f, 0.35f, 0.20f, 0.00f, -0.20f, -0.35f, -0.40f, -0.30f,
 -0.10f, 0.10f, 0.30f, 0.50f, 0.60f, 0.55f, 0.30f, 0.00f, -0.25f, -0.45f,
 -0.55f, -0.50f, -0.30f, -0.05f, 0.20f, 0.40f, 0.50f, 0.45f, 0.25f, 0.05f,
 -0.15f, -0.30f, -0.35f, -0.30f, -0.10f, 0.10f, 0.25f, 0.35f, 0.30f, 0.15f
};

void setup() {
  Serial.begin(115200);
  Serial.println("Mini VF ECG Simulation (100Hz)");
}

void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentTime;

    float baseValue = pgm_read_float_near(&ecgVF[ecgIndex]);
    float jitter = (random(-9, 10)) / 100.0f;
    float value = baseValue + jitter;

    Serial.println(value);

    ecgIndex++;
    if (ecgIndex >= ECG_LENGTH) ecgIndex = 0;
  }
}
