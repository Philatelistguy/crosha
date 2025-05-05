#include <Arduino.h>
#include <avr/pgmspace.h>
#include <FastLED.h>

#define LED_PIN 6
#define NUM_LEDS 47
#define BRIGHTNESS 255
#define LED_TYPE WS2812B
#define COLOR_ORDER GRB
#define SAMPLE_INTERVAL_MS 10
#define ECG_LENGTH 100
#define NORMAL_PIN 2
#define VF_PIN 3
#define ECG_PIN 4
const int heartPin = A1;

CRGB leds[NUM_LEDS];
int currentLedsLit = 0;
bool animationRunning = false;
unsigned long previousTime = 0;
bool sendECG = false;
int ecgIndex = 0;
unsigned long lastSampleTime = 0;
int ecgType = 0; // 0: none, 1: VF, 2: Normal
bool previousPinState = true;
bool manualStart = false;
bool manualStop = false;
bool ecgRunning = false;

// VF waveform
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

// Normal waveform
const float ecgNormal[ECG_LENGTH] PROGMEM = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.05f, 0.1f, 0.15f, 0.1f, 0.05f,
    0.0f, 0.0f, 0.0f,
    -0.15f, 0.3f, 1.0f, -0.4f, -0.1f,
    0.1f, 0.1f, 0.1f, 0.1f,
    0.1f, 0.15f, 0.25f, 0.2f, 0.15f, 0.1f,
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

void setup() {
    Serial.begin(115200);
    FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);
    FastLED.clear();
    FastLED.show();
    pinMode(NORMAL_PIN, INPUT_PULLUP);
    pinMode(VF_PIN, INPUT_PULLUP);
    pinMode(ECG_PIN, INPUT_PULLUP);
}

void loop() {
        if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();

        if (input.equalsIgnoreCase("shock")) {
            animateShock();
            FastLED.clear();
            FastLED.show();
            currentLedsLit = 0;
            Serial.println("CMD:executed");
        } else if (input.equalsIgnoreCase("CMD:start")) {
            startProgramAnimation();
        } else if (input.equalsIgnoreCase("CMD:Radia")) {
            radia();
        } else if (input.equalsIgnoreCase("CMD:stop")) {
            stop();
        } else if (input.equalsIgnoreCase("CMD:ECGSTART")) {
            sendECG = true;
            manualStart = true;
            manualStop = false;
            ecgRunning = true;
            ecgIndex = 0;
            lastSampleTime = millis();
            Serial.println("CMD:executed");
        } else if (input.equalsIgnoreCase("CMD:ECGSTOP")) {
            sendECG = false;
            manualStop = true;
            manualStart = false;
            ecgType = 0;
            ecgIndex = 0;
            lastSampleTime = 0;
            ecgRunning = false;
            delay(0.5);
            Serial.println("CMD:executed");
        } else {
            int percent = input.toInt();
            if (percent >= 0 && percent <= 100) {
                int targetLedsLit = map(percent, 0, 100, 0, NUM_LEDS);
                animateFadingProgress(currentLedsLit, targetLedsLit);
                currentLedsLit = targetLedsLit;
                Serial.println("CMD:executed");
            }
        }
    }

    bool currentPinState = (digitalRead(NORMAL_PIN) == HIGH && digitalRead(VF_PIN) == HIGH && digitalRead(ECG_PIN) == HIGH);
    if (currentPinState != previousPinState) {
        previousPinState = currentPinState;

        if (currentPinState) {
            ecgType = 0;
            sendECG = false;
            ecgRunning = false;
            Serial.println("No pin connected. Stopping ECG.");
        } else {
            if (digitalRead(NORMAL_PIN) == LOW)
                ecgType = 2;
            else if (digitalRead(VF_PIN) == LOW)
                ecgType = 1;
            else if (digitalRead(ECG_PIN) == LOW)
                ecgType = 3;

            if (manualStart && !manualStop) {
                sendECG = true;
                ecgIndex = 0;
                lastSampleTime = millis();
                ecgRunning = true;
                Serial.println("ECG re-started by pin reconnection");
            }
        }
    }

       if (sendECG && ecgRunning) {
        int type;
        if (digitalRead(NORMAL_PIN) == LOW)      type = 2;
        else if (digitalRead(VF_PIN) == LOW)     type = 1;
        else if (digitalRead(ECG_PIN) == LOW)    type = 3;
        else      type=0;
        unsigned long currentTime = millis();
        if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
            lastSampleTime = currentTime;
            float value = 0.0f;
            if (type == 1) {
                float base = pgm_read_float_near(&ecgVF[ecgIndex]);
                value = base + (random(-9, 10)) / 100.0f;
            } else if (type == 2) {
                float base = pgm_read_float_near(&ecgNormal[ecgIndex]);
                value = base + (random(-3, 4)) / 100.0f;
            } else if (type == 3) {
                int raw = analogRead(heartPin);
                value = ((float)raw / 1023.0f) * 5.0f;
            }
            Serial.print("ECG:");
            Serial.println(value);
            ecgIndex = (ecgIndex + 1) % ECG_LENGTH;
        }
    }
}

void startProgramAnimation() {
    if (animationRunning) return;
    animationRunning = true;
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = CHSV(map(i, 0, NUM_LEDS - 1, 0, 255), 255, 255);
    }
    unsigned long startTime = millis();
    int cycle = 0, brightness = 0;
    bool fadingUp = true;
    while (animationRunning) {
        unsigned long currentTime = millis();
        if (currentTime - startTime >= 20) {
            startTime = currentTime;
            brightness += fadingUp ? 5 : -5;
            brightness = constrain(brightness, 0, 255);
            FastLED.setBrightness(brightness);
            FastLED.show();
            if (brightness == 0 && !fadingUp) {
                cycle++;
                if (cycle >= 3) break;
                fadingUp = true;
            } else if (brightness == 255 && fadingUp) {
                fadingUp = false;
            }
        }
    }
    FastLED.setBrightness(BRIGHTNESS);     // ← put it back
    FastLED.clear();
    FastLED.show();
    animationRunning = false;
    Serial.println("CMD:executed");
}

void animateFadingProgress(int from, int to) {
    if (animationRunning) return;
    animationRunning = true;
    unsigned long startTime = millis();
    int step = (from < to) ? 1 : -1;
    int i = from;
    int b = (step > 0) ? 0 : 255;

    while (animationRunning && i != to) {
        unsigned long currentTime = millis();
        if (currentTime - startTime >= 10) {
            startTime = currentTime;
            if (step > 0) {
                b = min(b + 50, 255);
                leds[i] = CRGB(b, 0, 0);
            } else {
                b = max(b - 50, 0);
                leds[i - 1] = CRGB(b, 0, 0);
            }
            FastLED.show();
            i += step;
        }
    }

    for (int j = to; j < NUM_LEDS; j++) leds[j] = CRGB::Black;
    FastLED.show();
    animationRunning = false;
}

void animateShock() {
    if (animationRunning) return;
    animationRunning = true;
    FastLED.setBrightness(BRIGHTNESS);
    const int flashes = 6;
    CRGB violetColor = CHSV(160, 255, 200);
    unsigned long startTime = millis();
    int f = 0, subState = 0;

    while (animationRunning && f < flashes) {
        unsigned long currentTime = millis();
        if (currentTime - startTime >= 50) {
            startTime = currentTime;
            switch (subState) {
                case 0:
                    FastLED.clear();
                    for (int i = 0; i < NUM_LEDS / 5; i++) {
                        int idx = random(NUM_LEDS);
                        leds[idx] = violetColor;
                        leds[idx].fadeToBlackBy(random(100));
                    }
                    FastLED.show();
                    subState = 1;
                    break;
                case 1:
                    fill_solid(leds, NUM_LEDS, violetColor);
                    FastLED.show();
                    subState = 2;
                    break;
                case 2:
                    FastLED.clear();
                    FastLED.show();
                    subState = 0;
                    f++;
                    break;
            }
        }
    }
    animationRunning = false;
}

void stop() {
    animationRunning = false;
    FastLED.clear();
    FastLED.show();
    Serial.println("CMD:executed");
}

void radia() {
    if (animationRunning) return;
    animationRunning = true;
    FastLED.setBrightness(BRIGHTNESS);
    unsigned long startTime = millis();
    uint8_t hue = 0;
    int cycle = 0;

    while (animationRunning && cycle < 100) {
        unsigned long currentTime = millis();
        if (currentTime - startTime >= 50) {
            startTime = currentTime;
            for (int i = 0; i < NUM_LEDS; i++) {
                leds[i] = CHSV(hue + (i * 10), 255, 255);
            }
            FastLED.show();
            hue += 10;
            cycle++;
        }
    }
    FastLED.clear();
    FastLED.show();
    animationRunning = false;
    Serial.println("CMD:executed");
}
