
import serial
import numpy as np
from tensorflow.keras.models import load_model
import time
from sigfoxendpoint import send_message


# --- CONFIG ---
SERIAL_PORT = '/dev/ttyACM0'  # Update if needed
BAUD_RATE = 115200
WINDOW_SIZE = 300      # Or 360, depending on model
MODEL_PATH = 'cardionet1.0/ecg_model.keras'


# --- INIT ---
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
model = load_model(MODEL_PATH)
buffer = []
vf_vt_counter = 0             # Counter for consecutive VF/VT detections
THRESHOLD = 5                 # Number of consecutive abnormal windows to trigger "shock"

print("Listening to ECG stream...")

while True:
    try:
        line = ser.readline().decode().strip()
        if not line:
            continue

        value = float(line)
        buffer.append(value)

        if len(buffer) == WINDOW_SIZE:
            sample = np.array(buffer)

            if len(model.input_shape) == 3:
                sample = sample.reshape(1, WINDOW_SIZE, 1)
            else:
                sample = sample.reshape(1, WINDOW_SIZE)
            threshold=0.70
            prediction = model.predict(sample, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]
            class_names = ['Normal','VF','VT']
            result = class_names[predicted_class]
            print(f"Predicted Class: {result} with confidence: {confidence:.4f}")
            if result in ['VF', 'VT'] and confidence > threshold:
                vf_vt_counter += 1
                print(f"??  Warning: {result} detected {vf_vt_counter} time(s) in a row")
            else:
                vf_vt_counter = 0

            # Trigger shock after 5 consecutive abnormal windows
            if vf_vt_counter >= THRESHOLD:
                print("?? SHOCK TRIGGERED! Patient unresponsive to 5 consecutive VF/VT windows.")
                send_message("Hello")
                vf_vt_counter = 0  # Reset after shock

            buffer = []

    except Exception as e:
        print("Error:", e)
        time.sleep(1)
