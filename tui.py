import serial
import numpy as np
from tensorflow.keras.models import load_model
import time
import threading
import sys
import select
import termios
import tty
import requests
from pyfiglet import Figlet

# --- CONFIG ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
WINDOW_SIZE = 300
MODEL_PATH = 'cardionet1.0/ecg_model.keras'
THRESHOLD = 5
CONFIDENCE_THRESHOLD = 0.70
SPEECH_ENDPOINT = 'http://192.168.1.45:5000/speak'  # Update to your TTS endpoint


def speak(text):
    """Send text to speech endpoint as JSON."""
    try:
        resp = requests.post(SPEECH_ENDPOINT, json={'text': text})
        resp.raise_for_status()
    except Exception as e:
        print(f"⚠️ Speech request failed: {e}")


def print_banner():
    """
    Render ASCII art for 'CROSHA' and display a description box.
    """
    fig = Figlet(font='slant')
    print(fig.renderText('CROSHA'))
    desc = 'Cardia Resuscitation, Oxygenation and Smart Health Assistant'
    border = '=' * len(desc)
    print(border)
    print(desc)
    print(border)
    print()


def get_key(timeout=0.1):
    """Read a single keypress (non-blocking)."""
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    if dr:
        return sys.stdin.read(1)
    return None


def monitor_ecg(model, ser, start_event, exit_event):
    buffer = []
    vf_vt_counter = 0
    class_names = ['Normal', 'VF', 'VT']

    while not exit_event.is_set():
        if not start_event.is_set():
            time.sleep(0.1)
            continue

        line = ser.readline().decode(errors='ignore').strip()
        try:
            val = float(line)
            buffer.append(val)
            if monitor_ecg.verbose:
                print(f"Read: {val:.3f} (buffer {len(buffer)}/{WINDOW_SIZE})")
        except ValueError:
            if monitor_ecg.verbose:
                print(f"Ignored non-numeric: '{line}'")
            continue

        if len(buffer) >= WINDOW_SIZE:
            data = np.array(buffer)
            if len(model.input_shape) == 3:
                data = data.reshape(1, WINDOW_SIZE, 1)
            else:
                data = data.reshape(1, WINDOW_SIZE)
            pred = model.predict(data, verbose=0)[0]
            idx = int(np.argmax(pred))
            confidence = float(pred[idx])
            result = class_names[idx]

            print(f"Predicted Class: {result}    Confidence: {confidence:.4f}")

            if result in ['VF', 'VT'] and confidence > CONFIDENCE_THRESHOLD:
                vf_vt_counter += 1
                print(f"⚠️ Consecutive VF/VT: {vf_vt_counter}/{THRESHOLD}")
                speak("Preparing to shock patient.")
            else:
                vf_vt_counter = 0

            if vf_vt_counter >= THRESHOLD:
                print("💥 SHOCK TRIGGERED! Patient unresponsive.")
                speak("Shock triggered. Patient unresponsive.")
                vf_vt_counter = 0

            buffer.clear()

# Initialize verbose flag
monitor_ecg.verbose = False


def main():
    print_banner()
    speak("CROSHA is now starting up.")
    print("Loading model and serial port...")
    model = load_model(MODEL_PATH)
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Initialization complete. Ready to monitor.\n")
    speak("CROSHA is now ready for monitoring.")

    print("Controls: [1] Start  [2] Stop  [3] Toggle Verbose  [4] Exit (press key anytime)\n")

    start_event = threading.Event()
    exit_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_ecg,
        args=(model, ser, start_event, exit_event),
        daemon=True
    )
    monitor_thread.start()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        while not exit_event.is_set():
            key = get_key()
            if not key:
                continue
            if key == '1':
                start_event.set()
                print("🟢 Started monitoring.")
                speak("Monitoring started.")
            elif key == '2':
                start_event.clear()
                print("🔴 Stopped monitoring.")
                speak("Monitoring stopped.")
            elif key == '3':
                monitor_ecg.verbose = not monitor_ecg.verbose
                state = 'enabled' if monitor_ecg.verbose else 'disabled'
                print(f"📝 Verbose {state}.")
                speak(f"Verbose mode {state}.")
            elif key == '4':
                print("Exiting...")
                speak("Exiting CROSHA. Goodbye.")
                exit_event.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        monitor_thread.join()

if __name__ == '__main__':
    main()
