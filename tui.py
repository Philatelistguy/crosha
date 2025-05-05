import serial
from sigfoxendpoint import send_message
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
SPEECH_ENDPOINT = 'http://crosha-audio.local:5000/speak'  # Update to your TTS endpoint


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
    desc = 'Cardiac Resuscitation, Oxygenation and Smart Health Assistant'
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


def wait_for_execution(ser):
    """Wait for the Arduino to send back 'executed'."""
    ser.flushInput()
    while True:
        response = ser.readline().decode(errors='ignore').strip()
        if response == "CMD:executed":
            if monitor_ecg.verbose:
                print("Arduino response: executed")
            break
        time.sleep(0.1)
def monitor_ecg(model, ser, start_event, exit_event):
    buffer = []
    vf_vt_counter = 0
    class_names = ['Normal', 'VF', 'VT']

    while not exit_event.is_set():
        if not start_event.is_set():
            time.sleep(0.1)
            continue
        ser.flushInput()
        line = ser.readline().decode().strip()
        if line.startswith("ECG:"):
            try:
                val = float(line.split("ECG:")[1])  # Extract the value after "ECG:"
                buffer.append(val)
                if monitor_ecg.verbose:
                    print(f"Read: {val:.3f} (buffer {len(buffer)}/{WINDOW_SIZE})")
            except (ValueError, IndexError):
                if monitor_ecg.verbose:
                    print(f"Ignored malformed data: '{line}'")
                continue
        else:
            if monitor_ecg.verbose:
                print(f"Ignored non-ECG data: '{line}'")
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
                if vf_vt_counter == 1:
                    ser.write(('CMD:ECGSTOP\n').encode('utf-8'))
                    ser.write(('20\n').encode('utf-8'))
                    wait_for_execution(ser)
                    ser.write(('CMD:ECGSTART\n').encode('utf-8'))
                elif vf_vt_counter == 2:
                    ser.write(('CMD:ECGSTOP\n').encode('utf-8'))
                    ser.write(('40\n').encode('utf-8'))
                    wait_for_execution(ser)
                    ser.write(('CMD:ECGSTART\n').encode('utf-8'))
                elif vf_vt_counter == 3:
                    ser.write(('CMD:ECGSTOP\n').encode('utf-8'))
                    ser.write(('60\n').encode('utf-8'))
                    wait_for_execution(ser)
                    ser.write(('CMD:ECGSTART\n').encode('utf-8'))
                elif vf_vt_counter == 4:
                    ser.write(('CMD:ECGSTOP\n').encode('utf-8'))
                    ser.write(('80\n').encode('utf-8'))
                    wait_for_execution(ser)
                    ser.write(('CMD:ECGSTART\n').encode('utf-8'))
                elif vf_vt_counter == 5:
                    ser.write(('CMD:ECGSTOP\n').encode('utf-8'))
                    ser.write(('100\n').encode('utf-8'))
                    wait_for_execution(ser)
                    ser.write(('CMD:ECGSTART\n').encode('utf-8'))

                print(f"⚠️ Consecutive VF/VT: {vf_vt_counter}/{THRESHOLD}")

                speak("Preparing to shock patient.")
            else:
                vf_vt_counter = 0
                ser.write(('0\n').encode('utf-8'))
                wait_for_execution(ser)

            if vf_vt_counter >= THRESHOLD:
                ser.write(('CMD:ECGSTOP\n').encode('utf-8'))
                ser.write(('0\n').encode('utf-8'))
                wait_for_execution(ser)
                print("💥 SHOCK TRIGGERED! Patient unresponsive.")
                speak("Shock triggered. Patient unresponsive.")
                send_message("shock triggered")
                ser.write(('shock\n').encode('utf-8'))
                wait_for_execution(ser)
                print("💤 Shock delivered. Waiting for 5 seconds before resetting counter.")
                print("🔄 Resetting shock counter.")
                vf_vt_counter = 0
                ser.write(('CMD:ECGSTART\n').encode('utf-8'))
                print("🔄 Resuming monitoring after shock.")
                speak("Resuming monitoring after shock.")
                start_event.set()

            buffer.clear()

# Initialize verbose flag
monitor_ecg.verbose = False


def main():
    print_banner()
    speak("Cardiac Resuscitation, Oxygenation and Smart Health Assistant is now starting up.")
    print("Loading model and serial port...")
    model = load_model(MODEL_PATH)
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for the serial connection to establish
    ser.flushInput()
    ser.write(('0\n').encode('utf-8'))
    wait_for_execution(ser)
    print("Initialization complete. Ready to monitor.\n")
    ser.write(('CMD:start\n').encode('utf-8'))
    wait_for_execution(ser)
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
                ser.write(('0\n').encode('utf-8'))
                wait_for_execution(ser)
                ser.write(('CMD:ECGSTART\n').encode('utf-8'))
                print(ser.readline().decode(errors='ignore').strip())
                speak("Monitoring started.") #give a delay of 1 second before starting the monitoring
            elif key == '2':
                start_event.clear()
                vf_vt_counter = 0
                ser.write(('0\n').encode('utf-8'))
                wait_for_execution(ser)
                ser.write(('CMD:ECGSTOP\n').encode('utf-8'))
                print("🔴 Stopped monitoring.")
                speak("Monitoring stopped.")
            elif key == '3':
                monitor_ecg.verbose = not monitor_ecg.verbose
                state = 'enabled' if monitor_ecg.verbose else 'disabled'
                print(f"📝 Verbose {state}.")
                speak(f"Verbose mode {state}.")
            elif key == '4':
                ser.write('0\n'.encode('utf-8'))
                wait_for_execution(ser)
                ser.write(b'CMD:ECGSTOP\n')
                print("Exiting...")
                speak("Exiting CROSHA. Goodbye.")
                exit_event.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        monitor_thread.join()

if __name__ == '__main__':
    main()