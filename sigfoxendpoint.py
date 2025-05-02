from flask import Flask, request, jsonify
import serial
import time


# Configure serial connection to Sigfox module
ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)

def send_to_sigfox(hex_data):
    command = f"AT$SF={hex_data}\r".encode()
    ser.write(command)
    time.sleep(1)
    response = ser.read(64)
    return response.decode(errors='ignore')

def send_message(message):
    
    # Convert text to hexadecimal
    hex_data = message.encode().hex()
    
    # Send to Sigfox
    response = send_to_sigfox(hex_data)
