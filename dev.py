import numpy as np
import tensorflow as tf
from tensorflow import keras
from sigfoxendpoint import send_message
# --- Labels ---
NORMAL = 0
VF = 1
VT = 2
function_names = ['Normal', 'VF', 'VT']
# --- Load Model ---
model = tf.keras.models.load_model("ecg_model1.keras")
print("Model loaded successfully!")
# --- Load ECG Data ---
import numpy as np
from scipy.signal import butter, filtfilt
import random
WINDOW_SIZE = 300  # Number of samples to use for each ECG segment ( ~0.83 seconds)
def trigger_defibrillator():
    """Simulate triggering a defibrillator."""
    print("Defibrillator triggered!")

def predict_and_trigger(model, ecg_segment, threshold=0.9):
    """
    Predicts the class of an ECG segment and triggers the defibrillator if VF or VT is detected
    with a confidence above a threshold.

    Args:
        model: The trained Keras model.
        ecg_segment: A NumPy array representing the ECG segment to classify.
        threshold: The confidence threshold for triggering the defibrillator.
    """
    print("Making prediction and potentially triggering defibrillator...")
    # Reshape the ECG segment to match the model's input shape
    ecg_segment = np.expand_dims(ecg_segment, axis=0)  # Add batch dimension

    prediction = model.predict(ecg_segment)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    class_names = ['Normal', 'VF', 'VT']
    print(f"Predicted Class: {class_names[predicted_class]} with confidence: {confidence:.4f}")

    if (predicted_class == VF or predicted_class == VT) and confidence >= threshold:
        trigger_defibrillator()
        send_message("Person shocked")
    else:
        print("No immediate action required.")
    print("Prediction and trigger check complete.")
if __name__ == "__main__":
        sample_ecg_segment = np.random.rand(WINDOW_SIZE)  # Dummy ECG segment
        predict_and_trigger(model, sample_ecg_segment, threshold=0.7)
