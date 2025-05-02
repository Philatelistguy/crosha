import numpy as np
import wfdb
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint # Add this import
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# --- Configuration ---
DATABASE_PATH = './mitdb/'  # Replace with the actual path to your MIT-BIH data
SAMPLE_RATE = 360  # Samples per second for the MIT-BIH database
WINDOW_SIZE = 300  # Number of samples to use for each ECG segment ( ~0.83 seconds)
NUM_CLASSES = 3  # Normal, Ventricular Fibrillation (VF), Ventricular Tachycardia (VT)
EPOCHS = 10
BATCH_SIZE = 32
TEST_SIZE = 0.2  # Proportion of data to use for testing


# --- Labels ---
NORMAL = 0
VF = 1
VT = 2

# --- Mappings from annotations in MIT-BIH to labels ---
ANNOTATION_MAPPING = {
    'N': NORMAL,  # Normal Beat
    'VF': VF,      # Ventricular Fibrillation
    'VT': VT,      # Ventricular Tachycardia
    'V': VT,       # Ventricular beat (can be VT)
    '+': VF,      # Fusion of ventricular and normal beat (considered like VF for simplicity)
    '!': VF       # Ventricular flutter wave (considered like VF)
    # Add more mappings as needed for your use case. Handle unmapped cases gracefully.
}


# --- Helper Functions ---

def load_data(database_path, annotation_mapping, window_size, sample_rate, num_classes):
    """Loads ECG data and labels from the MIT-BIH database.

    Args:
        database_path (str): Path to the MIT-BIH database directory.
        annotation_mapping (dict): Mapping from annotation symbols to numerical labels.
        window_size (int): Size of the ECG signal window to use.
        sample_rate (int): Sample rate of the ECG data.
        num_classes (int): Number of classes to classify.

    Returns:
        tuple: A tuple containing the ECG data (X) and the corresponding labels (y) as NumPy arrays.
    """
    print("Loading data...")
    X = []
    y = []
    record_names = [f[:-4] for f in os.listdir(database_path) if f.endswith('.atr')] # get .atr records names
    print(f"Found {len(record_names)} record files.")

    for record_name in record_names:
        print(f"Processing record: {record_name}")
        try:
            record = wfdb.rdrecord(os.path.join(database_path, record_name), sampto=SAMPLE_RATE * 60) # Limit records to 1 minute
            annotation = wfdb.rdann(os.path.join(database_path, record_name), 'atr', sampto=SAMPLE_RATE * 60) # Limit records to 1 minute
            
            signals = record.p_signal[:, 0]  # Use only the first lead for simplicity
            print(f"Loaded record and annotations for {record_name}. Signal length: {len(signals)} samples.")

            for i, symbol in enumerate(annotation.symbol):
                if symbol in annotation_mapping:
                    label = annotation_mapping[symbol]
                    sample = annotation.sample[i]

                    # Extract a window around the beat
                    start = max(0, sample - window_size // 2)
                    end = min(len(signals), sample + window_size // 2)

                    if end - start == window_size: # Only consider windows of the correct size
                        segment = signals[start:end]
                        X.append(segment)
                        y.append(label)
                        print(f"Extracted segment with label {label} from record {record_name} at sample {sample}.")
                    else:
                        print(f"Skipping segment at sample {sample} due to insufficient data around the point.")
        except Exception as e:
            print(f"Error processing record {record_name}: {e}") # Report errors

    X = np.array(X)
    y = keras.utils.to_categorical(np.array(y), num_classes=num_classes)
    print(f"Data loading complete.  X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def build_model(input_shape, num_classes):
    """Builds a simple 1D CNN model for ECG classification.

    Args:
        input_shape (tuple): The shape of the input data (window_size,).
        num_classes (int): The number of classes to classify.

    Returns:
        keras.Model: A compiled Keras model.
    """
    print("Building model...")
    model = keras.Sequential([
        keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape), # Add a channel dimension
        keras.layers.Conv1D(32, 5, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 5, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Use categorical_crossentropy for one-hot encoded labels
                  metrics=['accuracy'])
    print("Model compilation complete.")
    return model


def plot_history(history):
    """Plots the training and validation accuracy and loss curves."""
    print("Plotting training history...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    print("Plotting complete.")


def evaluate_model(model, X_test, y_test, class_names):
    """Evaluates the trained model on the test set.

    Args:
        model (keras.Model): The trained Keras model.
        X_test (np.array): The test data.
        y_test (np.array): The test labels (one-hot encoded).
        class_names (list): A list of class names for the classification report.
    """
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1) # Convert one-hot to class labels


    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_classes))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    print("Evaluation complete.")



def trigger_defibrillator():
    """
    Simulates triggering a defibrillator.  This is a placeholder function.
    In a real application, this would interact with hardware.
    """
    print("CRITICAL: Ventricular Fibrillation or Ventricular Tachycardia DETECTED!")
    print("Initiating Defibrillation Sequence...")
    # In a real-world scenario, you would interact with the defibrillator hardware here.
    # This is where the code to control the defibrillator's charging and shock delivery
    # would reside.  This would likely involve serial communication or other device-specific protocols.
    print("Defibrillation Shock Delivered.")

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
    else:
        print("No immediate action required.")
    print("Prediction and trigger check complete.")



# --- Main Execution ---

if __name__ == "__main__":
    print("Starting main execution...")
    # Load and preprocess the data
    X, y = load_data(DATABASE_PATH, ANNOTATION_MAPPING, WINDOW_SIZE, SAMPLE_RATE, NUM_CLASSES)

    # Split into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=np.argmax(y, axis=1))  # Stratify to maintain class balance
    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Build the model
    input_shape = (WINDOW_SIZE,)
    model = build_model(input_shape, NUM_CLASSES)
    model.summary() # Display model architecture

    # Train the model
    print("Training the model...")
    # Define the checkpoint callback
    checkpoint_filepath = 'best_ecg_model.keras'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False, # Save the full model
        monitor='val_accuracy',  # Monitor validation accuracy
        mode='max',              # Save the model with the maximum validation accuracy
        save_best_only=True,     # Only save the best model
        verbose=1)

    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=0.1,
                        verbose=1,
                        callbacks=[model_checkpoint_callback]) # Add the callback here
    print("Model training complete.")

    # Load the best model saved by the checkpoint
    print(f"Loading the best model from {checkpoint_filepath}...")
    model = keras.models.load_model(checkpoint_filepath)

    # Plot training history
    plot_history(history)

    # Evaluate the model
    class_names = ['Normal', 'VF', 'VT']
    evaluate_model(model, X_test, y_test, class_names)

    # Example usage: Simulate receiving a new ECG segment and making a prediction
    # Create a sample ECG segment (replace with your actual ECG data)
    print("Simulating prediction on a sample ECG segment...")
    sample_ecg_segment = np.random.rand(WINDOW_SIZE)  # Dummy ECG segment
    predict_and_trigger(model, sample_ecg_segment)


    print("Finished main execution.")