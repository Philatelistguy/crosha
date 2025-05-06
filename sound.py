import pvorca
import sounddevice as sd
import numpy as np
from flask import Flask, request, jsonify
app = Flask(__name__)

# Initialise ORCA
orca = pvorca.create(access_key="HlPxVqdBOLNqVBxscWXQKjgZnQ40EIu4Rq8/PMH+sfB8ZhkrXFo6ew==")
@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text'"}), 400

    text = data['text']

    try:
        # Use ORCA to generate speech
        audio_data, _ = orca.synthesize(text)

        # Convert to NumPy array
        audio_np = np.array(audio_data, dtype=np.int16)

        # Play audio at 22050 Hz
        sd.play(audio_np, samplerate=22050)
        sd.wait()

        return jsonify({"status": "spoken"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)	
    