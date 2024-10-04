from flask import Flask, request, jsonify, send_from_directory
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import os
from dotenv import load_dotenv
from pydub import AudioSegment
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

max_audio_length = 20 * 1000

@app.route('/')
def index():
    return send_from_directory('./', 'index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    
    try:
        # Read the audio file and convert to wav
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(16000).set_channels(1)

        if len(audio) > max_audio_length:
            return jsonify({'error': 'Audio too long'}), 400
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        # Normalize
        samples = samples / np.max(np.abs(samples))

        # Process the audio data
        input_features = processor(
            audio=samples, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features

        # Generate token ids
        predicted_ids = model.generate(input_features)

        # Decode token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return jsonify({'transcript': transcription[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
port = os.environ.get("PORT", 5000)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)