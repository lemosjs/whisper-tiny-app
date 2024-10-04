import os
import tempfile
import ffmpeg
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import WhisperProcessor, WhisperForConditionalGeneration

app = Flask(__name__)
CORS(app)
# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

@app.route('/')
def index():
    return send_from_directory('./', 'index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        # Convert to WAV using ffmpeg-python
        output_path = temp_audio_path.replace('.mp3', '.wav')
        stream = ffmpeg.input(temp_audio_path)
        stream = ffmpeg.output(stream, output_path, ar='16000', ac=1, acodec='pcm_s16le')
        ffmpeg.run(stream, overwrite_output=True)

        # Read the WAV file
        audio_data, _ = (
            ffmpeg
            .input(output_path)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar='16000')
            .run(capture_stdout=True, capture_stderr=True)
        )
        samples = np.frombuffer(audio_data, np.float32).flatten()

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

        # Clean up temporary files
        os.unlink(temp_audio_path)
        os.unlink(output_path)

        return jsonify({'transcript': transcription[0]})
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

port = os.environ.get("PORT", 3003)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)