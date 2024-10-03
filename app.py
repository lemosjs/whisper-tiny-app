import torch
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
from transformers import pipeline
from pydub import AudioSegment
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for all routes


MODEL_NAME = "openai/whisper-tiny"
BATCH_SIZE = 8
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

MAX_DURATION_SECONDS = 15
MAX_FILE_SIZE_MB = 10

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return jsonify({"error": f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"}), 400

    # Save file temporarily
    temp_path = "temp_audio"
    file.save(temp_path)
    
    try:
        # Check audio duration
        audio = AudioSegment.from_file(temp_path)
        duration_seconds = len(audio) / 1000
        if duration_seconds > MAX_DURATION_SECONDS:
            return jsonify({"error": f"Audio duration exceeds {MAX_DURATION_SECONDS} seconds limit"}), 400

        # Process the file
        with open(temp_path, "rb") as audio_file:
            result = pipe(audio_file, batch_size=BATCH_SIZE, return_timestamps=True)
        
        return jsonify({
            "transcript": result["text"],
            "chunks": result.get("chunks"),
            "duration": duration_seconds
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

port = os.getenv("PORT", 5000)

if __name__ == '__main__':
    print(f"Server is running on port {port}")
    app.run(host="0.0.0.0", port=port)