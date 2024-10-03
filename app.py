import torch
from flask import Flask, request, jsonify
from transformers import pipeline
import io

app = Flask(__name__)

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

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        file_content = file.read()
        file_stream = io.BytesIO(file_content)
        
        try:
            text = pipe(file_stream, batch_size=BATCH_SIZE, return_timestamps=True)["text"]
            return jsonify({"transcript": text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    print("Server is running on port 5000")
    app.run(debug=True, port=5000)