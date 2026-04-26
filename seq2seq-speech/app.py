from flask import Flask, request, render_template, jsonify
import os
import speech_recognition as sr
import io
import wave
import struct

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'error': 'No audio uploaded'})

        file = request.files['audio']
        wav_path = 'uploads/input.wav'

        # Read raw bytes
        audio_bytes = file.read()

        # Write as WAV directly
        with open(wav_path, 'wb') as f:
            f.write(audio_bytes)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return jsonify({'status': 'success', 'text': text})

    except sr.UnknownValueError:
        return jsonify({'status': 'error', 'error': 'Could not understand. Speak clearly.'})
    except sr.RequestError as e:
        return jsonify({'status': 'error', 'error': f'Service error: {str(e)}'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)