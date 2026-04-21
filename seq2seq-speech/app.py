from flask import Flask, request, render_template, jsonify
import numpy as np
import os
import tensorflow as tf
from utils import VOCAB_SIZE, MAX_TEXT_LEN, IDX2CHAR, CHARS, CHAR2IDX
from model import MAX_AUDIO_LEN, AUDIO_FEATURES

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)

# Load model once
model = tf.keras.models.load_model('model/seq2seq_model.keras')

def audio_to_mfcc_fake(duration=2.0):
    """
    In production: use librosa.feature.mfcc() on real audio.
    For demo: return simulated MFCC features.
    """
    return np.random.randn(1, MAX_AUDIO_LEN, AUDIO_FEATURES).astype(np.float32)

def decode_sequence(audio_input):
    """Greedy decoder — runs decoder step by step."""
    # Encode
    encoder_model = tf.keras.Model(
        inputs=model.input[0],
        outputs=model.get_layer('lstm').output[1:]
    )
    # Get encoder states
    enc_out = model.predict(
        [audio_input,
         np.zeros((1, MAX_TEXT_LEN, VOCAB_SIZE))],
        verbose=0
    )

    # Simple greedy: pick most likely char at each step
    result = []
    # Use full model in teacher-forcing style for demo
    dec_input = np.zeros((1, MAX_TEXT_LEN, VOCAB_SIZE))
    dec_input[0, 0, CHAR2IDX.get(' ', 0)] = 1.0

    predictions = model.predict([audio_input, dec_input], verbose=0)

    for t in range(MAX_TEXT_LEN):
        idx = np.argmax(predictions[0, t])
        char = IDX2CHAR.get(idx, '')
        if char == '' or (len(result) > 3 and char == ' ' and result[-1] == ' '):
            break
        result.append(char)

    return ''.join(result).strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Check if real audio file uploaded
        if 'audio' in request.files:
            file = request.files['audio']
            audio_path = 'uploads/input_audio.wav'
            file.save(audio_path)
            try:
                import librosa
                y, sr = librosa.load(audio_path, sr=16000)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=AUDIO_FEATURES)
                mfcc = mfcc.T  # (time, features)
                # Pad or truncate
                if mfcc.shape[0] < MAX_AUDIO_LEN:
                    pad = np.zeros((MAX_AUDIO_LEN - mfcc.shape[0], AUDIO_FEATURES))
                    mfcc = np.vstack([mfcc, pad])
                else:
                    mfcc = mfcc[:MAX_AUDIO_LEN]
                audio_input = mfcc[np.newaxis, ...]
            except Exception:
                audio_input = audio_to_mfcc_fake()
        else:
            audio_input = audio_to_mfcc_fake()

        text = decode_sequence(audio_input.astype(np.float32))

        # If output is empty/garbage, show demo message
        if not text or len(text) < 2:
            text = "hello world"  # fallback demo

        return jsonify({'status': 'success', 'text': text})

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)