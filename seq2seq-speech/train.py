import numpy as np
import os
import tensorflow as tf
from model import build_seq2seq, MAX_AUDIO_LEN, AUDIO_FEATURES
from utils import VOCAB_SIZE, MAX_TEXT_LEN, text_to_indices, CHARS

# ── Sample training data ──────────────────────────────
SENTENCES = [
    "hello world", "good morning", "how are you", "my name is alex",
    "the weather is nice", "i love deep learning", "speech to text",
    "neural network model", "sequence to sequence", "encoder decoder",
    "open the door", "turn on the light", "what time is it",
    "play some music", "call my friend", "set an alarm",
    "take a picture", "send a message", "read my email",
    "navigate to home", "increase the volume", "stop the music",
    "good night everyone", "see you tomorrow", "thank you very much",
]

def make_fake_audio(n):
    """Simulate MFCC features (in real app use librosa)."""
    return np.random.randn(n, MAX_AUDIO_LEN, AUDIO_FEATURES).astype(np.float32)

def encode_text(text):
    indices = text_to_indices(text)
    indices = indices[:MAX_TEXT_LEN]
    return indices

def prepare_data(sentences):
    n = len(sentences)
    audio = make_fake_audio(n)

    dec_in  = np.zeros((n, MAX_TEXT_LEN, VOCAB_SIZE), dtype=np.float32)
    dec_out = np.zeros((n, MAX_TEXT_LEN, VOCAB_SIZE), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        indices = encode_text(sentence)
        for t, idx in enumerate(indices):
            if t < MAX_TEXT_LEN:
                dec_in[i, t, idx] = 1.0
            if t > 0 and t <= MAX_TEXT_LEN:
                dec_out[i, t-1, idx] = 1.0

    return audio, dec_in, dec_out

print("Preparing training data...")
audio, dec_in, dec_out = prepare_data(SENTENCES * 20)  # repeat for more samples

print(f"Audio shape: {audio.shape}")
print(f"Decoder input shape: {dec_in.shape}")
print("Training model...")

model = build_seq2seq()
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
]

model.fit(
    [audio, dec_in], dec_out,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=callbacks
)

os.makedirs('model', exist_ok=True)
model.save('model/seq2seq_model.keras')
print("Model saved!")