import tensorflow as tf
from tensorflow.keras import layers, models
from utils import VOCAB_SIZE, MAX_TEXT_LEN

AUDIO_FEATURES = 13   # MFCC features
MAX_AUDIO_LEN  = 100  # timesteps

def build_seq2seq():
    # ── Encoder ──────────────────────────────────────
    enc_input = layers.Input(shape=(MAX_AUDIO_LEN, AUDIO_FEATURES), name='encoder_input')
    enc_lstm1, h1, c1 = layers.LSTM(256, return_sequences=True, return_state=True)(enc_input)
    _, h2, c2          = layers.LSTM(256, return_state=True)(enc_lstm1)
    enc_states = [h2, c2]

    # ── Decoder ──────────────────────────────────────
    dec_input  = layers.Input(shape=(None, VOCAB_SIZE), name='decoder_input')
    dec_lstm   = layers.LSTM(256, return_sequences=True, return_state=True)
    dec_out, _, _ = dec_lstm(dec_input, initial_state=enc_states)
    dec_dense  = layers.Dense(VOCAB_SIZE, activation='softmax')
    dec_output = dec_dense(dec_out)

    model = models.Model([enc_input, dec_input], dec_output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model