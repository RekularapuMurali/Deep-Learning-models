import string

# Character vocabulary
CHARS = [' '] + list(string.ascii_lowercase) + ["'"]
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
MAX_TEXT_LEN = 50

def text_to_indices(text):
    text = text.lower()
    return [CHAR2IDX.get(c, 0) for c in text if c in CHAR2IDX]

def indices_to_text(indices):
    return ''.join([IDX2CHAR.get(i, '') for i in indices])