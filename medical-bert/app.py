from flask import Flask, request, render_template, jsonify
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from data.medical_data import IDX2LABEL, LABELS

app = Flask(__name__)

SAVE_DIR = 'model'
MAX_LEN  = 64
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading Medical-BERT...")
tokenizer = BertTokenizer.from_pretrained(SAVE_DIR)
model     = BertForSequenceClassification.from_pretrained(SAVE_DIR)
model.to(device)
model.eval()
print("Ready!")

def predict(text):
    inputs = tokenizer(text, max_length=MAX_LEN, padding='max_length',
                       truncation=True, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs     = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    top_idx   = int(np.argmax(probs))
    label     = IDX2LABEL[top_idx]
    confidence = float(probs[top_idx]) * 100

    all_scores = [
        {'label': IDX2LABEL[i], 'score': round(float(probs[i]) * 100, 1)}
        for i in range(len(LABELS))
    ]
    all_scores.sort(key=lambda x: x['score'], reverse=True)
    return label, confidence, all_scores

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text or len(text) < 5:
            return jsonify({'status': 'error', 'error': 'Please enter valid symptoms'})

        label, confidence, all_scores = predict(text)
        return jsonify({
            'status':     'success',
            'label':      label,
            'confidence': round(confidence, 1),
            'all_scores': all_scores
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)