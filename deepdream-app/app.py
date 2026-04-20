from flask import Flask, request, render_template, send_file, jsonify
import os
from deep_dream import run_deep_dream

app = Flask(__name__)

os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dream', methods=['POST'])
def dream():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    input_path = 'uploads/input.jpg'
    output_path = 'outputs/dream_output.jpg'
    file.save(input_path)

    run_deep_dream(input_path, output_path)

    return jsonify({'status': 'success'})

@app.route('/download')
def download():
    return send_file('outputs/dream_output.jpg', as_attachment=True, download_name='deep_dream_result.jpg')

if __name__ == '__main__':
    app.run(debug=True)