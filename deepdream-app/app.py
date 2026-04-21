from flask import Flask, request, render_template, send_file, jsonify
import os
import threading
from deep_dream import run_deep_dream

app = Flask(__name__)

os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

status = {"state": "idle", "error": ""}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dream', methods=['POST'])
def dream():
    global status
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    input_path = 'uploads/input.jpg'
    output_path = 'outputs/dream_output.jpg'
    file.save(input_path)

    status = {"state": "processing", "error": ""}

    def run():
        global status
        try:
            run_deep_dream(input_path, output_path)
            status = {"state": "done", "error": ""}
        except Exception as e:
            status = {"state": "error", "error": str(e)}

    threading.Thread(target=run).start()
    return jsonify({'status': 'started'})

@app.route('/status')
def get_status():
    return jsonify(status)

@app.route('/download')
def download():
    return send_file('outputs/dream_output.jpg', as_attachment=True,
                     download_name='deep_dream_result.jpg')

if __name__ == '__main__':
    app.run(debug=False)