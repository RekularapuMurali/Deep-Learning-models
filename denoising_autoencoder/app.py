from flask import Flask, request, render_template, send_file, jsonify
import numpy as np
import os
import threading
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

model = tf.keras.models.load_model('model/dae_model.keras')  # changed from .h5
status = {"state": "idle", "error": ""}

def denoise_image(input_path, output_path):
    img = Image.open(input_path).convert('L')
    original_size = img.size  # save original size to restore later

    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = img_array[np.newaxis, ..., np.newaxis]  # (1,128,128,1)

    denoised = model.predict(img_array)
    denoised_img = (denoised[0, ..., 0] * 255).astype(np.uint8)

    # Restore to original size
    result = Image.fromarray(denoised_img).resize(original_size, Image.LANCZOS)
    result.save(output_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/denoise', methods=['POST'])
def denoise():
    global status
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    file = request.files['image']
    input_path = 'uploads/input.jpg'
    output_path = 'outputs/denoised_output.jpg'
    file.save(input_path)
    status = {"state": "processing", "error": ""}

    def run():
        global status
        try:
            denoise_image(input_path, output_path)
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
    return send_file('outputs/denoised_output.jpg', as_attachment=True,
                     download_name='denoised_result.jpg')

if __name__ == '__main__':
    app.run(debug=False)