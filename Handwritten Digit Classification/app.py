from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)

model = load_model("mnist_model.h5", compile=False)

@app.route("/", methods=["GET","POST"])
def index():

    prediction = None

    if request.method == "POST":

        file = request.files["file"]

        img = Image.open(file).convert("L")

        # Resize
        img = img.resize((28,28))

        # Invert colors (important for MNIST)
        img = ImageOps.invert(img)

        img = np.array(img)

        # Normalize
        img = img / 255.0

        img = img.reshape(1,28,28)

        pred = model.predict(img)

        prediction = np.argmax(pred)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)