from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

model = load_model("house_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET","POST"])
def index():

    prediction = None

    if request.method == "POST":

        try:
            data = [float(x) for x in request.form.values()]
            data = scaler.transform([data])

            price = model.predict(data)[0][0]
            prediction = round(price * 100000,2)

        except:
            prediction = "Invalid input"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)