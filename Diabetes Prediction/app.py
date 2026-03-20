from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

model = load_model("diabetes_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET","POST"])
def index():

    prediction = None

    if request.method == "POST":

        try:
            data = [
                float(request.form['preg']),
                float(request.form['glucose']),
                float(request.form['bp']),
                float(request.form['skin']),
                float(request.form['insulin']),
                float(request.form['bmi']),
                float(request.form['pedigree']),
                float(request.form['age'])
            ]

            data = scaler.transform([data])

            result = model.predict(data)[0][0]

            confidence = result * 100

            if result > 0.5:
                prediction = f"Diabetic ({confidence:.2f}%)"
            else:
                prediction = f"Not Diabetic ({100-confidence:.2f}%)"

        except Exception as e:
            print("ERROR:", e)
            prediction = "Invalid Input"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)