from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# Load model & scaler
model = load_model("house_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":
        try:
            # FIXED: Explicit feature order
            data = [
                float(request.form["income"]),
                float(request.form["age"]),
                float(request.form["rooms"]),
                float(request.form["bedrooms"]),
                float(request.form["population"]),
                float(request.form["occupancy"]),
                float(request.form["lat"]),
                float(request.form["lon"])
            ]

            # Scale input
            data_scaled = scaler.transform([data])

            # Predict
            price = model.predict(data_scaled)[0][0]

            # Convert to USD
            prediction = f"{price * 100000:,.2f}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)