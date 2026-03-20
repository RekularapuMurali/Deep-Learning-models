# 🧠 Deep Learning Dashboard (MLP Based Web Applications)

An AI-powered web application that demonstrates multiple deep learning models using **Multi-Layer Perceptron (MLP)** for different machine learning tasks, including regression, multi-class classification, and binary classification.

---

## 🚀 Features

* Interactive web-based interface for all models
* Real-time predictions using trained neural networks
* Input validation and error handling
* Visualization of inputs using charts
* Modular structure with separate projects

---

## 📌 Projects Included

### 🏠 House Price Prediction (Regression)

* Predicts house prices using the California Housing dataset
* Outputs price in USD
* Uses MLP regression model

---

### ✍️ Handwritten Digit Recognition (MNIST)

* Classifies handwritten digits (0–9)
* Upload image → Predict digit
* Uses MLP multi-class classifier

---

### 🩺 Diabetes Prediction (Binary Classification)

* Predicts whether a person is diabetic or not
* Outputs result with confidence score
* Uses MLP binary classifier

---

## 🧰 Technologies Used

* **Backend**: Python, Flask
* **Frontend**: HTML, CSS, JavaScript
* **AI/ML**: TensorFlow, Keras, scikit-learn
* **Visualization**: Chart.js
* **Libraries**: NumPy, Pandas, Joblib, Pillow

---

## 📁 Project Structure

```
Deep-Learning-Dashboard/
│
├── house_price/
│   ├── app.py
│   ├── model/
│   │   ├── house_model.h5
│   │   └── scaler.pkl
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── digit_recognition/
│   ├── app.py
│   ├── model/
│   │   └── mnist_model.h5
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── diabetes_prediction/
│   ├── app.py
│   ├── model/
│   │   ├── diabetes_model.h5
│   │   └── scaler.pkl
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd Deep-Learning-Dashboard
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

---

## ▶️ Running the Projects

### 🏠 House Price Prediction

```
cd house_price
python app.py
```

---

### ✍️ Digit Recognition

```
cd digit_recognition
python app.py
```

---

### 🩺 Diabetes Prediction

```
cd diabetes_prediction
python app.py
```

---

## 🌐 Access the Application

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 📊 Usage

### House Price Prediction

* Enter housing features
* Click **Predict Price**
* View predicted house price

---

### Digit Recognition

* Upload a handwritten digit image
* Click **Predict Digit**
* View predicted digit

---

### Diabetes Prediction

* Enter medical details
* Click **Predict**
* View result with confidence score

---

## 🎯 Learning Outcomes

* Implementation of MLP for different problem types
* Integration of deep learning models with Flask
* Data preprocessing and feature scaling
* Building end-to-end AI web applications

---

## 📌 Future Improvements

* Combine all models into a single dashboard
* Add real-time graph updates
* Deploy on cloud platforms
* Improve UI/UX with animations

---

## 👨‍💻 Author

**R Murali**
B.E Information Technology
Chaitanya Bharathi Institute of Technology

---

## 📜 License

This project is for academic and educational purposes.
