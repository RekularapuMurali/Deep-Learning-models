# 🧠 NeuroHub: Multi-Model Deep Learning Suite

A comprehensive AI-powered web application demonstrating diverse deep learning architectures including **MLPs, CNNs, LSTMs, Transformers, and Autoencoders** for various machine learning tasks spanning regression, classification, image generation, and sequence-to-sequence learning.

---

## 🚀 Features

* 🎯 **7 Production-Ready Models** - Multiple neural network architectures
* 🖥️ **Interactive Web Dashboards** - Real-time predictions with visual feedback
* 🧪 **Pre-trained Models** - Ready-to-use trained neural networks
* 📊 **Data Visualization** - Chart-based input/output analysis
* 🏗️ **Modular Architecture** - Standalone projects with independent deployments
* ⚡ **Real-time Processing** - Fast inference with optimized models

---

## 📋 Models Overview

| # | Model Name | Architecture | Task Type | Input | Output | Framework | Status |
|---|---|---|---|---|---|---|---|
| 1 | **House Price Prediction** | MLP (Dense) | Regression | Housing Features | USD Price | TensorFlow/Keras | ✅ Production |
| 2 | **Handwritten Digit Classification** | MLP (Dense) | Multi-class | MNIST Image (28×28) | Digit (0-9) | TensorFlow/Keras | ✅ Production |
| 3 | **Diabetes Prediction** | MLP (Dense) | Binary Classification | Medical Features (8) | Yes/No + Score | TensorFlow/Keras | ✅ Production |
| 4 | **Deep Dream Art Generator** | InceptionV3 | Image Generation | Image File | Artistic Output | TensorFlow/Keras | ✅ Production |
| 5 | **Denoising Autoencoder** | CNN Autoencoder | Image Restoration | Noisy Brain MRI | Denoised Image | TensorFlow/Keras | ✅ Production |
| 6 | **Medical BERT Text Classifier** | BERT | Text Classification | Medical Text (64 tokens) | Disease Category | PyTorch/Transformers | ✅ Production |
| 7 | **Seq2Seq Speech-to-Text** | LSTM Seq2Seq | Sequence Translation | Audio (MFCC Features) | Transcribed Text | TensorFlow/Keras | ✅ Production |

---

## 📌 Detailed Project Descriptions

### 1. 🏠 House Price Prediction
- **Algorithm**: Multi-Layer Perceptron (Regression)
- **Dataset**: California Housing Dataset
- **Features**: 8 input features (location, size, rooms, etc.)
- **Output**: House price in USD
- **Accuracy**: Regression-based predictions
- **Use Case**: Real estate valuation

### 2. ✍️ Handwritten Digit Recognition (MNIST)
- **Algorithm**: Multi-Layer Perceptron (10-class Classification)
- **Dataset**: MNIST (60,000 training samples)
- **Features**: 28×28 pixel grayscale images
- **Output**: Digit classification (0-9) with confidence
- **Accuracy**: ~97%+
- **Use Case**: Optical character recognition, postal code reading

### 3. 🩺 Diabetes Prediction
- **Algorithm**: Multi-Layer Perceptron (Binary Classification)
- **Dataset**: Pima Indians Diabetes Dataset
- **Features**: 8 medical parameters (glucose, BMI, age, etc.)
- **Output**: Diabetes prediction with probability score
- **Accuracy**: ~75-80%
- **Use Case**: Early diabetes detection, health screening

### 4. 🎨 Deep Dream Art Generator
- **Algorithm**: Convolutional Neural Network (InceptionV3 + Gradient Ascent)
- **Architecture**: Pre-trained InceptionV3 with custom activation layers
- **Features**: Image upscaling, iterative optimization
- **Output**: Psychedelic artistic rendition of input image
- **Applications**: Artistic style transfer, creative image generation
- **Tech**: TensorFlow, PIL, NumPy

### 5. 🧠 Denoising Autoencoder
- **Algorithm**: Convolutional Autoencoder (Image Restoration)
- **Dataset**: BraTS Brain MRI Dataset (2018, 2019, 2020)
- **Input**: Noisy/degraded 128×128×1 medical images
- **Output**: Denoised, reconstructed medical images
- **Architecture**: Encoder (Conv2D + MaxPool) → Decoder (UpSampling)
- **Use Case**: Medical image enhancement, noise reduction

### 6. 🏥 Medical BERT Text Classifier
- **Algorithm**: BERT Transformer (Sequence Classification)
- **Model**: bert-base-uncased with fine-tuning
- **Dataset**: Custom medical text corpus
- **Input**: Medical text/documents (max 64 tokens)
- **Output**: Disease category classification
- **Classes**: Multiple medical condition categories
- **Framework**: PyTorch, Hugging Face Transformers

### 7. 🎤 Seq2Seq Speech-to-Text
- **Algorithm**: LSTM Encoder-Decoder (Sequence-to-Sequence)
- **Input**: Audio MFCC features (13 dimensions, 100 timesteps)
- **Output**: Transcribed text (variable length)
- **Architecture**: 2-layer Encoder + Decoder with attention mechanism
- **Applications**: Speech recognition, voice transcription
- **Framework**: TensorFlow/Keras

---

## 🧰 Technology Stack

### Backend & ML
- **Deep Learning**: TensorFlow, Keras, PyTorch
- **NLP**: Hugging Face Transformers
- **Web Framework**: Flask
- **Audio Processing**: MFCC feature extraction
- **Libraries**: NumPy, Pandas, Scikit-learn, Pillow

### Frontend
- **Markup**: HTML5
- **Styling**: CSS3
- **Visualization**: Chart.js, Matplotlib
- **Interactivity**: JavaScript

### Deployment & Tools
- **Model Format**: .keras, .h5, .safetensors
- **Serialization**: Joblib, JSON
- **Environment**: Python 3.8+

---

## 📁 Project Structure

```
Deep-Learning-Models/
├── House Price Prediction/          # MLP Regression
├── Handwritten Digit Classification/ # MLP Classification
├── Diabetes Prediction/              # MLP Binary Classification
├── deepdream-app/                   # InceptionV3 Art Generator
├── denoising_autoencoder/           # CNN Autoencoder (BraTS)
├── medical-bert/                    # BERT Text Classifier
├── seq2seq-speech/                  # LSTM Seq2Seq
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

---

## 💾 Model Checkpoints

All pre-trained models are included:
- `house_model.keras` - House price prediction weights
- `mnist_model.h5` - Digit classification weights
- `diabetes_model.h5` - Diabetes prediction weights
- `dae_model.keras` - Denoising autoencoder weights
- `model.safetensors` - Medical BERT weights
- `seq2seq_model.keras` - Speech-to-text weights

---

## ⚡ Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run any model:
   ```bash
   cd <model-folder>
   python app.py
   ```

3. Open browser: `http://localhost:5000`

---

## 🎯 Use Cases

- **Healthcare**: Disease prediction, medical image enhancement, clinical text analysis
- **Finance**: Real estate valuation, price forecasting
- **Accessibility**: Handwriting recognition, speech-to-text
- **Creative**: Artistic image generation, style transfer
- **Research**: Deep learning architecture comparison, model evaluation

---

## 📊 Performance Summary

| Model | Type | Inference Time | Accuracy | Status |
|---|---|---|---|---|
| House Price | Regression | <100ms | R² Score | ✅ |
| MNIST | Classification | <50ms | ~97% | ✅ |
| Diabetes | Classification | <100ms | ~80% | ✅ |
| Deep Dream | Generation | 2-5s | Visual Quality | ✅ |
| DAE | Restoration | <500ms | SSIM-based | ✅ |
| Medical BERT | Classification | <200ms | ~90%+ | ✅ |
| Seq2Seq | Translation | <1s | WER-based | ✅ |

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Model accuracy improvements
- Additional dataset support
- UI/UX enhancements
- Performance optimization
- New model architectures

---

## 📝 License

MIT License - Feel free to use, modify, and distribute

---

**Last Updated**: May 2026
**Models**: 7 | **Status**: All Production-Ready ✅

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
