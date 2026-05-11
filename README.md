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

## 🎯 Use Cases

- **Healthcare**: Disease prediction, medical image enhancement, clinical text analysis
- **Finance**: Real estate valuation, price forecasting
- **Accessibility**: Handwriting recognition, speech-to-text
- **Creative**: Artistic image generation, style transfer
- **Research**: Deep learning architecture comparison, model evaluation

---

## � Project Structure

```
NeuroHub/
├── House Price Prediction/
│   ├── app.py                       # Flask application
│   ├── house_model.keras            # Pre-trained model
│   ├── scaler.pkl                   # Feature scaler
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   └── index.html
│   └── uploads/
│
├── Handwritten Digit Classification/
│   ├── app.py                       # Flask application
│   ├── mnist_model.h5               # Pre-trained model
│   ├── scaler.pkl                   # Feature scaler
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   └── index.html
│   └── uploads/
│
├── Diabetes Prediction/
│   ├── app.py                       # Flask application
│   ├── diabetes_model.h5            # Pre-trained model
│   ├── scaler.pkl                   # Feature scaler
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   └── index.html
│   └── uploads/
│
├── deepdream-app/
│   ├── app.py                       # Flask application
│   ├── deep_dream.py                # Deep Dream implementation
│   ├── templates/
│   │   └── index.html
│   ├── outputs/                     # Generated images
│   └── uploads/                     # Input images
│
├── denoising_autoencoder/
│   ├── app.py                       # Flask application
│   ├── dae_model.py                 # Model architecture
│   ├── train.py                     # Training script
│   ├── model/
│   │   └── dae_model.keras          # Pre-trained model
│   ├── dataset/                     # BraTS data (2018, 2019, 2020)
│   ├── templates/
│   │   └── index.html
│   ├── outputs/                     # Denoised images
│   └── uploads/                     # Input MRI scans
│
├── medical-bert/
│   ├── app.py                       # Flask application
│   ├── train.py                     # Training script
│   ├── data/
│   │   └── medical_data.py          # Dataset loader
│   ├── model/
│   │   ├── config.json              # Model config
│   │   ├── model.safetensors        # Pre-trained weights
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   └── templates/
│       └── index.html
│
├── seq2seq-speech/
│   ├── app.py                       # Flask application
│   ├── model.py                     # Model architecture
│   ├── train.py                     # Training script
│   ├── utils.py                     # Utility functions
│   ├── model/
│   │   └── seq2seq_model.keras      # Pre-trained model
│   ├── templates/
│   │   └── index.html
│   ├── outputs/                     # Transcription outputs
│   └── uploads/                     # Audio files
│
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 💾 Pre-trained Model Checkpoints

All models come with pre-trained weights:

| Model | File | Format | Size | Framework |
|---|---|---|---|---|
| House Price | `house_model.keras` | Keras | ~50KB | TensorFlow |
| MNIST | `mnist_model.h5` | H5 | ~100KB | TensorFlow |
| Diabetes | `diabetes_model.h5` | H5 | ~50KB | TensorFlow |
| Deep Dream | Built-in InceptionV3 | Pre-loaded | ~90MB | TensorFlow |
| DAE | `dae_model.keras` | Keras | ~200KB | TensorFlow |
| Medical BERT | `model.safetensors` | SafeTensors | ~440MB | PyTorch |
| Seq2Seq | `seq2seq_model.keras` | Keras | ~150KB | TensorFlow |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd NeuroHub
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- TensorFlow/Keras
- PyTorch & Transformers (for Medical BERT)
- Flask (web framework)
- NumPy, Pandas, Scikit-learn
- Pillow, librosa, matplotlib
- Joblib

---

## 🎯 Running Each Model

### 1. 🏠 House Price Prediction

```bash
cd "House Price Prediction"
python app.py
```
- Access: `http://localhost:5000`
- Input: Housing features (8 parameters)
- Output: Predicted house price in USD

### 2. ✍️ Handwritten Digit Classification

```bash
cd "Handwritten Digit Classification"
python app.py
```
- Access: `http://localhost:5000`
- Input: Image upload (MNIST format)
- Output: Digit prediction (0-9) with confidence

### 3. 🩺 Diabetes Prediction

```bash
cd "Diabetes Prediction"
python app.py
```
- Access: `http://localhost:5000`
- Input: Medical parameters (8 fields)
- Output: Diabetes risk prediction with probability

### 4. 🎨 Deep Dream Art Generator

```bash
cd deepdream-app
python app.py
```
- Access: `http://localhost:5000`
- Input: Any image file
- Output: Psychedelic artistic transformation
- Note: First run may take 5-10 seconds for InceptionV3 model loading

### 5. 🧠 Denoising Autoencoder

```bash
cd denoising_autoencoder
python app.py
```
- Access: `http://localhost:5000`
- Input: Noisy or degraded brain MRI images
- Output: Denoised medical image
- Dataset: BraTS (2018, 2019, 2020) - optional for training

### 6. 🏥 Medical BERT Text Classifier

```bash
cd medical-bert
python app.py
```
- Access: `http://localhost:5000`
- Input: Medical text or disease description
- Output: Disease category classification

### 7. 🎤 Seq2Seq Speech-to-Text

```bash
cd seq2seq-speech
python app.py
```
- Access: `http://localhost:5000`
- Input: Audio file (WAV/MP3)
- Output: Transcribed text
- Uses MFCC feature extraction

---

## ⚡ Quick Start (All Models)

```bash
# Install dependencies once
pip install -r requirements.txt

# Run any model
cd <model-directory>
python app.py

# Open browser: http://localhost:5000
```

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

Contributions are welcome! Areas for enhancement:
- Model accuracy improvements
- Additional dataset support
- UI/UX enhancements
- Performance optimization
- New model architectures
- Documentation improvements

Please submit pull requests with detailed descriptions of changes.

---

## 👨‍💻 Authors & Credits

**Project Creator**: R Murali  
**Institution**: Chaitanya Bharathi Institute of Technology  
**Qualification**: B.E Information Technology

Developed as part of deep learning research and educational initiatives.

---

## 📝 License

MIT License

Copyright (c) 2024-2026 NeuroHub Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

**Disclaimer**: These models are for educational and research purposes. Use responsibly in production environments.

---

## 📞 Contact & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review model-specific README files in each project folder

---

**Last Updated**: May 2026  
**Total Models**: 7 | **Status**: All Production-Ready ✅  
**Repository**: [NeuroHub](https://github.com/)
