# Computer Vision with Modern Deep Learning

## 🧩 Overview

This assignment demonstrates a broad spectrum of vision-related deep learning techniques using Colab notebooks. It is divided into four comprehensive parts:

- **Part 1**: Supervised Contrastive Learning vs Softmax Baseline
- **Part 2**: Transfer Learning on Images, Audio, Video, and NLP
- **Part 3**: Zero-Shot Learning and TFHub SOTA Models
- **Part 4**: Classic Vision Benchmarks + X-ray and 3D CT Classification

All code is modular, optimized for Colab, and leverages lightweight pretrained models for quick execution.

---


- 📓 [Colab 1:supervised contrastive learning (CIFAR-10)](https://colab.research.google.com/drive/1fj-kRdn6pWkbOqhMg8eoFgupmZ4EfFIi?usp=sharing)
- 📓 [Colab 2: Transfer Learning - Images, Audio, Video, Text](https://colab.research.google.com/drive/1y4J8I2e2g4ELtOZyXPpmCo4d_DEGMuAY?usp=sharing)
- 📓 [Colab 3: CLIP Zero-Shot + TFHub (Flowers)](https://colab.research.google.com/drive/1AmDiFtrEvnGbrI8E0sttueoeKXuTejpp?usp=sharing)
- 📓 [Colab 4: MNIST + Fashion + CIFAR10 – EfficientNet, BiT, SOTA](https://colab.research.google.com/drive/1or31jYtPtWQtRTp7eTkkoFDImcjPaC5S?usp=sharing)
- 📓 [Colab 5: X-ray + 3D CT Scan Classification](https://colab.research.google.com/drive/1hVZ6YwC-fhpQ_qB7ZLYXnUH4GhFOx9VY?usp=sharing)
- 🎥 [YouTube Video Explanation](https://www.youtube.com/watch?v=your-video-link)

---

## 🧠 Breakdown of Components

### 1️⃣ SupCon vs Softmax – CIFAR-10
- Implements supervised contrastive loss manually (no Addons)
- Dual-augmented pipeline for SupCon
- Final classifier (linear probe) over frozen encoder
- 📊 Includes accuracy curves, t-SNE projection, bar comparison

### 2️⃣ Transfer Learning on Modalities
- 🖼️ MobileNetV2 on Cats vs Dogs (feature + fine-tune)
- 🔊 YAMNet embeddings from audio (binary classifier)
- 🎥 I3D TFHub model with dummy videos (Logistic Regression)
- 📝 Universal Sentence Encoder for text classification

### 3️⃣ Zero-Shot + TFHub Fine-Tuning
- CLIP (ViT-B-32) for prompt-based image classification
- EfficientNetLite0 via `hub.load()` with mixed precision
- Dataset: tf_flowers
- Output: Accuracy chart and prediction validation

### 4️⃣ Vision Benchmarks – MNIST, Fashion, CIFAR-10
- Transfer learning with EfficientNetB0 and BiT (ResNet50V2)
- SOTA models: MLP-Mixer, ConvNeXt-lite
- Visuals: Training loss, accuracy, confusion matrix

### 5️⃣ Medical Imaging – X-ray + CT Scans
- 🩻 Chest X-ray Pneumonia detection (Kaggle dataset via kagglehub)
- 🧠 3D CNN on simulated CT volumes (64x64x64)
- Charts: Accuracy/loss and confusion matrix
- Classification report for pneumonia

---

## 🧪 Tools & Tech

- `TensorFlow 2.18`, `Keras`, `TFHub`, `open-clip`, `sklearn`, `Matplotlib`
- All models optimized for Colab (5–10 minute runtime each)

---

## 📌 Notes

- Encoders reused across tasks where possible (e.g. SupCon → classifier)
- Efficient use of feature extractors for low-resource inference
- Plots and evaluation included in each notebook
