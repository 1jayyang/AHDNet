

# Adenoid Facies Diagnosis Using Deep Learning

This repository contains the implementation of a deep learning–based system for **adenoid facies diagnosis**. The project aims to explore the potential of **artificial intelligence in facial disease detection and diagnosis**, providing an efficient and accessible auxiliary diagnostic tool.

## 🚀 Features

* **AI-driven facial diagnosis:** Detects adenoid facies from facial images using a convolutional neural network.
* **Simple training pipeline:** Easily replace the dataset with your own facial images to train a new model.
* **Lightweight inference:** Supports quick testing and deployment on personal or cloud platforms.
* **Online testing platform:** Users can upload their facial photos and receive instant diagnostic results.

## 🧠 Training

To train the model, simply run the following script:

```bash
python circle_train_face_detect.py
```

Replace the dataset path in the script with your own dataset.

---

## 🔍 Inference

To perform inference with a trained model, execute:

```bash
python predict.py
```

---

## 🌐 Online Demo

We have deployed an **interactive web-based diagnostic platform** for public use.
You can directly upload your photo to test the model here:
👉 **[Adenoid Facies AI Diagnosis – Online Demo](https://huggingface.co/spaces/jianligao4/ADE3)**

This online tool is designed as a **portable AI-assisted diagnostic system**, making adenoid facies screening accessible to everyone.

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch ≥ 1.10
* torchvision
* numpy, Pillow, gradio (for web deployment)

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## 📄 License

This repository is released for **research and educational purposes only**.
It is intended solely to explore the applications of artificial intelligence in medical facial analysis and **should not be used for clinical diagnosis** without proper validation and regulatory approval.

