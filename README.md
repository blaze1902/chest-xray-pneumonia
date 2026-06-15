# 🫁 Chest X-Ray Pneumonia Detection System

A Deep Learning based web application that detects 
Pneumonia from Chest X-Ray images using PyTorch 
and Streamlit.

## 🔍 About The Project

Pneumonia is a serious lung infection affecting 
millions worldwide. This project uses Transfer 
Learning with ResNet18 to classify Chest X-Ray 
images as NORMAL or PNEUMONIA with 93.75% accuracy.

---

## ✨ Features

- Upload any Chest X-Ray image (JPG/PNG)
- Instant NORMAL or PNEUMONIA prediction
- Confidence score display
- Clean and simple UI
- Deployed on Streamlit Cloud

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Deep Learning Framework |
| ResNet18 | Pretrained CNN Model |
| Streamlit | Web Application UI |
| Pillow | Image Processing |
| NumPy | Numerical Operations |

## project structure
chest-xray-pneumonia/
├── chest_xray/          # Dataset
│   ├── train/
│   ├── val/
│   └── test/
├── model.py             # Model Architecture
├── train.py             # Training Script
├── app.py               # Streamlit Web App
├── requirements.txt     # Dependencies
└── model_weights.pth    # Saved Model

---

## 🚀 How To Run Locally

### 1. Clone the repository
git clone https://github.com/blaze1902/chest-xray-pneumonia.git
cd chest-xray-pneumonia

### 2. Install dependencies
pip install -r requirements.txt

### 3. Train the model (optional)
python train.py

### 4. Run the app
streamlit run app.py

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Model | ResNet18 (Transfer Learning) |
| Training Epochs | 10 |
| Train Accuracy | 93.44% |
| Best Val Accuracy | 93.75% |
| Classes | NORMAL, PNEUMONIA |

---

## 📂 Dataset

Dataset used: Chest X-Ray Images (Pneumonia)
Source: Kaggle
Link: kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | 1341 | 3875 | 5216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

---

## ⚠️ Disclaimer

This tool is for educational purposes only.
Always consult a qualified doctor for 
medical diagnosis and advice.

---

## 👨‍💻 Author

**Aditya Rao**
GitHub: github.com/blaze1902




---

## 📁 Project Structure
