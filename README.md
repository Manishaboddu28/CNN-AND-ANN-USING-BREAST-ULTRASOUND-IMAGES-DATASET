# Breast Cancer Classification using ANN and CNN

This project demonstrates the use of **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)** for classifying **breast ultrasound images** into categories such as **benign**, **malignant**, and **normal**.

🧠 Built with **Python**, **Keras**, **TensorFlow**, and **Matplotlib**, this notebook is part of a deep learning exploration on medical image classification.

---

## 📁 Dataset

The dataset consists of labeled breast ultrasound images categorized into:
- `benign`
- `malignant`
- `normal`

> The dataset is organized and stored locally within this repository.

---

## 📌 Objectives

- Preprocess and prepare image data.
- Train an **ANN** model as a baseline.
- Train a **CNN** model for improved performance.
- Compare model accuracy, loss, and predictions.
- Visualize results with graphs and confusion matrices.

---

## 🛠️ Technologies Used

- Python 3.x  
- NumPy, Matplotlib  
- TensorFlow & Keras  
- Scikit-learn  
- OpenCV  

---

## 🧪 Models Implemented

### 🔹 Artificial Neural Network (ANN)
- Flattened image input  
- Dense layers with ReLU  
- Softmax output layer  
- Works as a baseline model  

### 🔹 Convolutional Neural Network (CNN)
- Conv2D + MaxPooling layers  
- ReLU activation  
- Dense + Dropout layers  
- Softmax output layer  
- Better performance on image data  

---

## 📊 Results

| Model | Training Accuracy | Validation Accuracy |
|-------|-------------------|---------------------|
| ANN   | ~70–80%           | Moderate            |
| CNN   | ~90–95%           | High Accuracy       |

> CNN outperformed ANN in handling raw image data due to its ability to learn spatial features.

---

## 📷 Visualizations

- Accuracy & Loss Plots  
- Confusion Matrix  
- Sample Predictions  

---

## 🧪 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-ann-cnn.git
   cd breast-cancer-ann-cnn
